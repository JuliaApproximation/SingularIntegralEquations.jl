

##
# Represent a binary hierarchical Domain, Space, and Fun
##

mutable struct HierarchicalDomain{S,T,HS} <: ApproxFun.UnivariateDomain{T}
    data::HS
    HierarchicalDomain{S,T,HS}(data::HS) where {S,T,HS} = new{S,T,HS}(data)
end

mutable struct HierarchicalSpace{S,T,HS} <: ApproxFun.Space{ApproxFun.AnyDomain,T}
    data::HS
    HierarchicalSpace{S,T,HS}(data::HS) where {S,T,HS} = new{S,T,HS}(data)
end

mutable struct HierarchicalFun{S,T,HS}
    data::HS
    HierarchicalFun{S,T,HS}(data::HS) where {S,T,HS} = new{S,T,HS}(data)
end


for (HDSF,etyp) in ((:HierarchicalDomain,:eltype),(:HierarchicalSpace,:rangetype),(:HierarchicalFun,:eltype))
    @eval begin
        export $HDSF

        $HDSF(data::Tuple{S1,S2}) where {S1,S2} = $HDSF{promote_type(S1,S2),promote_type($etyp(S1),$etyp(S2)),Tuple{S1,S2}}(data)
        $HDSF(data::Tuple{$HDSF{S1,T1,HS1},$HDSF{S2,T2,HS2}}) where {S1,S2,T1,T2,HS1,HS2} = $HDSF{promote_type(S1,S2),promote_type($etyp(S1),$etyp(S2),T1,T2),Tuple{$HDSF{S1,T1,HS1},$HDSF{S2,T2,HS2}}}(data)
        $HDSF(data::Tuple{S1,$HDSF{S,T,HS}}) where {S,S1,T,HS} = $HDSF{promote_type(S,S1),promote_type($etyp(S),$etyp(S1),T),Tuple{S1,$HDSF{S,T,HS}}}(data)
        $HDSF(data::Tuple{$HDSF{S,T,HS},S1}) where {S,S1,T,HS} = $HDSF{promote_type(S,S1),promote_type($etyp(S),$etyp(S1),T),Tuple{$HDSF{S,T,HS},S1}}(data)
        collectdata(H::$HDSF{S,T,NTuple{2,S}}) where {S,T} = collect(data(H))
        collectdata(H::$HDSF{S,T,Tuple{S,$HDSF{S,T,HS}}}) where {S,T,HS} = vcat(H.data[1],collectdata(H.data[2]))
        collectdata(H::$HDSF{S,T,Tuple{$HDSF{S,T,HS},S}}) where {S,T,HS} = vcat(collectdata(H.data[1]),H.data[2])

        $HDSF(data::Vector) = $HDSF(data,round(Int,log2(length(data))))

        function $HDSF(data::Vector,n::Int)
            @assert length(data) == 2^n
            if n == 1
                return $HDSF(tuple(data...))
            elseif n ≥ 2
                return $HDSF(($HDSF(data[1:2^(n-1)],n-1),$HDSF(data[1+2^(n-1):end],n-1)))
            end
        end

        Base.similar(H::$HDSF) = $HDSF(map(similar,data(H)))
        Base.similar(H::$HDSF{SS,V,T}, S) where {SS,V,T} = $HDSF(map(A->similar(A,S),data(H)))

        data(H::$HDSF) = H.data

        nlevels(H::$HDSF) = 1+mapreduce(nlevels,max,data(H))

        partition(H::$HDSF) = data(H)

        function partition(VH::Vector{H}) where H<:$HDSF
            n = length(VH)
            H11,H12 = partition(VH[1])
            H1,H2 = fill(H11,n),fill(H12,n)
            for i=1:n
                H1[i],H2[i] = partition(VH[i])
            end
            H1,H2
        end

        function collectdata(H::$HDSF{S}) where S
            ret = S[]
            append!(ret,mapreduce(collectdata,vcat,data(H)))
            ret
        end

        convert(::Type{$HDSF{S,T,HS}},M::$HDSF{S,T,HS}) where {S,T,HS} =
            M
        convert(::Type{$HDSF{S,T,HS}},M::$HDSF) where {S,T,HS} =
            $HDSF{S,T,HS}(convert(Vector{S},collectdata(M)))
        Base.promote_rule(::Type{$HDSF{S,T,HS}},::Type{$HDSF{SS,TT,HSS}}) where {S,T,HS,SS,TT,HSS} =
            $HDSF{promote_type(S,SS),promote_type(T,TT),promote_type(HS,HSS)}
    end
end


domain(H::HierarchicalDomain) = H
domain(H::HierarchicalSpace) = HierarchicalDomain(map(domain,data(H)))
space(H::HierarchicalFun) = HierarchicalSpace(map(space,data(H)))

ApproxFun.domaindimension(H::HierarchicalSpace) = 1
Space(H::HierarchicalDomain) = HierarchicalSpace(map(Space,data(H)))

PiecewiseSpace(H::HierarchicalSpace) = PiecewiseSpace(map(PiecewiseSpace,data(H))...)



Fun(f::Function,H::HierarchicalDomain) = HierarchicalFun((Fun(f,H.data[1]),Fun(f,H.data[2])))
Fun(f::Function,H::HierarchicalSpace) = HierarchicalFun((Fun(f,H.data[1]),Fun(f,H.data[2])))


# algebra

for op in (:+,:-)
    @eval begin
        $op(H::HierarchicalFun) = HierarchicalFun(map($op,data(H)))
        $op(H::HierarchicalFun,a::Number) = HierarchicalFun(($op(H.data[1],a),$op(H.data[2],a)))
        $op(a::Number,H::HierarchicalFun) = HierarchicalFun(($op(a,H.data[1]),$op(a,H.data[2])))

        $op(H::HierarchicalFun,J::HierarchicalFun) = HierarchicalFun(map($op,data(H),data(J)))
        $op(H::HierarchicalFun,J::Fun) = $op(J,full(H))
    end
end

*(H::HierarchicalFun,a::Number) = HierarchicalFun((H.data[1]*a,H.data[2]*a))
*(a::Number,H::HierarchicalFun) = H*a

for op in (:linebilinearform,:bilinearform)
    @eval begin
        $op(H::HierarchicalFun,J::HierarchicalFun) = $op(H.data[1],J.data[1])+$op(H.data[2],J.data[2])
        $op(H::HierarchicalFun,J::Fun{S}) where {S<:PiecewiseSpace} =
            sum(map($op,collectdata(H),components(J)))
        $op(J::Fun{S},H::HierarchicalFun) where {S<:PiecewiseSpace} =
            sum(map($op,components(J),collectdata(H)))
    end
end

Base.cumsum(H::HierarchicalFun) = HierarchicalFun((cumsum(H.data[1]),sum(H.data[1])+cumsum(H.data[2])))
Base.conj!(H::HierarchicalFun) = (map(conj!,data(H));H)
Base.copy!(H::HierarchicalFun,J::HierarchicalFun) = (map(copy!,data(H),data(J));H)

for op in (:(Base.zero),:(Base.ones),:(Base.abs),:(Base.abs2),:(Base.conj),:(Base.copy))
    @eval begin
        $op(H::HierarchicalFun) = HierarchicalFun(map($op,data(H)))
    end
end

# Formatted operations

for (op,opformatted) in ((:+,:⊕),(:-,:⊖))
    @eval begin
        function $opformatted(H::HierarchicalFun{S,T,Tuple{HS1,HS2}},J::Fun{P}) where {S,T,HS1<:HierarchicalFun,HS2<:HierarchicalFun,P<:PiecewiseSpace}
            H1,H2 = partition(H)
            n1,n2 = length(PiecewiseSpace(space(H1))),length(PiecewiseSpace(space(H2)))
            return HierarchicalFun(($opformatted(H1,Fun(components(J)[1:n1],PiecewiseSpace)),
                                $opformatted(H2,Fun(components(J)[1+n1:n1+n2],PiecewiseSpace))))
        end
        function $opformatted(H::HierarchicalFun{S,T,Tuple{HS1,HS2}},J::Fun{P}) where {S,T,HS1<:Fun,HS2<:HierarchicalFun,P<:PiecewiseSpace}
            H1,H2 = partition(H)
            n1,n2 = 1,length(PiecewiseSpace(space(H2)))
            return HierarchicalFun(($op(H1,Fun(components(J)[1:n1],PiecewiseSpace)),
                                    $opformatted(H2,Fun(components(J)[1+n1:n1+n2],PiecewiseSpace))))
        end
        function $opformatted(H::HierarchicalFun{S,T,Tuple{HS1,HS2}},J::Fun{P}) where {S,T,HS1<:HierarchicalFun,HS2<:Fun,P<:PiecewiseSpace}
            H1,H2 = partition(H)
            n1,n2 = length(PiecewiseSpace(space(H1))),1
            return HierarchicalFun(($opformatted(H1,Fun(components(J)[1:n1],PiecewiseSpace)),
                                    $op(H2,Fun(components(J)[1+n1:n1+n2],PiecewiseSpace))))
        end
        function $opformatted(H::HierarchicalFun{S,T,Tuple{HS1,HS2}},J::Fun{P}) where {S,T,HS1<:Fun,HS2<:Fun,P<:PiecewiseSpace}
            H1,H2 = partition(H)
            n1,n2 = 1,1
            return HierarchicalFun(($op(H1,Fun(components(J)[1:n1],PiecewiseSpace)),
                                    $op(H2,Fun(components(J)[1+n1:n1+n2],PiecewiseSpace))))
        end

        function $opformatted(J::Fun{P},H::HierarchicalFun{S,T,Tuple{HS1,HS2}}) where {S,T,HS1<:HierarchicalFun,HS2<:HierarchicalFun,P<:PiecewiseSpace}
            H1,H2 = partition(H)
            n1,n2 = length(PiecewiseSpace(space(H1))),length(PiecewiseSpace(space(H2)))
            return HierarchicalFun(($opformatted(Fun(components(J)[1:n1],PiecewiseSpace),H1),
                                    $opformatted(Fun(components(J)[1+n1:n1+n2],PiecewiseSpace),H2)))
        end
        function $opformatted(J::Fun{P},H::HierarchicalFun{S,T,Tuple{HS1,HS2}}) where {S,T,HS1<:Fun,HS2<:HierarchicalFun,P<:PiecewiseSpace}
            H1,H2 = partition(H)
            n1,n2 = 1,length(PiecewiseSpace(space(H2)))
            return HierarchicalFun(($op(Fun(components(J)[1:n1],PiecewiseSpace),H1),
                                    $opformatted(Fun(components(J)[1+n1:n1+n2],PiecewiseSpace),H2)))
        end
        function $opformatted(J::Fun{P},H::HierarchicalFun{S,T,Tuple{HS1,HS2}}) where {S,T,HS1<:HierarchicalFun,HS2<:Fun,P<:PiecewiseSpace}
            H1,H2 = partition(H)
            n1,n2 = length(PiecewiseSpace(space(H1))),1
            return HierarchicalFun(($opformatted(Fun(components(J)[1:n1],PiecewiseSpace),H1),
                                    $op(Fun(components(J)[1+n1:n1+n2],PiecewiseSpace),H2)))
        end
        function $opformatted(J::Fun{P},H::HierarchicalFun{S,T,Tuple{HS1,HS2}}) where {S,T,HS1<:Fun,HS2<:Fun,P<:PiecewiseSpace}
            H1,H2 = partition(H)
            n1,n2 = 1,1
            return HierarchicalFun(($op(Fun(components(J)[1:n1],PiecewiseSpace),H1),
                                    $op(Fun(components(J)[1+n1:n1+n2],PiecewiseSpace),H2)))
        end
    end
end
