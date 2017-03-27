

##
# Represent a binary hierarchical Domain, Space, and Fun
##

type HierarchicalDomain{S,T,HS} <: ApproxFun.UnivariateDomain{T}
    data::HS
    (::Type{HierarchicalDomain{S,T,HS}}){S,T,HS}(data::HS) = new{S,T,HS}(data)
end

type HierarchicalSpace{S,T,HS} <: ApproxFun.UnivariateSpace{T,ApproxFun.AnyDomain}
    data::HS
    (::Type{HierarchicalSpace{S,T,HS}}){S,T,HS}(data::HS) = new{S,T,HS}(data)
end

type HierarchicalFun{S,T,HS}
    data::HS
    (::Type{HierarchicalFun{S,T,HS}}){S,T,HS}(data::HS) = new{S,T,HS}(data)
end


for HDSF in (:HierarchicalDomain,:HierarchicalSpace,:HierarchicalFun)
    @eval begin
        export $HDSF

        if false #$HDSF == HierarchicalSpace
            $HDSF{S1,S2}(data::Tuple{S1,S2}) = $HDSF{promote_type(S1,S2),promote_type(eltype(S1),eltype(S2)),Tuple{S1,S2},typeof(HierarchicalDomain((domain(data[1]),domain(data[2]))))}(data)
            $HDSF{S1,S2,T1,T2,HS1,HS2,D1,D2}(data::Tuple{$HDSF{S1,T1,HS1,D1},$HDSF{S2,T2,HS2,D2}}) = $HDSF{promote_type(S1,S2),promote_type(eltype(S1),eltype(S2),T1,T2),Tuple{$HDSF{S1,T1,HS1,D1},$HDSF{S2,T2,HS2,D2}},typeof(HierarchicalDomain((domain(data[1]),domain(data[2]))))}(data)
            $HDSF{S,S1,T,HS,D}(data::Tuple{S1,$HDSF{S,T,HS,D}}) = $HDSF{promote_type(S,S1),promote_type(eltype(S),eltype(S1),T),Tuple{S1,$HDSF{S,T,HS,D}},typeof(HierarchicalDomain((domain(data[1]),domain(data[2]))))}(data)
            $HDSF{S,S1,T,HS,D}(data::Tuple{$HDSF{S,T,HS,D},S1}) = $HDSF{promote_type(S,S1),promote_type(eltype(S),eltype(S1),T),Tuple{$HDSF{S,T,HS,D},S1},typeof(HierarchicalDomain((domain(data[1]),domain(data[2]))))}(data)
            collectdata{S,T,D}(H::$HDSF{S,T,NTuple{2,S},D}) = collect(data(H))
            collectdata{S,T,HS,D}(H::$HDSF{S,T,Tuple{S,$HDSF{S,T,HS,D},D}}) = vcat(H.data[1],collectdata(H.data[2]))
            collectdata{S,T,HS,D}(H::$HDSF{S,T,Tuple{$HDSF{S,T,HS,D},S},D}) = vcat(collectdata(H.data[1]),H.data[2])
        else
            $HDSF{S1,S2}(data::Tuple{S1,S2}) = $HDSF{promote_type(S1,S2),promote_type(eltype(S1),eltype(S2)),Tuple{S1,S2}}(data)
            $HDSF{S1,S2,T1,T2,HS1,HS2}(data::Tuple{$HDSF{S1,T1,HS1},$HDSF{S2,T2,HS2}}) = $HDSF{promote_type(S1,S2),promote_type(eltype(S1),eltype(S2),T1,T2),Tuple{$HDSF{S1,T1,HS1},$HDSF{S2,T2,HS2}}}(data)
            $HDSF{S,S1,T,HS}(data::Tuple{S1,$HDSF{S,T,HS}}) = $HDSF{promote_type(S,S1),promote_type(eltype(S),eltype(S1),T),Tuple{S1,$HDSF{S,T,HS}}}(data)
            $HDSF{S,S1,T,HS}(data::Tuple{$HDSF{S,T,HS},S1}) = $HDSF{promote_type(S,S1),promote_type(eltype(S),eltype(S1),T),Tuple{$HDSF{S,T,HS},S1}}(data)
            collectdata{S,T}(H::$HDSF{S,T,NTuple{2,S}}) = collect(data(H))
            collectdata{S,T,HS}(H::$HDSF{S,T,Tuple{S,$HDSF{S,T,HS}}}) = vcat(H.data[1],collectdata(H.data[2]))
            collectdata{S,T,HS}(H::$HDSF{S,T,Tuple{$HDSF{S,T,HS},S}}) = vcat(collectdata(H.data[1]),H.data[2])
        end

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
        Base.similar{SS,V,T}(H::$HDSF{SS,V,T}, S) = $HDSF(map(A->similar(A,S),data(H)))

        data(H::$HDSF) = H.data

        nlevels(H::$HDSF) = 1+mapreduce(nlevels,max,data(H))

        partition(H::$HDSF) = data(H)

        function partition{H<:$HDSF}(VH::Vector{H})
            n = length(VH)
            H11,H12 = partition(VH[1])
            H1,H2 = fill(H11,n),fill(H12,n)
            for i=1:n
                H1[i],H2[i] = partition(VH[i])
            end
            H1,H2
        end

        function collectdata{S}(H::$HDSF{S})
            ret = S[]
            append!(ret,mapreduce(collectdata,vcat,data(H)))
            ret
        end

        Base.convert{S,T,HS}(::Type{$HDSF{S,T,HS}},M::$HDSF) = $HDSF(convert(Vector{S},collectdata(M)))
        Base.promote_rule{S,T,HS,SS,TT,HSS}(::Type{$HDSF{S,T,HS}},::Type{$HDSF{SS,TT,HSS}})=$HDSF{promote_type(S,SS),promote_type(T,TT),promote_type(HS,HSS)}
        Base.eltype{S,T}(::$HDSF{S,T})=T
        Base.eltype{S,T,HS}(::Type{$HDSF{S,T,HS}})=T
    end
end


domain(H::HierarchicalDomain) = H
domain(H::HierarchicalSpace) = HierarchicalDomain(map(domain,data(H)))
space(H::HierarchicalFun) = HierarchicalSpace(map(space,data(H)))

ApproxFun.basistype(H::HierarchicalSpace) = mapreduce(typeof,promote_type,data(H))
ApproxFun.domaindimension(H::HierarchicalSpace) = 1
Space(H::HierarchicalDomain) = HierarchicalSpace(map(Space,data(H)))

PiecewiseSpace(H::HierarchicalSpace) = PiecewiseSpace(map(PiecewiseSpace,data(H))...)



Fun(f::Function,H::HierarchicalDomain) = HierarchicalFun((Fun(f,H.data[1]),Fun(f,H.data[2])))
Fun(f::Function,H::HierarchicalSpace) = HierarchicalFun((Fun(f,H.data[1]),Fun(f,H.data[2])))

ApproxFun.depiece(f::Fun) = f
ApproxFun.depiece(H::HierarchicalFun) = depiece(map(depiece,data(H)))

# algebra

for op in (:+,:-,:.+,:.-,:.*)
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
        $op{S<:PiecewiseSpace}(H::HierarchicalFun,J::Fun{S}) = sum(map($op,collectdata(H),pieces(J)))
        $op{S<:PiecewiseSpace}(J::Fun{S},H::HierarchicalFun) = sum(map($op,pieces(J),collectdata(H)))
    end
end

Base.cumsum(H::HierarchicalFun) = HierarchicalFun((cumsum(H.data[1]),sum(H.data[1])+cumsum(H.data[2])))
Base.conj!(H::HierarchicalFun) = (map(conj!,data(H));H)
Base.copy!(H::HierarchicalFun,J::HierarchicalFun) = (map(copy!,data(H),data(J));H)

for op in (:(Base.zero),:(Base.ones),:(Base.abs),:(Base.abs2),:(Base.conj),:(Base.copy),:.^)
    @eval begin
        $op(H::HierarchicalFun) = HierarchicalFun(map($op,data(H)))
    end
end

# Formatted operations

for (op,opformatted) in ((:+,:⊕),(:-,:⊖))
    @eval begin
        function $opformatted{S,T,HS1<:HierarchicalFun,HS2<:HierarchicalFun,P<:PiecewiseSpace}(H::HierarchicalFun{S,T,Tuple{HS1,HS2}},J::Fun{P})
            H1,H2 = partition(H)
            n1,n2 = length(PiecewiseSpace(space(H1))),length(PiecewiseSpace(space(H2)))
            return HierarchicalFun(($opformatted(H1,depiece(pieces(J)[1:n1])),$opformatted(H2,depiece(pieces(J)[1+n1:n1+n2]))))
        end
        function $opformatted{S,T,HS1<:Fun,HS2<:HierarchicalFun,P<:PiecewiseSpace}(H::HierarchicalFun{S,T,Tuple{HS1,HS2}},J::Fun{P})
            H1,H2 = partition(H)
            n1,n2 = 1,length(PiecewiseSpace(space(H2)))
            return HierarchicalFun(($op(H1,depiece(pieces(J)[1:n1])),$opformatted(H2,depiece(pieces(J)[1+n1:n1+n2]))))
        end
        function $opformatted{S,T,HS1<:HierarchicalFun,HS2<:Fun,P<:PiecewiseSpace}(H::HierarchicalFun{S,T,Tuple{HS1,HS2}},J::Fun{P})
            H1,H2 = partition(H)
            n1,n2 = length(PiecewiseSpace(space(H1))),1
            return HierarchicalFun(($opformatted(H1,depiece(pieces(J)[1:n1])),$op(H2,depiece(pieces(J)[1+n1:n1+n2]))))
        end
        function $opformatted{S,T,HS1<:Fun,HS2<:Fun,P<:PiecewiseSpace}(H::HierarchicalFun{S,T,Tuple{HS1,HS2}},J::Fun{P})
            H1,H2 = partition(H)
            n1,n2 = 1,1
            return HierarchicalFun(($op(H1,depiece(pieces(J)[1:n1])),$op(H2,depiece(pieces(J)[1+n1:n1+n2]))))
        end

        function $opformatted{S,T,HS1<:HierarchicalFun,HS2<:HierarchicalFun,P<:PiecewiseSpace}(J::Fun{P},H::HierarchicalFun{S,T,Tuple{HS1,HS2}})
            H1,H2 = partition(H)
            n1,n2 = length(PiecewiseSpace(space(H1))),length(PiecewiseSpace(space(H2)))
            return HierarchicalFun(($opformatted(depiece(pieces(J)[1:n1]),H1),$opformatted(depiece(pieces(J)[1+n1:n1+n2]),H2)))
        end
        function $opformatted{S,T,HS1<:Fun,HS2<:HierarchicalFun,P<:PiecewiseSpace}(J::Fun{P},H::HierarchicalFun{S,T,Tuple{HS1,HS2}})
            H1,H2 = partition(H)
            n1,n2 = 1,length(PiecewiseSpace(space(H2)))
            return HierarchicalFun(($op(depiece(pieces(J)[1:n1]),H1),$opformatted(depiece(pieces(J)[1+n1:n1+n2]),H2)))
        end
        function $opformatted{S,T,HS1<:HierarchicalFun,HS2<:Fun,P<:PiecewiseSpace}(J::Fun{P},H::HierarchicalFun{S,T,Tuple{HS1,HS2}})
            H1,H2 = partition(H)
            n1,n2 = length(PiecewiseSpace(space(H1))),1
            return HierarchicalFun(($opformatted(depiece(pieces(J)[1:n1]),H1),$op(depiece(pieces(J)[1+n1:n1+n2]),H2)))
        end
        function $opformatted{S,T,HS1<:Fun,HS2<:Fun,P<:PiecewiseSpace}(J::Fun{P},H::HierarchicalFun{S,T,Tuple{HS1,HS2}})
            H1,H2 = partition(H)
            n1,n2 = 1,1
            return HierarchicalFun(($op(depiece(pieces(J)[1:n1]),H1),$op(depiece(pieces(J)[1+n1:n1+n2]),H2)))
        end
    end
end
