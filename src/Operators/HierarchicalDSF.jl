

##
# Represent a binary hierarchical Domain, Space, and Fun
##

for HDSF in (:HierarchicalDomain,:HierarchicalSpace,:HierarchicalFun)
    @eval begin
        export $HDSF

        type $HDSF{S,T,HS}
            data::HS
            $HDSF(data::HS) = new(data)
        end

        $HDSF{S}(data::NTuple{2,S}) = $HDSF{S,eltype(S),NTuple{2,S}}(data)

        $HDSF{S,T,HS}(data::Tuple{S,$HDSF{S,T,HS}}) = $HDSF{S,T,Tuple{S,$HDSF{S,T,HS}}}(data)
        $HDSF{S,T,HS}(data::Tuple{$HDSF{S,T,HS},S}) = $HDSF{S,T,Tuple{$HDSF{S,T,HS},S}}(data)

        $HDSF{S,T,HS}(data::NTuple{2,$HDSF{S,T,HS}}) = $HDSF{S,T,NTuple{2,$HDSF{S,T,HS}}}(data)
        $HDSF{S,T,HS1,HS2}(data::Tuple{$HDSF{S,T,HS1},$HDSF{S,T,HS2}}) = $HDSF{S,T,Tuple{$HDSF{S,T,HS1},$HDSF{S,T,HS2}}}(data)

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

        degree(H::$HDSF) = 1+mapreduce(degree,max,data(H))

        partition(H::$HDSF) = data(H)

        collectdata{S,T}(H::$HDSF{S,T,NTuple{2,S}}) = collect(data(H))
        collectdata{S,T,HS}(H::$HDSF{S,T,Tuple{S,$HDSF{S,T,HS}}}) = vcat(H.data[1],collectdata(H.data[2]))
        collectdata{S,T,HS}(H::$HDSF{S,T,Tuple{$HDSF{S,T,HS},S}}) = vcat(collectdata(H.data[1]),H.data[2])
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

domain(H::HierarchicalSpace) = HierarchicalDomain(map(domain,data(H)))
space(H::HierarchicalFun) = HierarchicalSpace(map(space,data(H)))

Space(H::HierarchicalDomain) = HierarchicalSpace(map(Space,data(H)))

Fun(f,H::HierarchicalDomain) = HierarchicalFun((Fun(f,H.data[1]),Fun(f,H.data[2])))
Fun(f,H::HierarchicalSpace) = HierarchicalFun((Fun(f,H.data[1]),Fun(f,H.data[2])))


function partition{HF<:HierarchicalFun}(H::Vector{HF})
    n = length(H)
    H11,H12 = partition(H[1])
    H1,H2 = fill(H11,n),fill(H12,n)
    for i=1:n
        H1[i],H2[i] = partition(H[i])
    end
    H1,H2
end

ApproxFun.depiece(f::Fun) = f
ApproxFun.depiece(H::HierarchicalFun) = depiece(map(depiece,data(H)))

# algebra

for op in (:+,:-,:.+,:.-,:.*)
    @eval begin
        $op(H::HierarchicalFun) = HierarchicalFun(map($op,data(H)))
        $op(H::HierarchicalFun,a::Number) = HierarchicalFun(($op(H.data[1],a),$op(H.data[2],a)))
        $op(a::Number,H::HierarchicalFun) = $op(H,a)

        $op(H::HierarchicalFun,J::HierarchicalFun) = HierarchicalFun(map($op,data(H),data(J)))
        $op(H::HierarchicalFun,J::Fun) = $op(full(H),J)
        $op(J::Fun,H::HierarchicalFun) = $op(H,J)
    end
end

*(H::HierarchicalFun,a::Number) = HierarchicalFun((H.data[1]*a,H.data[2]*a))
*(a::Number,H::HierarchicalFun) = H*a

for op in (:(Base.dot),:dotu)
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
