#
# Formatted operations preserve the most stringent array structure
#

import ApproxFun: ⊕
export ⊖,⊛,⊘

for (op,opformatted) in ((:+,:⊕),(:-,:⊖),(:*,:⊛),(:/,:⊘))
    @eval begin
        $opformatted(A) = $op(A)
        $opformatted(A,B) = $op(A,B)
    end
end

for (op,opformatted) in ((:+,:⊕),(:-,:⊖))
    @eval begin

        function $opformatted{S,T}(H::HierarchicalVector{S},J::Vector{T})
            @assert length(H) == length(J)
            K = similar(H, promote_type(S,T))
            for i=1:length(H) K[i] = $op(H[i],J[i]) end
            K
        end
        function $opformatted{S,T}(J::Vector{S},H::HierarchicalVector{T})
            @assert length(H) == length(J)
            K = similar(H, promote_type(S,T))
            for i=1:length(H) K[i] = $op(J[i],H[i]) end
            K
        end

        $opformatted(L::LowRankMatrix,a::Number) = $opformatted(L,LowRankMatrix(a,size(L)...))
        $opformatted(a::Number,L::LowRankMatrix) = $opformatted(LowRankMatrix(a,size(L)...),L)

        function $opformatted(L::LowRankMatrix,M::LowRankMatrix)
            N = $op(L,M)
            T = eltype(N)
            QU,RU = qr(N.U)
            QV,RV = qr(N.V)
            U,Σ,V = svd(RU*RV.')
            r = refactorsvd!(U,Σ,V)
            LowRankMatrix(QU[:,1:r]*U[1:r,1:r],QV[:,1:r]*V[1:r,1:r])
        end
        $opformatted(L::LowRankMatrix,A::AbstractMatrix) = $opformatted(L,LowRankMatrix(A))
        $opformatted(A::AbstractMatrix,L::LowRankMatrix) = $opformatted(LowRankMatrix(A),L)
    end
end
