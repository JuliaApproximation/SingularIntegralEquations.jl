#
# Formatted operations preserve the most stringent array structure
#

import ApproxFun: ⊕
export ⊖,⊛,⊘

for (op,opformatted) in ((:+,:⊕),(:-,:⊖),(:*,:⊛),(:/,:⊘))
    @eval begin
        $opformatted(A::AbstractArray) = $op(A)
        $opformatted(A::AbstractArray,B::AbstractArray) = $op(A,B)
        $opformatted(a::Number,B::AbstractArray) = $op(a,B)
        $opformatted(A::AbstractArray,b::Number) = $op(A,b)
    end
end

for (op,opformatted) in ((:+,:⊕),(:-,:⊖))
    @eval begin

        function $opformatted(H::HierarchicalVector,J::Vector)
            @assert length(H) == length(J)
            H1,H2 = data(H)
            n1,n2 = length(H1),length(H2)
            HierarchicalVector(($opformatted(H1,J[1:n1]),$opformatted(H2,J[1+n1:n1+n2])))
        end
        $opformatted(J::Vector,H::HierarchicalVector) = $opformatted(H,J)

        $opformatted(L::LowRankMatrix,a::Number) = $opformatted(L,LowRankMatrix(a,size(L)...))
        $opformatted(a::Number,L::LowRankMatrix) = $opformatted(L,a)

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
        $opformatted(A::AbstractMatrix,L::LowRankMatrix) = $opformatted(L,A)
    end
end
