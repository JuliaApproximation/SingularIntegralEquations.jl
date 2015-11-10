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

        function $opformatted(H::HierarchicalVector,J::Vector)
            @assert length(H) == length(J)
            H1,H2 = data(H)
            n1,n2 = length(H1),length(H2)
            HierarchicalVector(($opformatted(H1,J[1:n1]),$opformatted(H2,J[1+n1:n1+n2])))
        end
        function $opformatted(J::Vector,H::HierarchicalVector)
            @assert length(H) == length(J)
            H1,H2 = data(H)
            n1,n2 = length(H1),length(H2)
            HierarchicalVector(($opformatted(J[1:n1],H1),$opformatted(J[1+n1:n1+n2],H2)))
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
