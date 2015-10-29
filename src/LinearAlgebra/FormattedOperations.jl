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

        $opformatted(a::Number,L::LowRankMatrix) = $opformatted(LowRankMatrix(a,size(L)...),L)
        $opformatted(L::LowRankMatrix,a::Number) = $opformatted(L,LowRankMatrix(a,size(L)...))

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
