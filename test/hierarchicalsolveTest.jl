#
# test hierarchicalsolve
#
using ApproxFun, SingularIntegralEquations, Base.Test

M = 50
L = Array(LowRankMatrix{Float64},14)
[L[i] = LowRankMatrix(0.001rand(4M,2),0.001rand(4M,2)) for i=1:2]
[L[i] = LowRankMatrix(0.01rand(2M,2),0.01rand(2M,2)) for i=[3:4;9:10]]
[L[i] = LowRankMatrix(0.1rand(M,2),0.1rand(M,2)) for i=[5:8;11:14]]
D = Array(Diagonal{Float64},8)
[D[i] = Diagonal(1./collect(1:M)) for i=1:8]

H = HierarchicalMatrix(D,L)

@test rank(H) == 400
@test blockrank(H) == fill(2,8,8)+diagm(fill(48,8))
@test isfactored(H) == false

B = Array(Vector{Float64},8)
[B[i] = rand(M) for i=1:8]
b = rand(size(H,1))#HierarchicalVector(B)

full(H)\b
@time x = full(H)\b
H\b # includes compile time
H = HierarchicalMatrix(D,L)
@time xw = H\b # includes precomputation
@time xw = H\b # H.A's precomputed and factored :)

@test isfactored(H) == true

CH = cond(full(H))

@test norm(x-xw) ≤ 10CH^2*eps()
@test norm(H*x-b) ≤ 10CH*eps()
@test norm(H*xw-b) ≤ 10CH*eps()


H+H-(H-H)
@test norm(full(H-H)) < 10norm(full(H))*eps()
