#
# test LowRankMatrix
#
using ApproxFun, SingularIntegralEquations, Base.Test

L = lrzeros(Complex{Float64},10,20)
@test size(L) == (10,20)

@test full(L) == zeros(Complex{Float64},10,20)

@test_approx_eq full(lrones(10,10)) ones(10,10)

@test rank(lreye(10,10)) == 10

L = LowRankMatrix(1./collect(1:10),1./collect(1:10.0))

x = collect(linspace(-1,1,10))

@test norm(L*x-full(L)*x) < 200eps()

@test isa(L*L',LowRankMatrix)

A = rand(10,10)

@test isa(A*L,LowRankMatrix)
@test isa(L*A,LowRankMatrix)

@test rank(L+L) == 2rank(L⊕L)
@test rank(L-L) == 2rank(L⊖L)
@test rank(2L+1+full(L)) ≥ rank(L⊕full(L))
