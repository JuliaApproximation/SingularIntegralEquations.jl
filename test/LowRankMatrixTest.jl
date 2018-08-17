using ApproxFun, LowRankApprox, SingularIntegralEquations, Test

@testset "LowRankMatrix" begin
    L = LowRankApprox._LowRankMatrix(1 ./(1:10),1 ./(1:10.0))
    A = rand(10,10)

    @test isa(L⊕A,LowRankMatrix)

    @test rank(L+L) == 2rank(L⊕L)
    # @test rank(L-L) == 2rank(L⊖L)
    @test rank(2L+1+Matrix(L)) ≥ rank(L⊕Matrix(L))
end
