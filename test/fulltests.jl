using ApproxFun, SingularIntegralEquations, Test
    import ApproxFun: testbandedoperator, testraggedbelowoperator, testblockbandedoperator

include("runtests.jl")

@testset "Full" begin
    ## Memory CurveTests

    d = exp(im*Interval(0.1,0.2))
    x = Fun(d)
    w = 1/(sqrt(abs(first(d)-x))*sqrt(abs(last(d)-x)))
    testbandedoperator(Hilbert(space(w)))


    ## 3 domain ideal fluid flow

    Γ = Segment(-im,1.0-im) ∪ Curve(Fun(x->exp(0.8im)*(x+x^2-1+im*(x-4x^3+x^4)/6))) ∪ Circle(2.0,0.2)
        z = Fun(Γ)

    S = PiecewiseSpace(map(d->isa(d,Circle) ? Fourier(d) : JacobiWeight(0.5,0.5,Ultraspherical(1,d)),components(Γ)))
    H = Hilbert(S)
    testblockbandedoperator(H)

    #  TODO: fix testraggedbelowoperator(H)

    Ai = [Operator(Fun(ones(component(Γ,1)),Γ)) Fun(ones(component(Γ,2)),Γ) Fun(ones(component(Γ,3)),Γ) real(H)]

    @test ApproxFun.israggedbelow(Ai)
    @test ApproxFun.israggedbelow(Ai.ops[4])
    @test ApproxFun.israggedbelow(Ai.ops[4].op)



    B=ApproxFun.SpaceOperator(ApproxFun.BasisFunctional(3),S,ApproxFun.ConstantSpace(Float64))

    Ai=[Operator(0)                 0                 0                 B;
        Fun(ones(component(Γ,1)),Γ) Fun(ones(component(Γ,2)),Γ) Fun(ones(component(Γ,3)),Γ) real(H)]


    @time testblockbandedoperator(Ai)

    k=114;
        α=exp(k/50*im)
        @time a,b,c,ui= Ai \ [0; imag(α*z)]

    u =(x,y)->α*(x+im*y)+2cauchy(ui,x+im*y)
    @test u(1.1,0.2) ≈ (-0.8290718508107162+0.511097153754im)
end

println("Example Tests")
include("ExamplesTest.jl")


include("WienerHopfTest.jl")

include("IdealFluidFlowTest.jl")
