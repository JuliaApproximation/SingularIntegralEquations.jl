using ApproxFun, SingularIntegralEquations, Test
import ApproxFunBase: choosedomainspace, promotedomainspace, ConstantSpace, interlace,
                        testraggedbelowoperator, testblockbandedoperator, blocklengths

@testset "Ideal Fluid Flow" begin
    k = 50
    Γ = Segment(0.,1+0.5im)
    z = Fun(Γ)
    α = exp(-π*k/50im)

    @test choosedomainspace(Hilbert(),z) == JacobiWeight(-0.5,-0.5,ChebyshevDirichlet{1,1}(Γ))

    Ai=[1 Hilbert()]

    @test choosedomainspace(Ai.ops[1],space(z)) isa ConstantSpace

    S = choosedomainspace(Ai,space(z))
    @test component(S,1) isa ConstantSpace
    @test component(S,2) isa JacobiWeight{ChebyshevDirichlet{1,1,Segment{Complex{Float64}},Float64},
                                                            Segment{Complex{Float64}},Float64,Float64}




    AiS = promotedomainspace(Ai,S)
    @test domainspace(AiS) == S

    S = JacobiWeight(0.5,0.5,Ultraspherical(1,Γ))
    @time c,ui = [1 Hilbert(S)]\[imag(α*z)]

    u =(x,y) -> α*(x+im*y)+2cauchy(ui,x+im*y)

    @test u(0.1,0.2) ≈ 0.039532462109794025-0.3188804984805561im # empirical


    k = 227;
    Γ = 0.5+exp(im*Segment(0.1,-42))
    z = Fun(Γ)
    α = exp(-k/50im)

    Ai = [1 PseudoHilbert()]

    @test isa(choosedomainspace(Ai.ops[1],space(z)),ConstantSpace)
    @test isa(component(choosedomainspace(Ai,space(z)),1),ConstantSpace)
    @test isa(component(choosedomainspace(Ai,space(z)),2),
                JacobiWeight{ChebyshevDirichlet{1,1,Arc{Float64,Float64,Complex{Float64}},Float64},
                                                                        Arc{Float64,Float64,Complex{Float64}},Float64})

    S = choosedomainspace(Ai,space(z))
    AiS = promotedomainspace(Ai,S)
    @test domainspace(AiS) == S


    k = 227;
    Γ = 0.5+exp(im*Segment(0.1,-42))
    z = Fun(Γ)
    α = exp(-k/50im)
    S = JacobiWeight(0.5,0.5,Γ)
    @time c,ui = [1 PseudoHilbert(S)]\[imag(α*z)]


    u = (x,y) -> α*(x+im*y)+2pseudocauchy(ui,x+im*y)

    @test u(0.1,0.2) ≈ 0.6063720775017964 - 0.6382733554119975im # empirical



    Γ = Circle()
    z = Fun(Fourier(Γ))


    Ai = [0 DefiniteLineIntegral();
          1 real(Hilbert())]



    S = choosedomainspace(Ai,space([Fun(0.);z]))

    @test isa(component(S,1),ConstantSpace)
    @test component(S,2) == Fourier(Γ)

    @test domainspace(promotedomainspace(Ai,S)) == S


    A=promotedomainspace(Ai,S)

    testraggedbelowoperator(A)


    k=239;
    α=exp(-k/45im)

    c,ui=[0 DefiniteLineIntegral();
          1 real(Hilbert())]\[Fun(0.);imag(α*z)]


    u =(x,y)->α*(x+im*y)+2cauchy(ui,x+im*y)
    @test u(2.,1.1) ≈ 2.426592437403252-0.8340542386599383im



    ## Curve
    Γ=Curve(Fun(x->exp(0.8im)*(x+x^2-1+im*(x-4x^3+x^4)/6)))
        z=Fun(Γ)
        α=im
        S=JacobiWeight(0.5,0.5,Γ)
        @time c,ui=[1 real(Hilbert(S))]\[imag(α*z)]

    u=(x,y)->α*(x+im*y)+2cauchy(ui,x+im*y)

    @test u(0.1,0.2) ≈ (-1.1657816742288978-0.21306668168680534im)



    ## 2 intervals
    Γ=Segment(-1.,-0.5) ∪ Segment(-0.3,1.)
    z=Fun(Γ)

    S=PiecewiseSpace(map(d->JacobiWeight(0.5,0.5,Ultraspherical(1,d)),components(Γ)))


    k=114;
        α=exp(k/50*im)

    Ai = [ones(component(Γ,1))+zeros(component(Γ,2)) zeros(component(Γ,1))+ones(component(Γ,2)) Hilbert(S)]
    testblockbandedoperator(Ai)



    QR = qrfact(Ai)
          resizedata!(QR, :, Block(10)) # check for bug in resizedata!


    @time a,b,ui = [ones(component(Γ,1))+zeros(component(Γ,2)) zeros(component(Γ,1))+ones(component(Γ,2)) Hilbert(S)] \ [imag(α*z)]

    u=(x,y)->α*(x+im*y)+2cauchy(ui,x+im*y)

    @test u(0.1,0.2) ≈ (-0.5762722129639637 + 0.04348367760228282im) # empirical
end
