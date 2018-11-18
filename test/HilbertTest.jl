using Test, ApproxFun, DomainSets, SingularIntegralEquations, LinearAlgebra
    import ApproxFun: ∞, testbandedoperator, testfunctional, testblockbandedoperator, testraggedbelowoperator,
                        setcanonicaldomain
    import SingularIntegralEquations: testsies, ⁺, ⁻, mobius, joukowskyinverse, sqrtx2, Directed

@testset "Hilbert" begin
    for d in (ChebyshevInterval(), (-2)..(-1), Segment(im,1))
        testsies(d)
    end

    S=JacobiWeight(0.5,0.5,Ultraspherical(1,-2 .. -1))
    f=Fun(x->exp(x)*sqrt(x+2)*sqrt(-1-x),S)
    @test logkernel(f,-1.2) ≈ -0.05044654410790341  # Mathematica
    @test hilbert(f,-1.2) ≈ -0.057515957831535571  # Mathematica


    S=JacobiWeight(-0.5,-0.5,Chebyshev(-2 .. -1))
    @test rangespace(SingularIntegral(S,0))==Chebyshev(-2 .. -1)
    f=Fun(x->exp(x)/(sqrt(x+2)*sqrt(-1-x)),S)
    @test logkernel(f,-1.2) ≈ -0.39563660592242765  # Mathematica
    @test hilbert(f,-1.2) ≈ 0.26527878405434321204  # Mathematica


    x=Fun(identity)
    f=(exp(x)/(sqrt(1-x)*sqrt(x+1)))
    @test (Hilbert(f|>space,0)*f)(.1) ≈ (-0.8545003781055088)
    @test (Hilbert(0)*f)(.1) ≈ (-0.8545003781055088)
    @test (Hilbert()*f)(.1) ≈ 1.1404096104609646386

    x=Fun(identity,-1..2)
    f=(exp(x)/(sqrt(2-x)*sqrt(x+1)))
    @test (Hilbert(f|>space,0)*f)(.1) ≈ 0.49127801561694168644
    @test (Hilbert(0)*f)(.1) ≈ 0.49127801561694168644
    @test (Hilbert()*f)(.1) ≈ 1.6649936695644078289



    x=Fun(identity)
    f=(exp(x)*(sqrt(1-x)*sqrt(x+1)))
    @test (Hilbert()*f)(.1) ≈ 0.43723982258866913063

    x=Fun(identity,-1..2)
    f=(exp(x)*(sqrt(2-x)*sqrt(x+1)))
    @test (Hilbert()*f)(.1) ≈ 2.1380903070701673244

    x=Fun(identity)
    d=domain(x)
    w=1/sqrt(1-x^2)
    H=Hilbert(d)
    B=ldirichlet(d)

    for a in [sqrt(sqrt(5)-2)/2,1.,10.]
        L=H[w]+1/a/sqrt(1+a^2)*x
        u=[B,L]\[1.,0]
        usol = (1+a^2)/(x^2+a^2)
        @test norm(u-usol) <= eps(1000/a)
    end


    x = Fun(identity)
    w = 1/sqrt(1-x^2)
    H = Hilbert(space(w))

    testbandedoperator(H[w])

    @test  (H[w]*exp(x))(.1) ≈ hilbert(w*exp(x))(.1)


    x = Fun(identity)
    w = sqrt(1-x^2)
    H = Hilbert(space(w))

    testbandedoperator(H[w])

    @test (H[w]*exp(x))(.1) ≈ hilbert(w*exp(x))(.1)


    @testset "Stieltjes" begin
        ds1 = JacobiWeight(-0.5,-0.5,ApproxFun.ChebyshevDirichlet{1,1}())
        ds2 = JacobiWeight(-.5,-.5,Chebyshev())
        rs = Chebyshev(Segment(2,4+3im))
        f1 = Fun(x->exp(x)/sqrt(1-x^2),ds1)
        f2 = Fun(x->exp(x)/sqrt(1-x^2),ds2)
        S = Stieltjes(ds1,rs)

        testbandedoperator(S)

        z = 3+1.5im
        @test (S*f1)(z) ≈ stieltjes(f2,z) #val,err = quadgk(x->f1(x)./(z-x),-1.,1.)
        # Operator 1.1589646343327578 - 0.7273679005911196im
        # Function 1.1589646343327455 - 0.7273679005911283im


        ds1 = JacobiWeight(.5,.5,Ultraspherical(1))
        ds2 = JacobiWeight(.5,.5,Chebyshev())
        rs = Chebyshev(2..4)
        f1 = Fun(x->exp(x)*sqrt(1-x^2),ds1)
        f2 = Fun(x->exp(x)*sqrt(1-x^2),ds2)
        S = Stieltjes(ds1,rs)

        testbandedoperator(S)

        z = 3.
        @test (S*f1)(z) ≈ stieltjes(f2,z) #val,err = quadgk(x->f1(x)./(z-x),-1.,1.;reltol=eps())
        # Operator 0.6616422557285478 + 0.0im
        # Function 0.661642255728541 - 0.0im
    end
    @testset "Stieltjes integral" begin
        ds1 = JacobiWeight(-.5,-.5,ApproxFun.ChebyshevDirichlet{1,1}())
        ds2 = JacobiWeight(-.5,-.5,Chebyshev())
        rs = Chebyshev(2..4)
        f1 = Fun(x->exp(x)/sqrt(1-x^2),ds1)
        f2 = Fun(x->exp(x)/sqrt(1-x^2),ds2)
        S = Stieltjes(ds1,rs,0)

        testbandedoperator(S)

        z = 3.
        @test (S*f1)(z) ≈ SingularIntegralEquations.stieltjesintegral(f2,z)
        # Operator 3.6322473044237698 + 0.0im
        # Function 3.6322473044237515

        ds1 = JacobiWeight(.5,.5,Ultraspherical(1))
        ds2 = JacobiWeight(.5,.5,Chebyshev())
        rs = Chebyshev(2.0..4.0)
        f1 = Fun(x->exp(x)*sqrt(1-x^2),ds1)
        f2 = Fun(x->exp(x)*sqrt(1-x^2),ds2)
        S = Stieltjes(ds1,rs,0)

        testbandedoperator(S)

        z = 3.0
        @test (S*f1)(z) ≈ SingularIntegralEquations.stieltjesintegral(f2,z)
        # Operator 1.7772163062194861 + 0.0im
        # Function 1.7772163062194637
    end

    @testset "Circle Hilbert" begin
        H = Hilbert(Fourier(Circle()))
        testbandedoperator(H)
        @test bandwidths(H) == (1,1)
    end

    @testset "Cauchy" begin
        f2=Fun(sech,PeriodicLine())

        z = 1 + im
        @test cauchy(f2,1+im) ≈ (0.23294739894134472 + 0.10998776661109881im ) # Mathematica
        @test cauchy(f2,1-im) ≈ (-0.23294739894134472 + 0.10998776661109881im )


        f=Fun(sech,Line())
        @test cauchy(f,1+im) ≈ cauchy(f2,1+im)


        f=Fun(z->exp(exp(0.1im)*z+1/z),Laurent(Circle()))

        @test hilbert(f,exp(0.2im)) ≈ hilbert(Fun(f,Fourier),exp(0.2im))
        @test hilbert(f,exp(0.2im)) ≈ -hilbert(reverseorientation(f),exp(0.2im))
        @test hilbert(f,exp(0.2im)) ≈ -hilbert(reverseorientation(Fun(f,Fourier)),exp(0.2im))


        @test cauchy(f,0.5exp(0.2im)) ≈ cauchy(Fun(f,Fourier),0.5exp(0.2im))
        @test cauchy(f,0.5exp(0.2im)) ≈ -cauchy(reverseorientation(f),0.5exp(0.2im))
        @test cauchy(f,0.5exp(0.2im)) ≈ -cauchy(reverseorientation(Fun(f,Fourier)),0.5exp(0.2im))
        @test cauchy(f,0.5exp(0.2im)) ≈ (OffHilbert(space(f),Laurent(Circle(0.5)))*f)(0.5exp(0.2im))/(2im)

        @time testbandedoperator(OffHilbert(space(f),Laurent(Circle(0.5))))

        f=Fun(z->exp(exp(0.1im)*z+1/(z-1.)),Laurent(Circle(1.,0.5)))
        @test cauchy(f,0.5exp(0.2im)) ≈ cauchy(Fun(f,Fourier),0.5exp(0.2im))
        @test cauchy(f,0.5exp(0.2im)) ≈ -cauchy(reverseorientation(f),0.5exp(0.2im))
        @test cauchy(f,0.5exp(0.2im)) ≈ -cauchy(reverseorientation(Fun(f,Fourier)),0.5exp(0.2im))


        @time testbandedoperator(Hilbert(Laurent(Circle())))
        @time testbandedoperator(Hilbert(Fourier(Circle())))
    end

    @testset "Two circle test 1" begin
        Γ=Circle() ∪ Circle(0.5)
        f=Fun([Fun(z->z^(-1),component(Γ,1)),Fun(z->z,component(Γ,2))],PiecewiseSpace)
        A=I-(f-Fun(one,space(f)))*Cauchy(-1)

        S=ApproxFun.choosedomainspace(A,(f-Fun(one,space(f))))
        AS=ApproxFun.promotedomainspace(A,S)


        @time testblockbandedoperator(AS)


        u=A\(f-Fun(one,space(f)))


        @test 1+cauchy(u,.1) ≈ 1
        @test 1+cauchy(u,.8) ≈ 1/0.8
        @test 1+cauchy(u,2.) ≈ 1
    end

    @testset "Two circle test 2" begin
        c1=0.5+0.1;r1=3.;
        c2=-0.1+.2im;r2=0.3;
        d1=Circle(c1,r1)
        d2=Circle(c2,r2)
        z=Fun(identity,d2);
        C=Cauchy(Space(d1),Space(d2))

        @time testbandedoperator(C)

        @test norm((C*Fun(exp,d1)-Fun(exp,d2)).coefficients)<100eps()

        C2=Cauchy(Space(d2),Space(d1))
        @time testbandedoperator(C2)

        @test norm((C2*Fun(z->exp(1/z)-1,d2)+Fun(z->exp(1/z)-1,d1)).coefficients)<100000eps()
    end

    @testset "Two circle test 3" begin
        c1=0.1+0.1im;r1=.4;
        c2=-2+0.2im;r2=0.3;
        d1=Circle(c1,r1)
        d2=Circle(c2,r2)
        @test norm((Cauchy(d1,d2)*Fun(z->exp(1/z)-1,d1)+Fun(z->exp(1/z)-1,d2)).coefficients)<2000eps()

        @time testbandedoperator(Cauchy(d1,d2))
    end

    @testset "Legendre" begin
        #Legendre uses FastGaussQuadrature
        f=Fun(exp,Legendre())


        ω=2.
        d=Segment(0.5im,30.0im/ω)
        x=Fun(identity,Legendre(d))
        @test cauchy(exp(im*ω*x),1+im) ≈ (-0.025430235512791915911 + 0.0016246822285867573678im)
    end

    @testset "Arc" begin
        a=Arc(0.,1.,0.,π/2)
        ζ=Fun(identity,a)
        f=Fun(exp,a)*sqrt(abs((ζ-1)*(ζ-im)))

        z=exp(.1im)
        @test hilbert(f,z) ≈ im*(cauchy(f,(z)⁺)+cauchy(f,(z)⁻))
    end

    @testset "Functional" begin
        z = 0.1 + 0.2im
        x = Fun(identity)
        f = exp(x)*sqrt(1-x^2)

        testfunctional(Stieltjes(space(f),z))

        @test Stieltjes(space(f),z)*f ≈ stieltjes(f,z)

        a=Arc(0.,1.,0.,π/2)
        ζ=Fun(identity,a)
        f=Fun(exp,a)*sqrt(abs((ζ-1)*(ζ-im)))
        H=Hilbert()
        z=exp(.1im)

        @test (H*f)(z) ≈ hilbert(f,z)

        testbandedoperator(Hilbert(space(f)))
    end

    @testset "Piecewise singularintegral" begin
        P₀ = legendre(0,Segment(0,1)) + legendre(0,Segment(0,im))
        s = Fun(domain(P₀))
        let z=2+im
            @test singularintegral(1,P₀, z) ≈ linesum(P₀/(z-s))/π
        end
    end

    @testset "LogKernel is real" begin
        Γ = Interval(-1 , 0) , Interval(1,2)
        S = ∪(JacobiWeight.(-0.5,-0.5,Chebyshev.(Γ))...)
        @test OffSingularIntegral(S, ConstantSpace(ApproxFun.Point(3.0+im)), 0) isa Operator{Float64}
    end


    @testset "Ideal fluid flow" begin
        a = 0.3
        θ = 1.3
        Γ = Segment(-1,-a) ∪ Segment(a, 1)

        x = Fun(Γ)
        sp = PiecewiseSpace(JacobiWeight.(0.5,0.5,components(Γ))...)
        H = Hilbert(sp)


        o₁ = Fun(x -> -1 ≤ x ≤ -a ? 1 : 0, Γ )
        o₂ = Fun(x -> a ≤ x ≤ 1 ? 1 : 0, Γ )

        a, b, f = [o₁ o₂ H] \ [-2x*sin(θ)]

        @test Number(a) ≈ -Number(b) ≈ 1.151966678272007
        @test f(0.5) ≈ 0.6384624727021831
    end
end
