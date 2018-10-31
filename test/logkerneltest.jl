using Test, ApproxFun, SingularIntegralEquations
    import ApproxFun: ∞, testbandedoperator, testfunctional, testblockbandedoperator, testraggedbelowoperator, JacobiZ
    import SingularIntegralEquations: testsies, testsieeval, stieltjesmoment, Directed, _₂F₁, ⁺, ⁻

@testset "Log Kernel" begin
    testsieeval(Jacobi(0,0))

    a=1.0;b=2.0+im
    d=Segment(a,b)
    z=Fun(d)
    f=real(exp(z)/(sqrt(z-a)*sqrt(b-z)))
    S=space(f)
    testsies(S)


    a=1.0;b=2.0+im
    d=Segment(a,b)
    z=Fun(d)
    f=real(exp(z)*(sqrt(z-a)*sqrt(b-z)))
    S=space(f)
    testsies(S)


    a=1.0;b=2.0+im
    d=Segment(a,b)
    z=Fun(d)
    f=real(exp(z)/(sqrt(z-a)*sqrt(b-z)))
    S=JacobiWeight(-0.5,-0.5,ChebyshevDirichlet{1,1}(d))
    # TODO: move to testsies
    H=OffSingularIntegral(S,Chebyshev(3..4),0)
    @test (H*f)(3.5) ≈ logkernel(f,3.5)

    H=OffSingularIntegral(S,Chebyshev(Segment(3,4.0+im)),0)
    @test (H*f)(3.5+0.5im) ≈ logkernel(f,3.5+0.5im)


    ## Circle

    d=Circle(0.2,3.0)
    S=Fourier(d)
    ζ=Fun(d)
    f=real(ζ+1/(ζ-0.1))
    #z=0.1+0.1im;@test linesum(logabs(ζ-z)*f) ≈ logkernel(f,z)*π
    #z=5.0+0.1im;@test linesum(logabs(ζ-z)*f) ≈ logkernel(f,z)*π

    d=Circle(0.2,3.0)
    S=Fourier(d)
    H=Hilbert(S,0)
    ζ=Fun(d)
    f=real(ζ+1/(ζ-0.1))
    z=0.2+3im;@test (H*f)(z) ≈ logkernel(f,z)



    ## LogKernel for Legendre and Jacobi on intervel

    x=Fun()
    f=exp(x)



    @test isa(logkernel(f,2.0+im),Real)
    @test logkernel(f,2.0+im) ≈ sum(f*logabs(2.0+im-x))/π


    f=sqrt(1-x^2)*exp(x)
    @test logkernel(f,2.0+im) ≈ sum(f*logabs(2.0+im-x))/π

    x=Fun()
    f=(1-x)^0.1
    sp=space(f)
    @test logjacobimoment(sp.α,sp.β,2.0) ≈ sum((1-x)^sp.α*(1+x)^sp.β*logabs(2.0-x))

    @test logkernel(f,2.0) ≈ sum(f*logabs(2.0-x)/π)


    f=(1-x)^0.1*exp(x)

    @test stieltjes(f,2.0) ≈ sum(f/(2.0-x))

    @test logkernel(f,2.0) ≈ sum(f*logabs(2.0-x)/π)

    @test isa(logkernel(f,2.0+im),Real)
    @test logkernel(f,2.0+im) ≈ sum(f*logabs(2.0+im-x)/π)

    f=(1-x^2)^0.1*exp(x)
    sp=space(f)
    @test logjacobimoment(sp.α,sp.β,2.0) ≈ sum((1-x^2)^0.1*logabs(2.0-x))
    @test logkernel(f,2.0+im) ≈ sum(f*logabs(2.0+im-x)/π)


    f=(1-x)^(-0.1)*(1+x)^(-0.2)*exp(x)
    @test logkernel(f,2.0+im) ≈ sum(f*logabs(2.0+im-x)/π)
    @test isa(logkernel(f,2.0+im),Real)



    # other intervals

    x = Fun()
    f = exp(x)
    z = 2+im
    @test logkernel(f,z) ≈ linesum(f*logabs(x-z))/π
    @test logkernel(f, -10.0) ≈ linesum(f*logabs(x+10))/π



    x = Fun()
    f = real(sqrt(1-x)*exp(x))
    z = 2+im
    @test logkernel(f,z) ≈ linesum(f*logabs(x-z))/π
    @test logkernel(f, -10.0) ≈ linesum(f*logabs(x+10))/π
    @test_broken logkernel(f, 0.1) ≈ logkernel(f, 0.1+eps()im)


    x = Fun(0.1..1)
    f = real(sqrt(1-x)*exp(x))
    z = 2+im
    @test logkernel(f,z) ≈ linesum(f*logabs(x-z))/π
    @test logkernel(f, -10.0) ≈ linesum(f*logabs(x+10))/π
    @test_broken logkernel(f, 0.2) ≈ logkernel(f, 0.2+eps()im)


    ## Arc
    a=Arc(0.,1.,0.,π/2)
    testsieeval(Legendre(a);posdirection=(-1-im))




    @testset "Chebyshev singularities" begin
        for d in (ChebyshevInterval(), Interval(0,1), Segment(1+im, 2+3im))
            for k = 1:10
                u = Fun(JacobiWeight.(-0.5,-0.5,Ref(ChebyshevDirichlet{1,1}(d))), [zeros(4);1])
                x = Fun(d)
                for z in (20.0+im, 3.0)
                    ex = linesum(u*logabs(z-x))/π
                    @test logkernel(u, z) ≈ ex
                end
            end
        end

        for d in (ChebyshevInterval(), Interval(0,1))
            for k = 1:10
                u = Fun(JacobiWeight.(-0.5,-0.5,Ref(ChebyshevDirichlet{1,1}(d))), [zeros(4);1])
                x = Fun(d)
                for z in (20.0+im, 3.0)
                    ex_s = sum(u*log(z-x))
                    @test stieltjesintegral(u, z) ≈ ex_s
                end
            end
        end
    end
end
