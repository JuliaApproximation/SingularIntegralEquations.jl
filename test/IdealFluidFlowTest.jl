using Base.Test, ApproxFun, SingularIntegralEquations, Base.Test
    import ApproxFun: choosedomainspace, promotedomainspace, ConstantSpace, interlace,
                        testraggedbelowoperator, testbandedblockoperator, blocklengths

k=50
Γ=Interval(0.,1+0.5im)
z=Fun(Γ)
α=exp(-π*k/50im)

@test ApproxFun.choosedomainspace(Hilbert(),z) == JacobiWeight(-0.5,-0.5,ChebyshevDirichlet{1,1}(Γ))

Ai=ApproxFun.interlace([1 Hilbert()])

@test isa(ApproxFun.choosedomainspace(Ai.ops[1],space(z)),ApproxFun.ConstantSpace)


S=choosedomainspace(Ai,space(z))
@test isa(S,ApproxFun.TupleSpace{Tuple{ApproxFun.ConstantSpace{ApproxFun.AnyDomain},
            ApproxFun.JacobiWeight{ApproxFun.ChebyshevDirichlet{1,1,ApproxFun.Interval{Complex{Float64}}},
                                                                    ApproxFun.Interval{Complex{Float64}}}}})



AiS=promotedomainspace(Ai,S)
@test domainspace(AiS) == S

domainspace(Ai)
S=JacobiWeight(0.5,0.5,Ultraspherical(1,Γ))
@time c,ui=[1 Hilbert(S)]\imag(α*z)

u =(x,y)->α*(x+im*y)+2cauchy(ui,x+im*y)

@test_approx_eq u(0.1,0.2) 0.039532462109794025-0.3188804984805561im # empirical


k=227;
Γ=0.5+exp(im*Interval(0.1,-42))
z=Fun(Γ)
α=exp(-k/50im)

Ai=ApproxFun.interlace([1 PseudoHilbert()])

@test isa(ApproxFun.choosedomainspace(Ai.ops[1],space(z)),ApproxFun.ConstantSpace)
@test isa(ApproxFun.choosedomainspace(Ai,space(z)),ApproxFun.TupleSpace{Tuple{ApproxFun.ConstantSpace{ApproxFun.AnyDomain},
            ApproxFun.JacobiWeight{ApproxFun.ChebyshevDirichlet{1,1,ApproxFun.Arc{Float64,Float64,Complex{Float64}}},
                                                                    ApproxFun.Arc{Float64,Float64,Complex{Float64}}}}})


S=choosedomainspace(Ai,space(z))
AiS=promotedomainspace(Ai,S)
@test domainspace(AiS) == S


k=227;
Γ=0.5+exp(im*Interval(0.1,-42))
z=Fun(Γ)
α=exp(-k/50im)
S=JacobiWeight(0.5,0.5,Γ)
@time c,ui=[1 PseudoHilbert(S)]\imag(α*z)


u=(x,y)->α*(x+im*y)+2pseudocauchy(ui,x+im*y)

@test_approx_eq u(0.1,0.2) 0.6063720775017964 - 0.6382733554119975im # empirical



Γ=Circle()
z=Fun(Fourier(Γ))


Ai=ApproxFun.interlace([0 DefiniteLineIntegral();
      1 real(Hilbert())])



S=ApproxFun.choosedomainspace(Ai,space([Fun(0.);z]))

@test isa(S[1],ApproxFun.ConstantSpace)
@test S[2] == Fourier(Γ)

@test domainspace(ApproxFun.promotedomainspace(Ai,S)) == S


A=ApproxFun.promotedomainspace(Ai,S)


@which A.ops[3][2]
testraggedbelowoperator(A)


k=239;
α=exp(-k/45im)

c,ui=[0 DefiniteLineIntegral();
      1 real(Hilbert())]\[Fun(0.);imag(α*z)]


u =(x,y)->α*(x+im*y)+2cauchy(ui,x+im*y)
@test_approx_eq u(2.,1.1)  2.426592437403252-0.8340542386599383im



## Curve
Γ=Curve(Fun(x->exp(0.8im)*(x+x^2-1+im*(x-4x^3+x^4)/6)))
    z=Fun(Γ)
    α=im
    S=JacobiWeight(0.5,0.5,Γ)
    @time c,ui=[1 real(Hilbert(S))]\imag(α*z)

u=(x,y)->α*(x+im*y)+2cauchy(ui,x+im*y)

@test_approx_eq u(0.1,0.2) (-1.1657816742288978-0.21306668168680534im)



## 2 intervals
Γ=Interval(-1.,-0.5) ∪ Interval(-0.3,1.)
z=Fun(Γ)

S=PiecewiseSpace(map(d->JacobiWeight(0.5,0.5,Ultraspherical(1,d)),Γ))


k=114;
    α=exp(k/50*im)

Ai=ApproxFun.interlace([ones(Γ[1])+zeros(Γ[2]) zeros(Γ[1])+ones(Γ[2]) Hilbert(S)])
@time testbandedblockoperator(Ai)

@time a,b,ui=[ones(Γ[1])+zeros(Γ[2]) zeros(Γ[1])+ones(Γ[2]) Hilbert(S)]\imag(α*z)




u=(x,y)->α*(x+im*y)+2cauchy(ui,x+im*y)
