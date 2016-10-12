using Base.Test, ApproxFun, SingularIntegralEquations, Base.Test
    import ApproxFun: choosedomainspace, promotedomainspace, ConstantSpace, interlace

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
c,ui=[1 Hilbert(S)]\imag(α*z)

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
c,ui=[1 PseudoHilbert(S)]\imag(α*z)


u=(x,y)->α*(x+im*y)+2pseudocauchy(ui,x+im*y)

@test_approx_eq u(0.1,0.2) 0.6063720775017964 - 0.6382733554119975im # empirical



Γ=Circle()
z=Fun(Fourier(Γ))

Ai=ApproxFun.interlace([0 DefiniteLineIntegral();
      1 real(Hilbert())])


S=ApproxFun.choosedomainspace(Ai,space([Fun(0.);z]))

@test isa(S[1],ApproxFun.ConstantSpace)
@test S[2] == Laurent(Γ)

@test domainspace(ApproxFun.promotedomainspace(Ai,S))==S

promotedomainspace(Ai.ops[2],S[1])
A=ApproxFun.promotedomainspace(Ai,S)

@test isa(A[2,2],Complex128)


A[1:10,1:10]

k=239;
α=exp(-k/45im)

c,ui=[0 DefiniteLineIntegral(Fourier(Γ));
      1 real(Hilbert(Fourier(Γ)))]\[Fun(0.);imag(α*z)]



m=80;x = linspace(-2.,2.,m);y = linspace(-2.,2.,m+1)
  xx,yy = x.+0.*y',0.*x.+y'


u =(x,y)->α*(x+im*y)+2cauchy(ui,x+im*y)

@show u(2.,1.1)
@test_approx_eq u(2.,1.1)  2.426592437403252-0.8340542386599383im
plot(Γ)
  contour!(x,y,imag(u(xx,yy))';nlevels=100)

plot(Γ)

u =(x,y)->α*(x+im*y)+2cauchy(ui,x+im*y)
