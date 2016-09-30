using ApproxFun, SingularIntegralEquations, Base.Test
    import ApproxFun: choosedomainspace, promotedomainspace, ConstantSpace, interlace

using Plots

k=50
Γ=Interval(0.,1+0.5im)
z=Fun(Γ)
α=exp(-π*k/50im)

@test ApproxFun.choosedomainspace(Hilbert(),z) == JacobiWeight(-0.5,-0.5,ChebyshevDirichlet{1,1}(Γ))

Ai=ApproxFun.interlace([1 Hilbert()])


@test isa(ApproxFun.choosedomainspace(Ai.ops[1],space(z)),ApproxFun.ConstantSpace)
@test isa(ApproxFun.choosedomainspace(Ai,space(z)),ApproxFun.TupleSpace{Tuple{ApproxFun.ConstantSpace{ApproxFun.AnyDomain},
            ApproxFun.JacobiWeight{ApproxFun.ChebyshevDirichlet{1,1,ApproxFun.Interval{Complex{Float64}}},
                                                                    ApproxFun.Interval{Complex{Float64}}}}})


S=choosedomainspace(Ai,space(z))
AiS=promotedomainspace(Ai,S)
@test domainspace(AiS) == S


S=JacobiWeight(0.5,0.5,Ultraspherical(1,Γ))
c,ui=[1 Hilbert(S)]\imag(α*z)

imag(α*z).coefficients

u =(x,y)->α*(x+im*y)+2cauchy(ui,x+im*y)

m=200;x = linspace(-2.,2.,m);y = linspace(-1.,1.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'


u(0.5,0.25)

(Hilbert(S)*ui)(0.25+0.125im)

Number(c)+hilbert(ui,0.25+0.125im)

(Hilbert(S)*ui)|>domain

ui.coefficients

c

c
u(-1.+0.001,-0.5)
u(-1.-0.001,-0.5)
u(-0.001,0.)
imag(u(xx,yy))


domain(ui)
(Hilbert(S)*ui)(0.5+0.25im)

Γ
plotly()
plot(Γ)
    contour!(x,y,imag(u(xx,yy));nlevels=100)
    gui()


imag(u(xx,yy))



plot(1:10)
gr()



ApproxFun.choosedomainspace(Hilbert(),z)
