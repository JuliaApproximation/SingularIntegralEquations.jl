using ApproxFun, SingularIntegralEquations, Base.Test


k=50
Γ=Interval(0.,1+0.5im)
z=Fun(Γ)
α=exp(-π*k/50im)

@test ApproxFun.choosedomainspace(Hilbert(),z) == JacobiWeight(-0.5,-0.5,ChebyshevDirichlet{1,1}(Γ))

Ai=ApproxFun.interlace([1 Hilbert()])
@which ApproxFun.choosedomainspace(Ai,space(z))



ApproxFun.choosedomainspace(Ai.ops[1],z)

c,ui=[1 Hilbert()]\imag(α*z)
plot(Γ)
contour!(x,y,imag(u(xx,yy)))


ApproxFun.choosedomainspace(Hilbert(),z)
