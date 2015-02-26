using ApproxFun, SIE, Base.Test

println("Chebyshev addition test")

d = Interval([-1.,1.])
d2 = Interval([-length(d),length(d)])
f = Fun(exp,d2)
FLR = LowRankFun((x,y)->f[y-x]/sqrt(1-y^2),Chebyshev(d),JacobiWeight(-.5,-.5,Chebyshev(d)))
FPF = ProductFun((x,y)->f[y-x]/sqrt(1-y^2),Chebyshev(d),JacobiWeight(-.5,-.5,Chebyshev(d)))
G = ProductFun(f,Chebyshev(d),JacobiWeight(-.5,-.5,Chebyshev(d)))
G1 = ProductFun(f,Ultraspherical{1}(d),JacobiWeight(-.5,-.5,Chebyshev(d)))
G2 = ProductFun(f,Ultraspherical{1}(d),JacobiWeight(-.5,-.5,ApproxFun.ChebyshevDirichlet{1,1}(d)))


@test norm(pad(coefficients(FLR),size(coefficients(G),1),size(coefficients(G),2))-coefficients(G))<100eps()
@test norm(pad(coefficients(FPF),size(coefficients(G),1),size(coefficients(G),2))-coefficients(G))<100eps()
@test norm(f[.456-.123]/sqrt(1-.456^2)-G[.123,.456])≤2eps()
@test norm(G[.123,.456]-G1[.123,.456])≤2eps()
@test norm(G[.123,.456]-G2[.123,.456])≤2eps()

println("Fourier on PeriodicInterval tests")

f = Fun(θ->exp(sin(θ))+sin(cos(θ)),Fourier())
FLR = LowRankFun((θ,ϕ)->f[ϕ-θ],Fourier(),Fourier())
FPF = ProductFun((θ,ϕ)->f[ϕ-θ],Fourier(),Fourier())
G = ProductFun(f,Fourier(),Fourier())

@test norm(pad(coefficients(FLR),size(coefficients(G),1),size(coefficients(G),2))-coefficients(G))<400eps()
@test norm(pad(coefficients(FPF),size(coefficients(G),1),size(coefficients(G),2))-coefficients(G))<400eps()
@test norm(f[.456-.123]-G[.123,.456])≤2eps()

f = Fun(θ->1+sin(cos(θ)),CosSpace())
G = ProductFun(f,Fourier(),Fourier())
@test norm(f[.456-.123]-G[.123,.456])≤2eps()

f = Fun(θ->sin(sin(θ)),SinSpace())
G = ProductFun(f,Fourier(),Fourier())
@test norm(f[.456-.123]-G[.123,.456])≤2eps()

println("Laurent on PeriodicInterval tests")

f = Fun(θ->exp(sin(θ))+sin(cos(θ)),Laurent())
G = ProductFun(f,Laurent(),Laurent())
@test norm(f[.456-.123]-G[.123,.456])≤2eps()

f = Fun(θ->π+e*exp(im*θ)+sqrt(2)*exp(im*2θ)+catalan*exp(im*3θ)+γ*exp(im*4θ),Taylor(PeriodicInterval()))
G = ProductFun(f,Laurent(),Laurent())
@test norm(f[.456-.123]-G[.123,.456])≤2eps()

f = Fun(θ->e*exp(-im*θ)+sqrt(2)*exp(-im*2θ)+catalan*exp(-im*3θ)+γ*exp(-im*4θ),Hardy{false}(PeriodicInterval()))
G = ProductFun(f,Laurent(),Laurent())
@test norm(f[.456-.123]-G[.123,.456])≤10eps()

println("Timing tests: ")

gc_disable()

d = Interval([-2.5,-.5])
d2 = Interval([-length(d),length(d)])
f = Fun(x->besselj0(100x),d2)
ProductFun(f,Chebyshev(d),JacobiWeight(-.5,-.5,Chebyshev(d)))
@time G = ProductFun(f,Chebyshev(d),JacobiWeight(-.5,-.5,Chebyshev(d)))
ProductFun(f,Ultraspherical{1}(d),JacobiWeight(-.5,-.5,ApproxFun.ChebyshevDirichlet{1,1}(d)))
@time G = ProductFun(f,Ultraspherical{1}(d),JacobiWeight(-.5,-.5,ApproxFun.ChebyshevDirichlet{1,1}(d)))
println("Chebyshev addition: Time should be ~0.009 seconds.")

f = Fun(θ->besselj0(500*abs(2sin(θ/2))),CosSpace())
ProductFun(f,Fourier(),Fourier())
@time G = ProductFun(f,Fourier(),Fourier())
println("CosSpace addition: Time should be ~0.05 seconds.")

f = Fun(θ->besselj0(500*abs(2sin(θ/2))),Laurent())
ProductFun(f,Laurent(),Laurent())
@time G = ProductFun(f,Laurent(),Laurent())
println("Laurent addition: Time should be ~0.14 seconds.")

gc_enable()
