using ApproxFun, SIE, Base.Test

println("Chebyshev addition test")

d = Interval([-1.,1.])
f(x,y) = exp(y-x)
FLR = LowRankFun((x,y)->f(x,y)/sqrt(1-y^2),Chebyshev(d),JacobiWeight(-.5,-.5,Chebyshev(d)))
FPF = ProductFun((x,y)->f(x,y)/sqrt(1-y^2),Chebyshev(d),JacobiWeight(-.5,-.5,Chebyshev(d)))
G = SymmetricProductFun(f,Chebyshev(d),JacobiWeight(-.5,-.5,Chebyshev(d)))
G1 = SymmetricProductFun(f,Ultraspherical{1}(d),JacobiWeight(-.5,-.5,Chebyshev(d)))
G2 = SymmetricProductFun(f,Ultraspherical{1}(d),JacobiWeight(-.5,-.5,ApproxFun.ChebyshevDirichlet{1,1}(d)))


@test norm(pad(coefficients(FLR),size(coefficients(G),1),size(coefficients(G),2))-coefficients(G))<100eps()
@test norm(pad(coefficients(FPF),size(coefficients(G),1),size(coefficients(G),2))-coefficients(G))<100eps()
@test norm(f(.123,.456)/sqrt(1-.456^2)-G[.123,.456])≤10eps()
@test norm(G[.123,.456]-G1[.123,.456])≤2eps()
@test norm(G[.123,.456]-G2[.123,.456])≤2eps()

println("Fourier on PeriodicInterval tests")

f1 = Fun(θ->exp(sin(θ))+sin(cos(θ)),Fourier())
FLR = LowRankFun((θ,ϕ)->f1[ϕ-θ],Fourier(),Fourier())
FPF = ProductFun((θ,ϕ)->f1[ϕ-θ],Fourier(),Fourier())
G = ProductFun(f1,Fourier(),Fourier())

@test norm(pad(coefficients(FLR),size(coefficients(G),1),size(coefficients(G),2))-coefficients(G))<400eps()
@test norm(pad(coefficients(FPF),size(coefficients(G),1),size(coefficients(G),2))-coefficients(G))<400eps()
@test norm(f1[.456-.123]-G[.123,.456])≤2eps()

f1 = Fun(θ->1+sin(cos(θ)),CosSpace())
G = ProductFun(f1,Fourier(),Fourier())
@test norm(f1[.456-.123]-G[.123,.456])≤2eps()

f1 = Fun(θ->sin(sin(θ)),SinSpace())
G = ProductFun(f1,Fourier(),Fourier())
@test norm(f1[.456-.123]-G[.123,.456])≤2eps()

println("Laurent on PeriodicInterval tests")

f1 = Fun(θ->exp(sin(θ))+sin(cos(θ)),Laurent())
G = ProductFun(f1,Laurent(),Laurent())
@test norm(f1[.456-.123]-G[.123,.456])≤2eps()

f1 = Fun(θ->π+e*exp(im*θ)+sqrt(2)*exp(im*2θ)+catalan*exp(im*3θ)+γ*exp(im*4θ),Taylor(PeriodicInterval()))
G = ProductFun(f1,Laurent(),Laurent())
@test norm(f1[.456-.123]-G[.123,.456])≤2eps()

f1 = Fun(θ->e*exp(-im*θ)+sqrt(2)*exp(-im*2θ)+catalan*exp(-im*3θ)+γ*exp(-im*4θ),Hardy{false}(PeriodicInterval()))
G = ProductFun(f1,Laurent(),Laurent())
@test norm(f1[.456-.123]-G[.123,.456])≤10eps()

println("Timing tests: ")

gc_disable()

d = Interval([-2.5,-.5])
d2 = Interval([-length(d),length(d)])
f(x,y) = besselj0(100(y-x))
SymmetricProductFun(f,Chebyshev(d),Chebyshev(d))
@time G = SymmetricProductFun(f,Chebyshev(d),Chebyshev(d))
SymmetricProductFun(f,Ultraspherical{1}(d),ApproxFun.ChebyshevDirichlet{1,1}(d))
@time G = SymmetricProductFun(f,Ultraspherical{1}(d),ApproxFun.ChebyshevDirichlet{1,1}(d))
println("Chebyshev addition: Time should be ~0.01 seconds.")

f1 = Fun(θ->besselj0(500*abs(2sin(θ/2))),CosSpace())
ProductFun(f1,Fourier(),Fourier())
@time G = ProductFun(f1,Fourier(),Fourier())
println("CosSpace addition: Time should be ~0.05 seconds.")

f1 = Fun(θ->besselj0(500*abs(2sin(θ/2))),Laurent())
ProductFun(f1,Laurent(),Laurent())
@time G = ProductFun(f1,Laurent(),Laurent())
println("Laurent addition: Time should be ~0.14 seconds.")

gc_enable()
