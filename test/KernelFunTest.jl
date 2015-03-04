using ApproxFun, SIE, Base.Test

println("Chebyshev addition test")

d = Interval([-1.,1.])
fK(x,y) = exp(y-x)
FLR = LowRankFun((x,y)->fK(x,y)/sqrt(1-y^2),Chebyshev(d),JacobiWeight(-.5,-.5,Chebyshev(d)))
FPF = ProductFun((x,y)->fK(x,y)/sqrt(1-y^2),Chebyshev(d),JacobiWeight(-.5,-.5,Chebyshev(d)))
G = ProductFun(fK,Chebyshev(d),JacobiWeight(-.5,-.5,Chebyshev(d));method=:convolution)
G1 = ProductFun(fK,Ultraspherical{1}(d),JacobiWeight(-.5,-.5,Chebyshev(d));method=:convolution)
G2 = ProductFun(fK,Ultraspherical{1}(d),JacobiWeight(-.5,-.5,ApproxFun.ChebyshevDirichlet{1,1}(d));method=:convolution)


@test norm(pad(coefficients(FLR),size(coefficients(G),1),size(coefficients(G),2))-coefficients(G))<100eps()
@test norm(pad(coefficients(FPF),size(coefficients(G),1),size(coefficients(G),2))-coefficients(G))<100eps()
@test norm(fK(.123,.456)/sqrt(1-.456^2)-G[.123,.456])≤10eps()
@test norm(G[.123,.456]-G1[.123,.456])≤2eps()
@test norm(G[.123,.456]-G2[.123,.456])≤2eps()

println("Fourier on PeriodicInterval tests")

f2 = Fun(θ->exp(sin(θ))+sin(cos(θ)),Fourier())
FLR = LowRankFun((θ,ϕ)->f2[ϕ-θ],Fourier(),Fourier())
FPF = ProductFun((θ,ϕ)->f2[ϕ-θ],Fourier(),Fourier())
G = ProductFun(f2,Fourier(),Fourier();method=:convolution)

@test norm(pad(coefficients(FLR),size(coefficients(G),1),size(coefficients(G),2))-coefficients(G))<400eps()
@test norm(pad(coefficients(FPF),size(coefficients(G),1),size(coefficients(G),2))-coefficients(G))<400eps()
@test norm(f2[.456-.123]-G[.123,.456])≤2eps()

f2 = Fun(θ->1+sin(cos(θ)),CosSpace())
G = ProductFun(f2,Fourier(),Fourier();method=:convolution)
@test norm(f2[.456-.123]-G[.123,.456])≤2eps()

f2 = Fun(θ->sin(sin(θ)),SinSpace())
G = ProductFun(f2,Fourier(),Fourier();method=:convolution)
@test norm(f2[.456-.123]-G[.123,.456])≤2eps()

println("Laurent on PeriodicInterval tests")

f2 = Fun(θ->exp(sin(θ))+sin(cos(θ)),Laurent())
G = ProductFun(f2,Laurent(),Laurent();method=:convolution)
@test norm(f2[.456-.123]-G[.123,.456])≤2eps()

f2 = Fun(θ->π+e*exp(im*θ)+sqrt(2)*exp(im*2θ)+catalan*exp(im*3θ)+γ*exp(im*4θ),Taylor(PeriodicInterval()))
G = ProductFun(f2,Laurent(),Laurent();method=:convolution)
@test norm(f2[.456-.123]-G[.123,.456])≤2eps()

f2 = Fun(θ->e*exp(-im*θ)+sqrt(2)*exp(-im*2θ)+catalan*exp(-im*3θ)+γ*exp(-im*4θ),Hardy{false}(PeriodicInterval()))
G = ProductFun(f2,Laurent(),Laurent();method=:convolution)
@test norm(f2[.456-.123]-G[.123,.456])≤10eps()

println("Timing tests: ")

gc_disable()

d = Interval([-2.5,-.5])
fK(x,y) = besselj0(100(y-x))
ProductFun(fK,Chebyshev(d),Chebyshev(d);method=:convolution)
@time G = ProductFun(fK,Chebyshev(d),Chebyshev(d);method=:convolution)
ProductFun(fK,Ultraspherical{1}(d),ApproxFun.ChebyshevDirichlet{1,1}(d);method=:convolution)
@time G = ProductFun(fK,Ultraspherical{1}(d),ApproxFun.ChebyshevDirichlet{1,1}(d);method=:convolution)
println("Chebyshev addition: Time should be ~0.01 seconds.")

f2 = Fun(θ->besselj0(500*abs(2sin(θ/2))),CosSpace())
ProductFun(f2,Fourier(),Fourier();method=:convolution)
@time G = ProductFun(f2,Fourier(),Fourier();method=:convolution)
println("CosSpace addition: Time should be ~0.05 seconds.")

f2 = Fun(θ->besselj0(500*abs(2sin(θ/2))),Laurent())
ProductFun(f2,Laurent(),Laurent();method=:convolution)
@time G = ProductFun(f2,Laurent(),Laurent();method=:convolution)
println("Laurent addition: Time should be ~0.14 seconds.")

gc_enable()
