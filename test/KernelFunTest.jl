using ApproxFun, SIE, Base.Test

println("Chebyshev addition test")

d = Interval([-1.,1.])
fK(x,y) = exp(abs2(y-x))
FLR = LowRankFun(fK,Chebyshev(d),Chebyshev(d))
FPF = ProductFun(fK,Chebyshev(d),Chebyshev(d))
G = convolutionProductFun(fK,Chebyshev(d),Chebyshev(d))
G1 = convolutionProductFun(fK,Ultraspherical{1}(d),Chebyshev(d))
G2 = convolutionProductFun(fK,Ultraspherical{1}(d),ChebyshevDirichlet{1,1}(d))


@test norm(pad(coefficients(FLR),size(coefficients(G),1),size(coefficients(G),2))-coefficients(G))<10000eps()
@test norm(pad(coefficients(FPF),size(coefficients(G),1),size(coefficients(G),2))-coefficients(G))<10000eps()
@test norm(fK(.123,.456)-G[.123,.456])≤100eps()
@test norm(G[.123,.456]-G1[.123,.456])≤100eps()
@test norm(G[.123,.456]-G2[.123,.456])≤100eps()

println("Fourier on PeriodicInterval tests")

f2 = Fun(θ->exp(sin(θ))+sin(cos(θ)),Fourier())
FLR = LowRankFun((θ,ϕ)->f2[ϕ-θ],Fourier(),Fourier())
FPF = ProductFun((θ,ϕ)->f2[ϕ-θ],Fourier(),Fourier())
G = convolutionProductFun(f2,Fourier(),Fourier())

@test norm(pad(coefficients(FLR),size(coefficients(G),1),size(coefficients(G),2))-coefficients(G))<400eps()
@test norm(pad(coefficients(FPF),size(coefficients(G),1),size(coefficients(G),2))-coefficients(G))<400eps()
@test norm(f2[.456-.123]-G[.123,.456])≤400eps()

f2 = Fun(θ->1+sin(cos(θ)),CosSpace())
G = convolutionProductFun(f2,Fourier(),Fourier())
@test norm(f2[.456-.123]-G[.123,.456])≤400eps()

f2 = Fun(θ->sin(sin(θ)),SinSpace())
G = convolutionProductFun(f2,Fourier(),Fourier())
@test norm(f2[.456-.123]-G[.123,.456])≤400eps()

println("Laurent on PeriodicInterval tests")

f2 = Fun(θ->exp(sin(θ))+sin(cos(θ)),Laurent())
G = convolutionProductFun(f2,Laurent(),Laurent())
@test norm(f2[.456-.123]-G[.123,.456])≤400eps()

f2 = Fun(θ->π+e*exp(im*θ)+sqrt(2)*exp(im*2θ)+catalan*exp(im*3θ)+γ*exp(im*4θ),Taylor(PeriodicInterval()))
G = convolutionProductFun(f2,Laurent(),Laurent())
@test norm(f2[.456-.123]-G[.123,.456])≤2eps()

f2 = Fun(θ->e*exp(-im*θ)+sqrt(2)*exp(-im*2θ)+catalan*exp(-im*3θ)+γ*exp(-im*4θ),Hardy{false}(PeriodicInterval()))
G = convolutionProductFun(f2,Laurent(),Laurent())
@test norm(f2[.456-.123]-G[.123,.456])≤10eps()

println("Timing tests: ")

gc_disable()

d = Interval([-2.5,-.5])
fK(x,y) = besselj0(100(y-x))
convolutionProductFun(fK,Chebyshev(d),Chebyshev(d))
@time G = convolutionProductFun(fK,Chebyshev(d),Chebyshev(d))
convolutionProductFun(fK,Ultraspherical{1}(d),ChebyshevDirichlet{1,1}(d))
@time G = convolutionProductFun(fK,Ultraspherical{1}(d),ChebyshevDirichlet{1,1}(d))
println("Chebyshev addition: Time should be ~0.01 seconds.")

f2 = Fun(θ->besselj0(500*abs(2sin(θ/2))),CosSpace())
convolutionProductFun(f2,Fourier(),Fourier())
@time G = convolutionProductFun(f2,Fourier(),Fourier())
println("CosSpace addition: Time should be ~0.05 seconds.")

f2 = Fun(θ->besselj0(500*abs(2sin(θ/2))),Laurent())
convolutionProductFun(f2,Laurent(),Laurent())
@time G = convolutionProductFun(f2,Laurent(),Laurent())
println("Laurent addition: Time should be ~0.14 seconds.")

gc_enable()
