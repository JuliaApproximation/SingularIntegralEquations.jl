using Plots,ApproxFun,SingularIntegralEquations;  gr()

##
#  Ideal fluid flow consists of level sets of the imagainary part of a function
# that is asymptotic to c*z and whose imaginary part vanishes on Γ
#
#
# On the unit interval, -2*hilbert(ui) gives the imaginary part of cauchy(ui,z)
#  So if we want to find ui defined on Γ so that hilbert(ui) = imag(c*z)
#  then c*z + 2cauchy(u,z) vanishes on Γ
##





m=80;x = linspace(-2.,2.,m);y = linspace(-1.,1.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'

k=50
    Γ=Segment(0.,1+0.5im)
    z=Fun(Γ)
    α=exp(-π*k/50im)
    S=JacobiWeight(0.5,0.5,Γ)
    c,ui=[1 Hilbert(S)] \ [imag(α*z)]


1 + 3


D = Derivative() : Chebyshev() → Ultraspherical(1)


D^2 + Fun(cos)



Derivative([1,0]) : Chebyshev()^2




xx = yy = linspace(-10,10,100)



using Plots


zz = xx' .+ im .* yy
contour(xx, yy, imag(im      .* zz.^(1/2) ); nlevels=100)

plot([0,-10], [0,10]; color=:black)
    plot!([0,-10], [0,-10]; color=:black)
    contour!(xx, yy, imag(im * zz.^(4/6) ); nlevels=100)
plot([0,-10], [0,10]; color=:black)
    plot!([0,-10], [0,-10]; color=:black)
    contour!(xx, yy, imag( zz.^(4/3) ); nlevels=100)

plot()



randn(10,10)

Hilbert() : JacobiWeight(0.5,0.5, Ultraspherical(1))



u =(x,y)->α*(x+im*y)+2cauchy(ui,x+im*y)
plot(Γ)
    contour!(x,y,imag(u(xx,yy))';nlevels=100)

##
# On an arc, the Hilbert transform no longer gives the imaginary part
#  However, pseudohilbert gives the imaginary part of pseudocauchy
#  So if we want to find ui defined on Γ so that pseudohilbert(ui) = imag(c*z)
#  then c*z + 2pseudocauchy(u,z) vanishes on Γ
##

m=80;x = linspace(-2.,2.,m);y = linspace(-2.,2.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'

k=227;
    Γ=0.5+exp(im*Segment(0.1,-42))
    z=Fun(Γ)
    α=exp(-k/50im)
    S=JacobiWeight(0.5,0.5,Γ)
    c,ui=[1 PseudoHilbert(S)]\imag(α*z)


u=(x,y)->α*(x+im*y)+2pseudocauchy(ui,x+im*y)
plot(Γ)
    contour!(x,y,imag(u(xx,yy))';nlevels=50)


##
#  Circle
##

Γ=Circle()
z=Fun(Laurent(Γ))


u=(x,y)->α*(x+im*y)+2cauchy(ui,x+im*y)

m=80;x = linspace(-2.,2.,m);y = linspace(-2.,2.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'

k=239;
    α=exp(-k/45im)
    c,ui=[0 DefiniteLineIntegral();
          1 real(Hilbert())]\[Fun(0.);imag(α*z)]

plot(Γ)
    contour!(x,y,imag(u(xx,yy))';nlevels=50)


##
# On a curve, the Hilbert transform may be complex, so we
# take the real part
##

m=80;x = linspace(-2.,2.,m);y = linspace(-2.,2.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'


Γ=Curve(Fun(x->exp(0.8im)*(x+x^2-1+im*(x-4x^3+x^4)/6)))
    z=Fun(Γ)
    α=im
    S=JacobiWeight(0.5,0.5,Γ)
    c,ui=[1 real(Hilbert(S))]\ [imag(α*z)]

u=(x,y)->α*(x+im*y)+2cauchy(ui,x+im*y)
plot(Γ)
    contour!(x,y,imag(u.(x',y));nlevels=100)


##
#  Two intervals requires explicitely stating the space (for now)
##

Γ=Segment(-1.,-0.5) ∪ Segment(-0.3,1.)
z=Fun(Γ)

S=PiecewiseSpace(map(d->JacobiWeight(0.5,0.5,Ultraspherical(1,d)),Γ))


m=80;x = linspace(-2.,2.,m);y = linspace(-1.,1.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'

k=114;
    α=exp(k/50*im)
    a,b,ui=[ones(Γ[1])+zeros(Γ[2]) zeros(Γ[1])+ones(Γ[2]) Hilbert(S)]\imag(α*z)


u= (x,y) -> α*(x+im*y)+2cauchy(ui,x+im*y)
    plot(Γ)
    contour!(x,y,imag(u(xx,yy))';nlevels=100)



Γ=Segment(-1.,0.) ∪ Segment(0.5im,1.)
z=Fun(Γ)
S=PiecewiseSpace(map(d->JacobiWeight(0.5,0.5,Ultraspherical(1,d)),Γ))

k=114;
    α=exp(k/50*im)
    a,b,ui=[ones(Γ[1])+zeros(Γ[2]) zeros(Γ[1])+ones(Γ[2])  real(Hilbert(S))]\imag(α*z)

u=(x,y)->α*(x+im*y)+2cauchy(ui,x+im*y)

m=80;x = linspace(-2.,2.,m);y = linspace(-1.,1.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'
plot(Γ)
    contour!(x,y,imag(u(xx,yy))';nlevels=100)


# Segment and Curve


Γ=Segment(-im,1.0-im) ∪ Curve(Fun(x->exp(0.8im)*(x+x^2-1+im*(x-4x^3+x^4)/6)))
z=Fun(Γ)

S=PiecewiseSpace(map(d->JacobiWeight(0.5,0.5,Ultraspherical(1,d)),Γ))

m=80;x = linspace(-2.,2.,m);y = linspace(-3.,2.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'


k=114;
    α=exp(k/50*im)
    a,b,ui=[ones(Γ[1])+zeros(Γ[2]) zeros(Γ[1])+ones(Γ[2]) real(Hilbert(S))]\imag(α*z)


plot(Γ)
    contour!(x,y,imag(u(xx,yy))';nlevels=100)


## Segment, Curve and Circle

Γ=Segment(-im,1.0-im) ∪ Curve(Fun(x->exp(0.8im)*(x+x^2-1+im*(x-4x^3+x^4)/6))) ∪ Circle(2.0,0.2)
    z=Fun(Γ)

S=PiecewiseSpace(map(d->isa(d,Circle) ? Fourier(d) : JacobiWeight(0.5,0.5,Ultraspherical(1,d)),Γ))



# This is a temporary work around as DefiniteLineIntegral is not implemented for curves
B=ApproxFun.SpaceOperator(ApproxFun.BasisFunctional(3),S,ApproxFun.ConstantSpace(Float64))



k=114;
    α=exp(k/50*im)
    a,b,c,ui=[0                 0                 0                 DefiniteLineIntegral(S);
              Fun(ones(Γ[1]),Γ) Fun(ones(Γ[2]),Γ) Fun(ones(Γ[3]),Γ) real(Hilbert(S))]\Any[0.;imag(α*z)]


m=160;x = linspace(-4.,4.,m);y = linspace(-2.,2.,m+1)
  xx,yy = x.+0.*y',0.*x.+y'

u =(x,y)->α*(x+im*y)+2cauchy(ui,x+im*y)

plot(Γ)
    contour!(x,y,imag(u(xx,yy))';nlevels=100)
