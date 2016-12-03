using Plots,ApproxFun,SingularIntegralEquations;  pyplot()

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

<<<<<<< HEAD
=======

d=Circle()
z=Fun(d)
f=exp(real(z))
f(exp(im*0.1))

>>>>>>> 2297bc251a8a2f9f01355295f61b9fcb7ba10547
k=50
    Γ=Segment(0.,1+0.5im)
    z=Fun(Γ)
    α=exp(-π*k/50im)
<<<<<<< HEAD
    S=JacobiWeight(0.5,0.5,Γ)
    c,ui=[1 Hilbert(S)]\imag(α*z)

u =(x,y)->α*(x+im*y)+2cauchy(ui,x+im*y)
plot(Γ)
    contour!(x,y,imag(u(xx,yy))';nlevels=50)
=======
    S=JacobiWeight(0.5,0.5,Ultraspherical(1,Γ))
    c,ui=[1 Hilbert(S)]\imag(α*z)
plot(Γ)
    contour!(x,y,imag(u(xx,yy))';nlevels=100,legend=false)
>>>>>>> 2297bc251a8a2f9f01355295f61b9fcb7ba10547


##
# On an arc, the Hilbert transform no longer gives the imaginary part
#  However, pseudohilbert gives the imaginary part of pseudocauchy
#  So if we want to find ui defined on Γ so that pseudohilbert(ui) = imag(c*z)
#  then c*z + 2pseudocauchy(u,z) vanishes on Γ
##


<<<<<<< HEAD
=======
u=(x,y)->α*(x+im*y)+2pseudocauchy(ui,x+im*y)

>>>>>>> 2297bc251a8a2f9f01355295f61b9fcb7ba10547
m=80;x = linspace(-2.,2.,m);y = linspace(-2.,2.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'

k=227;
    Γ=0.5+exp(im*Segment(0.1,-42))
    z=Fun(Γ)
    α=exp(-k/50im)
<<<<<<< HEAD
    S=JacobiWeight(0.5,0.5,Γ)
    c,ui=[1 PseudoHilbert(S)]\imag(α*z)


u=(x,y)->α*(x+im*y)+2pseudocauchy(ui,x+im*y)
plot(Γ)
    contour!(x,y,imag(u(xx,yy))';nlevels=50)



=======
    S=JacobiWeight(0.5,0.5,Ultraspherical(1,Γ))
    c,ui=[1 PseudoHilbert(S)]\imag(α*z)
    plot(Γ)
    contour!(x,y,imag(u(xx,yy))';nlevels=100,legend=false)
>>>>>>> 2297bc251a8a2f9f01355295f61b9fcb7ba10547

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



25

DefiniteLineIntegral(space(z))[1]
typeof(space(z))==Laurent{typeof(Γ)}
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
    c,ui=[1 real(Hilbert(S))]\imag(α*z)

u=(x,y)->α*(x+im*y)+2cauchy(ui,x+im*y)
plot(Γ)
    contour!(x,y,imag(u(xx,yy))';nlevels=100)


##
#  Two intervals requires explicitely stating the space (for now)
##

Γ=Segment(-1.,-0.5)∪Segment(-0.3,1.)
z=Fun(Γ)

S=PiecewiseSpace(map(d->JacobiWeight(0.5,0.5,Ultraspherical(1,d)),Γ))


<<<<<<< HEAD
=======
u= (x,y) -> α*(x+im*y)+2cauchy(ui,x+im*y)

>>>>>>> 2297bc251a8a2f9f01355295f61b9fcb7ba10547
m=80;x = linspace(-2.,2.,m);y = linspace(-1.,1.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'

k=114;
    α=exp(k/50*im)
<<<<<<< HEAD
    a,b,ui=[ones(Γ[1])+zeros(Γ[2]) zeros(Γ[1])+ones(Γ[2]) Hilbert(S)]\imag(α*z)
=======
    a,b,ui=[ones(Γ[1])+zeros(Γ[2]) zeros(Γ[1])+ones(Γ[2]) Hilbert(ds)]\imag(α*z)
    plot(Γ)
    contour!(x,y,imag(u(xx,yy)))


import ApproxFun:colstop,interlace,bandwidth

Ai=interlace([ones(Γ[1])+zeros(Γ[2]) zeros(Γ[1])+ones(Γ[2]) Hilbert(ds)])
Ai[1:100,1:100]


@which colstop(Ai,1)

bandwidth(Ai,1)
Ai.bandinds

colstop(Ai.ops[3],1)

ApproxFun.isbanded
ApproxFun.isbanded(ApproxFun.interlace([ones(Γ[1])+zeros(Γ[2]) zeros(Γ[1])+ones(Γ[2]) Hilbert(ds)]))

rangespace(Hilbert(ds))

>>>>>>> 2297bc251a8a2f9f01355295f61b9fcb7ba10547

u=(x,y)->α*(x+im*y)+2cauchy(ui,x+im*y)
plot(Γ)
    contour!(x,y,imag(u(xx,yy))';nlevels=100)



Γ=Segment(-1.,0.)∪Segment(0.5im,1.)
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


Γ=Segment(-im,1.0-im)∪Curve(Fun(x->exp(0.8im)*(x+x^2-1+im*(x-4x^3+x^4)/6)))
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

Γ=Segment(-im,1.0-im)∪Curve(Fun(x->exp(0.8im)*(x+x^2-1+im*(x-4x^3+x^4)/6)))∪Circle(2.0,0.2)
    z=Fun(Γ)

S=PiecewiseSpace(map(d->isa(d,Circle)?Fourier(d):JacobiWeight(0.5,0.5,Ultraspherical(1,d)),Γ))



# This is a temporary work around as DefiniteLineIntegral is not implemented for curves
B=ApproxFun.SpaceOperator(ApproxFun.BasisFunctional(3),S,ApproxFun.ConstantSpace())
k=114;
    α=exp(k/50*im)
    a,b,c,ui=[0                 0                 0                 B;
              Fun(ones(Γ[1]),Γ) Fun(ones(Γ[2]),Γ) Fun(ones(Γ[3]),Γ) real(Hilbert(S))]\Any[0.;imag(α*z)]


m=160;x = linspace(-4.,4.,m);y = linspace(-2.,2.,m+1)
  xx,yy = x.+0.*y',0.*x.+y'

u =(x,y)->α*(x+im*y)+2cauchy(ui,x+im*y)

plot(Γ)
    contour!(x,y,imag(u(xx,yy))';nlevels=100)
