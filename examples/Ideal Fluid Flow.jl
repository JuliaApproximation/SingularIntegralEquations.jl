using ApproxFun,SingularIntegralEquations
using Gadfly


##
#  Ideal fluid flow consists of level sets of the imagainary part of a function
# that is asymptotic to c*z and whose imaginary part vanishes on Γ
#
#
# On the unit interval, -2*hilbert(ui) gives the imaginary part of cauchy(ui,z)
#  So if we want to find ui defined on Γ so that hilbert(ui) = imag(c*z)
#  then c*z + 2cauchy(u,z) vanishes on Γ
##

Γ=Interval()
z=Fun(Γ)


u(x,y)=c*(x+im*y)+2cauchy(ui,x+im*y)

m=80;x = linspace(-2.,2.,m);y = linspace(-1.,1.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'

k=107;
    c=exp(-k/50im)
    ui=[BasisFunctional(1);Hilbert()]\[0.;imag(c*z)]
    Gadfly.plot(ApproxFun.layer(Γ),
                layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-2.5:.05:2.5])))



##
# On an arc, the Hilbert transform no longer gives the imaginary part
#  However, pseudohilbert gives the imaginary part of pseudocauchy
#  So if we want to find ui defined on Γ so that pseudohilbert(ui) = imag(c*z)
#  then c*z + 2pseudocauchy(u,z) vanishes on Γ
##


u(x,y)=c*(x+im*y)+2pseudocauchy(ui,x+im*y)

m=80;x = linspace(-2.,2.,m);y = linspace(-2.,2.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'

k=264;
    Γ=exp(im*Interval(0.1,7))
    z=Fun(Γ)
    c=exp(-k/50im)
    ui=[BasisFunctional(1);PseudoHilbert()]\[0.;imag(c*z)]
    Gadfly.plot(ApproxFun.layer(Γ),
                layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-3:.05:3])))


##
# On a curve, the Hilbert transform may be complex, so we
# take the real part
##

u(x,y)=c*(x+im*y)+2cauchy(ui,x+im*y)

m=80;x = linspace(-2.,2.,m);y = linspace(-2.,2.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'


Γ=Curve(Fun(x->exp(0.8im)*(x+x^2-1+im*(x-4x^3+x^4)/6)))
    z=Fun(Γ)
    c=im
    ui=[BasisFunctional(1);real(Hilbert())]\Any[0.;imag(c*z)]
    Gadfly.plot(ApproxFun.layer(Γ),
                layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-3:.05:3])))




##
#  Two intervals requires explicitely stating the space (for now)
##



Γ=Interval(-1.,-0.5)∪Interval(0.5,1.)
z=Fun(Γ)


u(x,y)=c*(x+im*y)+2cauchy(ui,x+im*y)

m=80;x = linspace(-2.,2.,m);y = linspace(-1.,1.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'

k=20;
    c=exp(-k/50im)
    ui=[BasisFunctional(1);
        BasisFunctional(2);
        Hilbert()]\[0.;0.;imag(c*z)]
    Gadfly.plot(ApproxFun.layer(Γ),
                layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-2.5:.05:2.5])))



Γ=Interval(-1.,0.)∪Interval(0.5im,1.)
z=Fun(Γ)


u(x,y)=c*(x+im*y)+2cauchy(ui,x+im*y)

m=80;x = linspace(-2.,2.,m);y = linspace(-2.,1.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'

k=20;
    c=exp(-k/50im)
    ui=[BasisFunctional(1);
        BasisFunctional(2);
        real(Hilbert())]\[0.;0.;imag(c*z)]
    Gadfly.plot(ApproxFun.layer(Γ),
                layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-2.5:.05:2.5])))




1


Γ=Interval(-im,1.0-im)∪Curve(Fun(x->exp(0.8im)*(x+x^2-1+im*(x-4x^3+x^4)/6)))
z=Fun(Γ)
u(x,y)=c*(x+im*y)+2cauchy(ui,x+im*y)

k=20;
    c=exp(-π*k/50im)
    ui=[BasisFunctional(1);
        BasisFunctional(2);
        real(Hilbert())]\Any[0.;0.;imag(c*z)]

m=100;x = linspace(-3.,3.,m);y = linspace(-3.,2.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'
    myplot=Gadfly.plot(ApproxFun.layer(Γ),
                layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-5.:.07:4.])))

draw(SVG("/Users/solver/Desktop/myplot.svg", 4inch, 3inch), myplot)

Pkg.add("Cairo")

