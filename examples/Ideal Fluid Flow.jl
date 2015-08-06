using ApproxFun,SingularIntegralEquations
using Gadfly

Γ=Interval()
z=Fun(Γ)


u(x,y)=c*(x+im*y)+2cauchy(ui,x+im*y)

m=80;x = linspace(-2.,2.,m);y = linspace(-1.,1.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'

k=102;
    c=exp(-k/50im)
    ui=[BasisFunctional(1);Hilbert()]\[0.;imag(c*z)]
    Gadfly.plot(ApproxFun.layer(Γ),
    layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-2.5:.05:2.5])))



Γ=exp(im*Interval(0.1,1))
z=Fun(Γ)
u(x,y)=c*(x+im*y)+2cauchy(ui,x+im*y)

m=80;x = linspace(-2.,2.,m);y = linspace(-1.,1.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'


sp=MappedSpace(Γ,JacobiWeight(-0.5,-0.5,ChebyshevDirichlet{1,1}()))
k=102;
    c=exp(-k/50im)
    ui=[BasisFunctional(1);real(Hilbert(sp))]\Any[0.;imag(c*z)]

Hilbert(JacobiWeight(-0.5,-0.5,ChebyshevDirichlet{1,1}()))[1:10,1:10]
Hilbert(sp)*Fun([1.],sp)

hilbert(Fun([1.],sp),exp(0.1im))
cauchy(+1,Fun([1.],sp),exp(0.1im))-cauchy(-1,Fun([1.],sp),exp(0.1im))
Fun([1.],sp)[exp(0.1im)]
[1:10,1:10]

ui
hilbert(ui,exp(0.1im))
Gadfly.plot(ApproxFun.layer(Γ),
    layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-2.5:.05:2.5])))



a,b=first(Γ),last(Γ)
w=1/(sqrt(abs(z-a))*sqrt(abs(z-b)))

sp=MappedSpace(Γ,JacobiWeight(-0.5,-0.5,ChebyshevDirichlet{1,1}()))
H=Hilbert(sp)
(H*Fun(w,sp))[1.]




(real(H)*Fun(w,sp))[exp(0.1im)]-real(hilbert(w,exp(0.1im)))

[1:10,1:10]
H*w
real(hilbert(w,exp(im*0.5)))+imag(2cauchy(false,w,exp(im*0.5)))
ApproxFun.RealOperator(H)[1:10,1:10]

using SO
H[1:20,1:20]|>chopm
2im*cauchy(Fun(w.coefficients,w.space.space),tocanonical(w,Inf))
cauchy(false,w,exp(im*0.5))+cauchy(true,w,exp(im*0.5))







[1:20,1:20]

u(x,y)=c*(x+im*y)+2cauchy(ui,x+im*y)

m=80;x = linspace(-2.,2.,m);y = linspace(-1.,1.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'

k=102;
c=exp(-k/50im)
ui=[BasisFunctional(1);Hilbert()]\Any[0.;imag(c*z)]
Gadfly.plot(ApproxFun.layer(Γ),
    layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-2.5:.05:2.5])))

