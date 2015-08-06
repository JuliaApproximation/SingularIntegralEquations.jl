using ApproxFun,SingularIntegralEquations,Gadfly

Γ=Interval()
z=Fun(Γ)


u(x,y)=c*(x+im*y)+2cauchy(ui,x+im*y)

m=100;x = linspace(-2.,2.,m);y = linspace(-1.,1.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'

k=80;
    c=exp(-k/50im)
    ui=[BasisFunctional(1);Hilbert()]\[0.;imag(c*z)]
    Gadfly.plot(ApproxFun.layer(Γ),
    layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-2.5:.05:2.5])))

