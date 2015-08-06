using ApproxFun,SingularIntegralEquations,PyPlot
    setplotter("PyPlot")

Γ=Interval()
z=Fun(Γ)
c=exp(-.5im)
ui=[BasisFunctional(1);Hilbert()]\[0.;imag(c*z)]

u(x,y)=c*(x+im*y)+2cauchy(ui,x+im*y)

m=200;x = linspace(-2.,2.,m);y = linspace(-1.,1.,m)
xx,yy = x.+0.*y',0.*x.+y'

clf();
    ApproxFun.plot(Γ)
    contour(x,y,imag(u(xx,yy)).';levels=[-2.5:.02:2.5])

