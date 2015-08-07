using ApproxFun,SingularIntegralEquations
,Gadfly



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

k=102;
    c=exp(-k/50im)
    ui=[BasisFunctional(1);Hilbert()]\[0.;imag(c*z)]
    Gadfly.plot(ApproxFun.layer(Γ),
    layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-2.5:.05:2.5])))

##
# On the arc, the Hilbert transform no longer gives the imaginary part
#  However, pseudohilbert gives the imaginary part of pseudocauchy
#  So if we want to find ui defined on Γ so that pseudohilbert(ui) = imag(c*z)
#  then c*z + 2pseudocauchy(u,z) vanishes on Γ
##


u(x,y)=c*(x+im*y)+2pseudocauchy(ui,x+im*y)

m=80;x = linspace(-2.,2.,m);y = linspace(-2.,2.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'

k=201;
    Γ=exp(im*Interval(0.1,3))
    z=Fun(Γ)
    c=exp(-k/50im)
    ui=[BasisFunctional(1);PseudoHilbert()]\[0.;imag(c*z)]
    Gadfly.plot(ApproxFun.layer(Γ),
    layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-3:.05:3])))




m=80;x = linspace(-2.,2.,m);y = linspace(-2.,2.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'

k=201;
    Γ=Curve(Fun(x->x+im*(x+2)^2))
    z=Fun(Γ)
    c=exp(-k/50im)
    ui=real(H)\imag(c*z)




a,b=first(Γ),last(Γ)
w=1/(sqrt(abs(z-a))*sqrt(abs(z-b)))

real(hilbert(w,4*im))

H=Hilbert(space(w))




[1:10,1:10]
(real(H)*w)[4im]

H[1:10,1:10]
H\imag(c*z)

(H*w)[4im]

[1:10,1:10]


2cauchy(+1,w,4im)

im*(cauchy(+1,w,4im)+cauchy(-1,w,4im))

f=w
fm=Fun(f.coefficients,space(f).space)

im*(cauchy(+1,fm,0.)+cauchy(-1,fm,0.))
hilbert(fm,0.)
stieltjes(fm,3.)
-2π*im*cauchy(fm,3.)

z=4im
rts=complexroots(domain(f).curve-z)
di=domain(fm)
    mapreduce(rt->in(rt,di)?hilbert(fm,rt):-stieltjes(fm,rt)/π,+,rts)



w[4im]

cauchy(+1,w,4im)

+cauchy(-1,w,4im)


ApproxFun.plot(Γ)

