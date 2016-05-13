using Plots,ApproxFun,SingularIntegralEquations;  gadfly()

##
#  Ideal fluid flow consists of level sets of the imagainary part of a function
# that is asymptotic to c*z and whose imaginary part vanishes on Γ
#
#
# On the unit interval, -2*hilbert(ui) gives the imaginary part of cauchy(ui,z)
#  So if we want to find ui defined on Γ so that hilbert(ui) = imag(c*z)
#  then c*z + 2cauchy(u,z) vanishes on Γ
##



u(x,y)=α*(x+im*y)+2cauchy(ui,x+im*y)

m=80;x = linspace(-2.,2.,m);y = linspace(-1.,1.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'

c

ui|>space
Number(c)+hilbert(ui)(0.25+0.125im)

imag(α*(0.25+0.125im))

Hilbert()*ui
x=Fun()
f=Fun(exp)*sqrt(1-x^2)
cauchy(f,.1+.000001im)
2cauchy(f,.1,+)
f(.1)-im*hilbert(f,.1)

d=Circle()
z=Fun(d)
f=exp(real(z))
f(exp(im*0.1))

2imag(cauchy(f,exp(im*0.1),+))+real(hilbert(f,exp(im*0.1)))


im*sum(f/z)
linesum(f)

∪

k=50
    Γ=Interval(0.,1+0.5im)
    z=Fun(Γ)
    α=exp(-π*k/50im)
    c,ui=[1 Hilbert()]\imag(α*z)
    plot(Γ)
    contour!(x,y,imag(u(xx,yy))).o


##
# On an arc, the Hilbert transform no longer gives the imaginary part
#  However, pseudohilbert gives the imaginary part of pseudocauchy
#  So if we want to find ui defined on Γ so that pseudohilbert(ui) = imag(c*z)
#  then c*z + 2pseudocauchy(u,z) vanishes on Γ
##


u(x,y)=α*(x+im*y)+2pseudocauchy(ui,x+im*y)

m=80;x = linspace(-2.,2.,m);y = linspace(-2.,2.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'

k=227;
    Γ=0.5+exp(im*Interval(0.1,-42))
    z=Fun(Γ)
    α=exp(-k/50im)
    c,ui=[1 PseudoHilbert()]\imag(α*z)
    plot(Γ)
    contour!(x,y,imag(u(xx,yy))).o

##
#  Circle
##

Γ=Circle()
z=Fun(Fourier(Γ))


u(x,y)=α*(x+im*y)+2cauchy(ui,x+im*y)

m=80;x = linspace(-2.,2.,m);y = linspace(-2.,2.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'

k=239;
    α=exp(-k/45im)
    c,ui=[0 BasisFunctional(1);
          1 real(Hilbert())]\[0.,imag(α*z)]

    plot(Γ)
    contour!(x,y,imag(u(xx,yy))).o

##
# On a curve, the Hilbert transform may be complex, so we
# take the real part
##

u(x,y)=α*(x+im*y)+2cauchy(ui,x+im*y)

m=80;x = linspace(-2.,2.,m);y = linspace(-2.,2.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'


Γ=Curve(Fun(x->exp(0.8im)*(x+x^2-1+im*(x-4x^3+x^4)/6)))
    z=Fun(Γ)
    α=im
    c,ui=[1 real(Hilbert())]\imag(α*z)
    plot(Γ)
    contour!(x,y,imag(u(xx,yy))).o




##
#  Two intervals requires explicitely stating the space (for now)
##

Γ=Interval(-1.,-0.5)∪Interval(-0.3,1.)
z=Fun(Γ)

ds=PiecewiseSpace(map(d->JacobiWeight(0.5,0.5,Ultraspherical{1}(d)),Γ.domains))


u(x,y)=α*(x+im*y)+2cauchy(ui,x+im*y)

m=80;x = linspace(-2.,2.,m);y = linspace(-1.,1.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'

k=114;
    α=exp(k/50*im)
    a,b,ui=[ones(Γ[1]) ones(Γ[2]) Hilbert(ds)]\imag(α*z)
    plot(Γ)
    contour!(x,y,imag(u(xx,yy))).o


Γ=Interval(-1.,0.)∪Interval(0.5im,1.)
z=Fun(Γ)
ds=PiecewiseSpace(map(d->JacobiWeight(0.5,0.5,Ultraspherical{1}(d)),Γ.domains))

u(x,y)=α*(x+im*y)+2cauchy(ui,x+im*y)

m=80;x = linspace(-2.,2.,m);y = linspace(-1.,1.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'


k=114;
    α=exp(k/50*im)
    a,b,ui=[ones(Γ[1]) ones(Γ[2]) real(Hilbert(ds))]\imag(α*z)
    plot(Γ)
    contour!(x,y,imag(u(xx,yy)))




Γ=Interval(-im,1.0-im)∪Curve(Fun(x->exp(0.8im)*(x+x^2-1+im*(x-4x^3+x^4)/6)))
z=Fun(Γ)

ds=PiecewiseSpace([JacobiWeight(0.5,0.5,Ultraspherical{1}(Γ[1])),
                       MappedSpace(Γ[2],JacobiWeight(0.5,0.5,Ultraspherical{1}()))])


m=80;x = linspace(-2.,2.,m);y = linspace(-3.,2.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'


k=114;
    α=exp(k/50*im)
    a,b,ui=[ones(Γ[1]) ones(Γ[2]) real(Hilbert(ds))]\imag(α*z)
Gadfly.plot(ApproxFun.layer(Γ),
                layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-3.:.05:4.])))



## Channel






# TODO: rename
using ApproxFun,SingularIntegralEquations,SO,Plots
    import ApproxFun:RealUnivariateSpace,ComplexBasis,canonicalspace,bandinds,addentries!,conversion_rule,
            spacescompatible,toeplitz_addentries!,domainscompatible,rangespace, ConcreteConversion,
            coefficients
    import SingularIntegralEquations:cauchy,ConcreteHilbert, stieltjes
immutable FourierDirichlet{DD} <: RealUnivariateSpace{DD}
    domain::DD
end

FourierDirichlet()=FourierDirichlet(PeriodicInterval())

    spacescompatible(a::FourierDirichlet,b::FourierDirichlet)=domainscompatible(a,b)

    canonicalspace(S::FourierDirichlet)=Fourier(domain(S))

    bandinds{DD}(::ConcreteConversion{FourierDirichlet{DD},Fourier{DD}})=-1,1
    function addentries!{DD}(C::ConcreteConversion{FourierDirichlet{DD},Fourier{DD}},A,kr::Range,::Colon)
        toeplitz_addentries!([1.],[0.,1.],A,kr)
    end


    conversion_rule{DD}(b::FourierDirichlet{DD},a::Fourier{DD})=b

    setdomain(F::FourierDirichlet,d::Domain)=FourierDirichlet(d)


    function coefficients(v::Vector,a::FourierDirichlet,b::Fourier)
        ret=zeros(typeof(v),length(v)+1)
        ret[2]=v[1]
        for k=2:length(v)
            ret[k-1]+=v[k]
            ret[k+1]+=v[k]
        end
        ret
    end





bandinds{L<:PeriodicLine}(H::ConcreteHilbert{FourierDirichlet{L}})=-2,2
    rangespace{L<:PeriodicLine}(H::ConcreteHilbert{FourierDirichlet{L}})=Fourier(domain(H))

    function addentries!{L<:PeriodicLine}(H::ConcreteHilbert{FourierDirichlet{L}},A,kr::Range,::Colon)
        if 1 in kr
            A[1,1]+=1
        end
        addentries!(real(Hilbert(FourierDirichlet(Circle()))),A,kr,:)
        A
    end


stieltjes{DD}(sp::FourierDirichlet{DD},cfs,z)=stieltjes(Fun(Fun(f,sp),Fourier),z)
    stieltjes{PL}(f::Fun{Fourier{PL}},z::Vector)=Complex128[stieltjes(f,zk) for zk in z]
    stieltjes{PL}(f::Fun{Fourier{PL}},z::Matrix)=reshape(stieltjes(f,vec(z)),size(z,1),size(z,2))::Matrix{Complex128}
    stieltjes{PL}(f::Fun{FourierDirichlet{PL}},z)=stieltjes(Fun(f,Fourier),z)



S=FourierDirichlet(PeriodicLine()-1.im)
PS=PiecewiseSpace([S,JacobiWeight(0.5,0.5,Ultraspherical{1}(Interval(-0.5im,1.+1.1im))),
                    JacobiWeight(0.5,0.5,Ultraspherical{1}(Interval(-0.5,-0.5+.5im)))])



H=Hilbert(PS)

f=Fun(z->imag(z+1.im),rangespace(H))
Γ=domain(PS)

a,b,ui=linsolve([ones(Γ[2]) ones(Γ[3]) real(H)],f;tolerance=1E-10)
m=80;x = linspace(-2.,3.,m);y = linspace(-1.,3.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'


u(x,y)=(x+im*y)+2cauchy(ui,x+im*y)
    vals=imag(u(xx,yy))

pyplot()
plot(Γ[2];color=:Blue,legend=false)
    plot!(Γ[3];color=:Blue)
    plot!(Interval(-2.-im,3.0-im);color=:Blue)
    contour!(x,y,vals;ylims=(-2.,3.),levels=-1.:.1:3.)

Main.PyPlot.savefig("halfg.png";dpi=200,bbox_inches="tight")

png("halfplane")

gadfly()
pdf("halfplane")

Gadfly.plot(ApproxFun.layer(Γ[2]),

            ApproxFun.layer(Interval(-2.-im,3.0-im)),
                layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-1.:0.05:2.])))


pieces(ui)[2]|>ApproxFun.plot
g=pieces(ui)[1]
    g=Fun(g.coefficients,space(g).space)
    g=Fun(g,Fourier)

g|>ApproxFun.plot



S1=MappedSpace(PeriodicLine()-1.im,FourierDirichlet())
S2=MappedSpace(PeriodicLine()+1.im,FourierDirichlet())

PS=PiecewiseSpace([S1,S2])
H=Hilbert(PS)

using SO

ds=PS[1]
rs=MappedSpace(domain(PS[2]),Fourier())
b=Fun([zeros(18);1.],ds);
    Fun(x->-stieltjes(b,x)/π,rs).coefficients|>chopm

g=Fun(b,MappedSpace(domain(PS[1]),Fourier()))
@which cauchy(b,5.0-1.im)
cauchy(g,5.0-1.im)
Fun(x->-stieltjes(b,x)/π,rs,20)

v1=chop(Fun(x->-stieltjes(b,x)/π,rs).coefficients,tol)

f=Fun(z->imag(z+1.im),rangespace(H))
Γ=domain(PS)

a,ui=linsolve([ones(Γ[2]) real(H)],f;tolerance=1E-5)
m=80;x = linspace(-2.,3.,m);y = linspace(-1.5,2.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'

u(x,y)=(x+im*y)+2cauchy(ui,x+im*y)
Gadfly.plot(ApproxFun.layer(Γ[2]),
            ApproxFun.layer(Interval(-2.-im,3.0-im)),
                layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-1.:0.05:2.])))




d=Circle()
f=Fun(exp,d)
g=Fun(cos,d)


s=z->imag(conj(z)*cauchy(f,z)+cauchy(g,z))
pyplot()
x=y=linspace(-1.1,1.1,40)
z=Float64[s(x+im*y) for x in x, y in y]

s(0.999999999exp(im*1.))-imag(exp(-im)*f(exp(im))+g(exp(im)))
s(0.999999999exp(im*1.))
exp(dual(0.99999999999exp(im*1.),exp(im*1.)))
z=dual(0.99999999999exp(im*1.),exp(im*1.))
conj(z(&exp(

exp(im*1.)*exp(exp(im*1.))
z=Fun(d)
Multiplication(Fun([0.,0.,1.],Fourier(d)),Fourier(d))
Multiplication(conj(z),Laurent(d))
imag(f(z) +  conj(z)*g(z))

g(z)==r(z)/2
f(z)==q(z)-z/2*r(z)


imag(q(z) +  (z+conj(z))/2*r(z))


imag(exp(z))
imag(conj(z)(1/abs(z)*f(z)      +z*f'(z)))+z*g(z)) == 0


realpart(imag(z*exp(z))           )
realpart(imag(conj(z)(1/abs(z)*f(z)      +z*f'(z)))+z*g(z))        )
dualpart(imag(conj(z)*exp(z))          )-(realpart(imag(conj(z)/abs(z)*exp(z)           + conj(z)*z*exp(z))))




conj(z)



s(dual(0.99999999999exp(im*1.),exp(im*1.)))
dualpart(cos((10E5)*dual(1,2)))+2*10E5*sin(10E5)
2exp(1.)
expm([1 1 1/2;
      0 1 1;
      0 0 1])
cauchy(f,exp(im),+)
cauchy(g,exp(im),+)
gui()
surface(x,y,z)
s(.1+.2im)


real(Hilbert())  # Im g(z)


