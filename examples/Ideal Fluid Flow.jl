using ApproxFun,SingularIntegralEquations,Gadfly
set_default_plot_format(:svg)

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


k=101;
    Γ=Interval(0.,1+0.5im)
    z=Fun(Γ)
    α=exp(-π*k/50im)
    c,ui=[1 Hilbert()]\imag(α*z)
    Gadfly.plot(ApproxFun.layer(Γ),
                layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-2.5:.05:2.5])))
##
# On an arc, the Hilbert transform no longer gives the imaginary part
#  However, pseudohilbert gives the imaginary part of pseudocauchy
#  So if we want to find ui defined on Γ so that pseudohilbert(ui) = imag(c*z)
#  then c*z + 2pseudocauchy(u,z) vanishes on Γ
##


u(x,y)=α*(x+im*y)+2pseudocauchy(ui,x+im*y)

m=80;x = linspace(-2.,2.,m);y = linspace(-2.,2.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'

k=217;
    Γ=0.5+exp(im*Interval(0.1,5))
    z=Fun(Γ)
    α=exp(-k/50im)
    c,ui=[1 PseudoHilbert()]\imag(α*z)
    Gadfly.plot(ApproxFun.layer(Γ),
                layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-3:.05:3])))


##
#  Circle
##

Γ=Circle()
z=Fun(Fourier(Γ))


u(x,y)=α*(x+im*y)+2cauchy(ui,x+im*y)

m=80;x = linspace(-2.,2.,m);y = linspace(-2.,2.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'

k=154;
    α=exp(-k/45im)
    c,ui=[0 BasisFunctional(1);
          1 real(Hilbert())]\[0.,imag(α*z)]

    Gadfly.plot(ApproxFun.layer(Γ),
                layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-2.5:.05:2.5])))

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
    Gadfly.plot(ApproxFun.layer(Γ),
                    layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-3:.05:3])))




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
    Gadfly.plot(ApproxFun.layer(Γ),
                layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-3.:.05:3.])))


Γ=Interval(-1.,0.)∪Interval(0.5im,1.)
z=Fun(Γ)
ds=PiecewiseSpace(map(d->JacobiWeight(0.5,0.5,Ultraspherical{1}(d)),Γ.domains))

u(x,y)=α*(x+im*y)+2cauchy(ui,x+im*y)

m=80;x = linspace(-2.,2.,m);y = linspace(-1.,1.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'


k=114;
    α=exp(k/50*im)
    a,b,ui=[ones(Γ[1]) ones(Γ[2]) real(Hilbert(ds))]\imag(α*z)
    Gadfly.plot(ApproxFun.layer(Γ),
                layer(x=x,y=y,z=imag(u(xx,yy)),Main.Gadfly.Geom.contour(levels=[-3.:.05:3.])))




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
using ApproxFun,SingularIntegralEquations,SO
    import ApproxFun:RealUnivariateSpace,ComplexBasis,canonicalspace,bandinds,addentries!,conversion_rule,
            spacescompatible,toeplitz_addentries!,domainscompatible,rangespace
    import SingularIntegralEquations:cauchy
immutable FourierDirichlet <: RealUnivariateSpace
    domain::PeriodicDomain
end

FourierDirichlet()=FourierDirichlet(PeriodicInterval())

    spacescompatible(a::FourierDirichlet,b::FourierDirichlet)=domainscompatible(a,b)

    canonicalspace(S::FourierDirichlet)=Fourier(domain(S))

    bandinds(::Conversion{FourierDirichlet,Fourier})=-1,1
    function addentries!(C::Conversion{FourierDirichlet,Fourier},A,kr::Range)
        toeplitz_addentries!([1.],[0.,1.],A,kr)
    end


    conversion_rule(b::FourierDirichlet,a::Fourier)=b





bandinds{L<:PeriodicLine,T}(H::Hilbert{MappedSpace{FourierDirichlet,L,T}})=-2,2
    rangespace{L<:PeriodicLine,T}(H::Hilbert{MappedSpace{FourierDirichlet,L,T}})=MappedSpace(domain(H),Fourier())

    function addentries!{L<:PeriodicLine,T}(H::Hilbert{MappedSpace{FourierDirichlet,L,T}},A,kr::Range)
        if 1 in kr
            A[1,1]+=1
        end
        addentries!(real(Hilbert(FourierDirichlet(Circle()))),A,kr)
        A
    end


cauchy(f::Fun{FourierDirichlet},z)=cauchy(Fun(f,Fourier),z)
cauchy{PL,T}(f::Fun{MappedSpace{Fourier,PL,T}},z::Vector)=Complex128[cauchy(f,zk) for zk in z]
cauchy{PL,T}(f::Fun{MappedSpace{Fourier,PL,T}},z::Matrix)=reshape(cauchy(f,vec(z)),size(z,1),size(z,2))::Matrix{Complex128}
cauchy{PL,T}(f::Fun{MappedSpace{FourierDirichlet,PL,T}},z)=cauchy(Fun(f,MappedSpace(domain(f),Fourier())),z)



S=MappedSpace(PeriodicLine()-1.im,FourierDirichlet())

PS=PiecewiseSpace([S,JacobiWeight(0.5,0.5,Ultraspherical{1}(Interval(-0.5im,1.+1.1im)))])
H=Hilbert(PS)

f=Fun(z->imag(z+1.im),rangespace(H))
Γ=domain(PS)

a,ui=linsolve([ones(Γ[2]) real(H)],f;tolerance=1E-10)
m=80;x = linspace(-2.,3.,m);y = linspace(-1.5,2.,m+1)
    xx,yy = x.+0.*y',0.*x.+y'

u(x,y)=(x+im*y)+2cauchy(ui,x+im*y)
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
