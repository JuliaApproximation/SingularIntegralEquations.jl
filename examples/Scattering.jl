# This file calculates scattering from the Helmholtz equation via the adaptive spectral method.
# Δu + k^2u = 0,
# u|Γ = 0,
# u^i = e^{im k x⋅d},
# u = u^i + u^s.
# The normal derivative ∂u∂n of the entire wave is calculated on the sound-hard line [-1,1].
# The scattered wave is calculated by convolving ∂u∂n with the fundamental solution.
# Then, the total wave is obtained by summing the incident and the scattered waves.

using ApproxFun,SIE
include("Scatteraux.jl")

k = 50.
ω = 2π
d = (1,-3)
d = d[1]/hypot(d[1],d[2]),d[2]/hypot(d[1],d[2])
ui(x,y) = exp(im*k*(d⋅(x,y)))


#=
    dom = Interval(-1.,1.)
    sp = Space(dom)
    cwsp = CauchyWeight{0}(sp⊗sp)
    uiΓ,⨍ = Fun(t->ui(real(t),imag(t)),sp),DefiniteLineIntegral(dom)

    g1(x,y) = -besselj0(k*abs(y-x))/2π
    g2(x,y) = (besselj0(k*abs(y-x))*(im*π/2+logabs(y-x))/2π - bessely0(k*abs(y-x))/4)/π
    g3(x,y) = im/4*hankelh1(0,k*abs(y-x))

    @time G = ProductFun(g1,cwsp) + ProductFun(g2,sp,sp;method=:convolution)
    L,f = ⨍[G],uiΓ

    @time ∂u∂n = L\f
    println("The length of ∂u∂n is: ",length(∂u∂n))
    us(x,y) = -linesum(g3,∂u∂n,complex(x,y))/π
    #us_old(x,y) = reshape(Fun(t->-g3(x-real(t),im*(y-imag(t)))*∂u∂n[t],ApproxFun.ArraySpace(space(∂u∂n),size(x)),length(∂u∂n)).coefficients[1:length(x)],size(x))
=#

#=
    N = 6
    r = rand(2N+1)
    cr = cumsum(r)
    ccr = -3+(cr-cr[1])*6/(cr[end]-cr[1]) # For a nice plot, try: [-3.0,-2.4710248798864565,-1.7779535080542614,-0.999257770563108,-0.9160576190726175,-0.5056650643725802,0.7258681480228484,1.2291671942613505,1.3417993440008456,1.485081132919861,1.7601585357456848,2.9542404467603642,3.0]
    dom = UnionDomain(Interval(ccr+(3-ccr[end-1])/2)[1:2:end])
=#

    #N = 5
    #dom = Interval(-2.5-.5im,-1.5-.5im)∪Interval(-1.5+.5im,-.5+.5im)∪Interval(-.5-.5im,.5-.5im)∪Interval(.5+.5im,1.5+.5im)∪Interval(1.5-.5im,2.5-.5im)

#    N = 3
#    dom = ∪(Interval([-1.0-0.4im,0.1+0.4im,-0.9-0.5im],[-0.1+0.4im,1.0-0.4im,0.9-0.5im]))

    dom = ∪(Interval([-1.0-0.4im,-0.5-0.4im,0.1+0.4im,0.2+0.0im,-1.4-0.75im],[-0.1+0.4im,-0.2+0.0im,1.0-0.4im,0.5-0.4im,1.4-0.75im]))

    N = length(dom)
    sp = Space(dom)
    cwsp = CauchyWeight{0}(sp⊗sp)
    uiΓ,⨍ = Fun(t->ui(real(t),imag(t)),sp),DefiniteLineIntegral(dom)

    g1(x,y) = -besselj0(k*abs(y-x))/2π
    g2(x,y) = (besselj0(k*abs(y-x))*(im*π/2+logabs(y-x))/2π - bessely0(k*abs(y-x))/4)/π
    g3(x,y) = im/4*hankelh1(0,k*abs(y-x))

    @time G = GreensFun(g1,cwsp) + GreensFun(g2,sp⊗sp)

    L,f = ⨍[G],uiΓ

    @time ∂u∂n = L\f
    println("The length of ∂u∂n is: ",length(∂u∂n))

    us(x,y) = -linesum(g3,∂u∂n,complex(x,y))/π

#=
    ∂u∂nv = vec(∂u∂n)
    wsp = map(space,∂u∂nv)
function us_old(x,y)
    ret = Fun(t->-g3(x-real(t),im*(y-imag(t)))*∂u∂nv[1][t],ApproxFun.ArraySpace(wsp[1],size(x)),length(∂u∂nv[1])).coefficients[1:length(x)].*length(domain(sp[1]))/2
    for i=2:N
        ret += Fun(t->-g3(x-real(t),im*(y-imag(t)))*∂u∂nv[i][t],ApproxFun.ArraySpace(wsp[i],size(x)),length(∂u∂nv[i])).coefficients[1:length(x)]*length(domain(sp[i]))/2
    end
    reshape(ret,size(x))
end
=#

#=
∂u∂nvtest = vec(Fun(∂u∂n.coefficients,wsp))
function ustest(x,y)

    temp1 = vec(Fun(t->g1(x-real(t),im*(y-imag(t))),sp))
    temp2 = vec(Fun(t->g2(x-real(t),im*(y-imag(t))),sp))

    ret = -1/π*logkernel(temp1[1]*∂u∂nvtest[1],complex(x,y)) - linesum(temp2[1]*∂u∂nvtest[1])
    for i=2:N
        ret += -1/π*logkernel(temp1[i]*∂u∂nvtest[i],complex(x,y)) - linesum(temp2[i]*∂u∂nvtest[i])
    end
    ret
end
@vectorize_2arg Number ustest
=#
