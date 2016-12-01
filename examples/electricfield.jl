# This file calculates the field strength at the center of a Faraday cage. Warning, the run-time is long.
# See /Laplace.jl for a more interactive version.
using ApproxFun, SingularIntegralEquations

z_0 = 2.0
ui(x,y) = logabs(complex(x,y)-z_0)
g1(x,y) = 1/2

function electricfield(N,r)
    # Set the domains.
    cr = exp(im*2π*[0:N-1]/N)
    crl = (1-2r)cr
    crr = (1+2r)cr
    dom = ∪(Segment.(crl,crr))
    #dom = ∪(Circle.(cr,ones(length(cr))r))
    #dom = ∪(Segment.(crl[1:2:end],crr[1:2:end]) ∪ ∪(Circle.(cr[2:2:end],ones(length(cr[2:2:end]))r))

    sp = Space(dom)
    cwsp = CauchyWeight(sp⊗sp,0)
    uiΓ,⨍ = Fun(t->ui(real(t),imag(t))+0im,sp),DefiniteLineIntegral(dom)

    G = GreensFun(g1,cwsp;method=:Cholesky)

    ∂u∂n = ⨍[G]\uiΓ

    us(x,y) = -logkernel(∂u∂n,complex(x,y))/2
    ut(x,y) = ui(x,y) + us(x,y)
    h = 1e-5
    abs((ut(h,0.)-ut(-h,0.))/2h)
end


NNr1,NTr1 = [4:15],[4:15,17:2:20,30:10:80]
ENr1 = Float64[@time electricfield(N,1e-1im) for N in NNr1]
println("Done ENr1")
ETr1 = Float64[@time electricfield(N,1e-1) for N in NTr1]
println("Done ETr1")

NNr2,NTr2 = [[5:10],[15:5:20],[30:10:80]],[[5:10],[15:5:20],[30:10:100]]
ENr2 = Float64[@time electricfield(N,1e-2im) for N in NNr2]
println("Done ENr2")
ETr2 = Float64[@time electricfield(N,1e-2) for N in NTr2]
println("Done ETr2")

#Nr3 = [[5:10],[15:5:20],[30:10:90],[100:50:200]]
#@time Er3 = Float64[electricfield(N,1e-3) for N in Nr3]
#@time Er4 = Float64[electricfield(N,1e-4) for N in Nr3]
