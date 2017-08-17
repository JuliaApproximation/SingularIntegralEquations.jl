
if !isdefined(:scatteraux_loaded)
    global scatteraux_loaded = true


    function g3neumann(x,y,k)
      z = k*abs(y-x)
      if z < 1/16
          C,z2 = 2log(2)-2γ,z*z
          z4=z2*z2
          z6=z2*z4
          ret = k^2*complex( ( (C+2)/8 - (C+3)*z2/64 + (3C+11)*z4/4608 - (6C+25)*z6/442386 -(1/4-z2/32+z4/768-z6/36864)*log(k) )/π, 1/8-z2/64+3z4/4608-6z6/442368)
      else
          ret = im*k/4*hankelh1(1,k*abs(y-x))./abs(y-x) - g1(x,y)./abs(y-x).^2/π -g2(x,y).*logabs(y-x)/π
      end
      ret
    end


    function makegif(x,y,u,L;plotfunction=Main.Plots.PyPlot.contourf,seconds=1,cmap="seismic",vert=1)
        tm=string(time_ns())
        dr = pwd()*"/"*tm*"mov"
        mkdir(dr)

        umax = maxabs(u)
        fps = 24
        MLen = seconds*fps
        for k=1:MLen
            t = 2π/ω*(k-1)/fps
            Main.PyPlot.clf()
            Main.PyPlot.axes(aspect="equal")
            plot!(dom;color=:black,legend=false)
            plotfunction(x,y,real(u*exp(-im*ω*t)),L;vmin=-umax*vert,vmax=umax*vert,cmap="seismic")
            xlabel!("\$x\$");ylabel!("\$y\$")
            Main.PyPlot.savefig(dr * "/" * lpad(k,max(4,ceil(Int,log10(MLen))),0) * ".png";dpi=150,bbox_inches="tight")
        end
        # Requires: brew install imagemagick
        run(`convert -delay 6 -loop 0 $dr/\*.png $dr/scattering.gif`)
        run(`open $dr/scattering.gif`)
    end
end

#=
# This is for the landing Helmholtz gif. Solve time on the order of ~70 seconds.
N = 10
r = 5e-2
cr = exp(im*2π*[-0.5:N-1.5]/N)
crl = (1-2im*r)cr
crr = (1+2im*r)cr
dom = ∪(Segment,crl[1:2:end],crr[1:2:end]) ∪ ∪(Circle,cr[2:2:end],ones(length(cr[2:2:end]))r)∪Circle(0.,0.5)
function ui(x,y)
    c = 2exp(im*2π*(0.)/N)
    val = hankelh1(0,k*abs(complex(x,y)-c))
    for i=2:N
        c = 2exp(im*2π*(i-1.)/N)
        val = val + hankelh1(0,k*abs(complex(x,y)-c))
    end
    im/4*val
end
=#

#=
# This is the random screen in Figure 6.3 of Slevinsky & Olver 2015. Solve time on the order of ~70 seconds.
ccr = [-3.0,-2.4710248798864565,-1.7779535080542614,-0.999257770563108,-0.9160576190726175,-0.5056650643725802,0.7258681480228484,1.2291671942613505,1.3417993440008456,1.485081132919861,1.7601585357456848,2.9542404467603642,3.0]
dom = ∪(Segment,(ccr+(3-ccr[end-1])/2)[1:2:end-1],(ccr+(3-ccr[end-1])/2)[2:2:end])
=#
