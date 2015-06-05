⋅(d,z) = d[1]*z[1]+d[2]*z[2]

function g3neumann(x,y)
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
@vectorize_2arg Number g3neumann

function makegif(x,y,u,L;plotfunction=plot,seconds=1,cmap="seismic",vert=1)
    tm=string(time_ns())
    dr = pwd()*"/"*tm*"mov"
    mkdir(dr)

    umax = maxabs(u)
    fps = 24
    MLen = seconds*fps
    for k=1:MLen
        clf()
        t = 2π/ω*(k-1)/fps
        axes(aspect="equal")
        setplotter("PyPlot")
        ApproxFun.plot(dom;color="black")
        plotfunction(x,y,real(u*exp(-im*ω*t)),L;vmin=-umax*vert,vmax=umax*vert,cmap=cmap)
        xlabel("\$x\$");ylabel("\$y\$")
        savefig(dr * "/" * lpad(k,int(ceil(log10(MLen))),0) * ".png";dpi=150,bbox_inches="tight")
    end
    # If it fails, try: brew install imagemagick
    run(`convert -delay 6 -loop 0 $dr/*.png $dr/scattering.gif`)
    run(`open $dr/scattering.gif`)
end
