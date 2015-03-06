⋅(d,z) = d[1]*z[1]+d[2]*z[2]

function makegif(x,y,u,L;plotfunction=plot,seconds=1)
    tm=string(time_ns())
    dr = pwd()*"/"*tm*"mov"
    mkdir(dr)

    umax = maxabs(u)
    fps = 24
    MLen = seconds*fps
    for k=1:MLen
        t = 2π/ω*(k-1)/fps
        axes(aspect="equal")
        for doma in dom.domains
            line = [doma.a,doma.b]
            plot(real(line),imag(line),"-k",linewidth=2.0)
        end
        plotfunction(x,y,real(u*exp(-im*ω*t)),L;vmin=-umax,vmax=umax)
        xlabel("\$x\$");ylabel("\$y\$")
        savefig(dr * "/" * lpad(k,int(ceil(log10(MLen))),0) * ".png";dpi=150,bbox_inches="tight")
        clf()
    end
    # If it fails, try: brew install imagemagick
    run(`convert -delay 6 -loop 0 $dr/*.png $dr/scattering.gif`)
    run(`open $dr/scattering.gif`)
end
