⋅(d,z) = d[1]*z[1]+d[2]*z[2]

function tomovie(x,y,u,L;plotfunction=plot,seconds=1)
    tm=string(time_ns())
    dr = pwd()*"/"*tm*"mov"
    mkdir(dr)

    line = [dom.a,dom.b]
    MLen = seconds*25
    for k=1:MLen+2
        t = 2π/ω*(k-1)/24
        plot(line,0line,"-k",linewidth=2.0)
        plotfunction(x,y,real(u*exp(-im*ω*t)),L)
        xlabel("\$x\$");ylabel("\$y\$")
        savefig(dr * "/" * string(int(k)) * ".png",dpi=150)
        clf()
    end

    run(`ffmpeg -r 25 -i $dr/%d.png -b:v 10MiB $dr/out.mpg`)
    run(`open $dr/out.mpg`)
end
