⋅(d,z) = d[1]*z[1]+d[2]*z[2]

cts = 100
Γ,ψ = ones(cts),ones(cts)
[(Γ[l],ψ[l]) = (l*Γ[l-1],ψ[l-1]+1/l) for l=2:cts]

function FK0(z)
    abz = abs(z)
    if abz <=.25
        upd = 1.
        ret = upd
        l=1
        zt2 = -(.5abz)^2
        while norm(upd) >= eps()
            upd = zt2^l/Γ[l]^2
            ret += upd
            l+=1
        end
    else
        ret = besselj(0,abz)
    end
    ret
end
function GK0(z)
    abz = abs(z)
    if abz <= .25
        upd = 1.
        ret = 0upd
        l=1
        zt2 = -(.5abz)^2
        while norm(upd) >= eps()
            upd = zt2^l/Γ[l]^2*ψ[l]
            ret += upd
            l+=1
        end
    else
        ret = besselj(0,abz)*(log(abz/2)+γ)-pi/2*bessely(0,abz)
    end
    ret
end

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

    run(`ffmpeg -r 25 -i $dr/%d.png -b:v 10MiB -bufsize 10MiB $dr/out.mpg`)
    run(`open $dr/out.mpg`)
end