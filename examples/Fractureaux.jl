cts = 100
Γ,ψ = ones(cts),ones(cts)
[(Γ[k],ψ[k]) = (k*Γ[k-1],ψ[k-1]+1/k) for k=2:cts]

function FK1(z)
    if abs(z) <=.25
        upd = 1.
        ret = upd
        l=1
        zt2 = (.5z)^2
        while norm(upd) >= eps()
            upd = zt2^l/(Γ[l]*Γ[l+1])
            ret += upd
            l+=1
        end
    else
        ret = 2besseli(1,abs(z))/abs(z)
    end
    .5ret
end
function GK1(z)
    if abs(z) <= .25
        upd = 1.
        ret = upd
        l=1
        zt2 = (.5z)^2
        while norm(upd) >= eps()
            upd = zt2^l/(Γ[l]*Γ[l+1])*(ψ[l]+ψ[l+1])
            ret += upd
            l+=1
        end
    else
        abz = abs(z)
        ret = 4/abz^2 - 4(besselk(1,abz)-besseli(1,abz)*(log(abz/2)+γ))/abz
    end
    .5ret
end