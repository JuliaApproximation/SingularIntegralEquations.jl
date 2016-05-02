

## stieltjes


function stieltjes{L<:PeriodicLine,SS}(S::Space{SS,L},f,z::Number,s...)
    S2=setdomain(S,Circle())
    stieltjes(S2,f,mappoint(domain(S),Circle(),z),s...)+hilbert(S2,f,-1)*π
end


function hilbert{L<:PeriodicLine,SS}(S::Space{SS,L},f,z::Number)
    S2=setdomain(S,Circle())
    hilbert(S2,f,mappoint(domain(f),Circle(),z))-hilbert(S2,f,-1)
end



# we use the fact that C^± (z^k + z^(k-1)) = z^k + z^(k-1) and 0
# for k > 0 and
# C^± (z^k + z^(k-1)) = 0 and -z^k - z^(k-1)
# for k < 0, the formula H = im*C^+  +  im*C^-
# and C± 1 = ±1/2  (understood as a PV integral) so that H 1 = 0


bandinds{DD<:PeriodicLine}(H::ConcreteHilbert{LaurentDirichlet{DD}})=0,0
rangespace{DD<:PeriodicLine}(H::ConcreteHilbert{LaurentDirichlet{DD}})=domainspace(H)


function getindex{T}(H::ConcreteHilbert{LaurentDirichlet{PeriodicLine{false,T}}},k::Integer,j::Integer)
    if k==j && iseven(k)
        -T(im)
    elseif k==j && isodd(k) && k > 0
        T(im)
    else
        zero(T)
    end
end
