

## stieltjes


function stieltjes(S::Space{<:PeriodicLine},f::AbstractVector,z::Number)
    S2=setdomain(S,Circle())
    stieltjes(S2,f,mappoint(domain(S),Circle(),z))+hilbert(S2,f,-1)*π
end


function hilbert(S::Space{<:PeriodicLine},f::AbstractVector,z::Number)
    S2=setdomain(S,Circle())
    hilbert(S2,f,mappoint(domain(f),Circle(),z))-hilbert(S2,f,-1)
end

function stieltjes(S::SumSpace{<:Any,<:PeriodicLine},f::AbstractVector,z::Number)
    S2=setdomain(S,Circle())
    stieltjes(S2,f,mappoint(domain(S),Circle(),z))+hilbert(S2,f,-1)*π
end

function hilbert(S::SumSpace{<:Any,<:PeriodicLine},f::AbstractVector,z::Number)
    S2=setdomain(S,Circle())
    hilbert(S2,f,mappoint(domain(f),Circle(),z))-hilbert(S2,f,-1)
end



# we use the fact that C^± (z^k + z^(k-1)) = z^k + z^(k-1) and 0
# for k > 0 and
# C^± (z^k + z^(k-1)) = 0 and -z^k - z^(k-1)
# for k < 0, the formula H = im*C^+  +  im*C^-
# and C± 1 = ±1/2  (understood as a PV integral) so that H 1 = 0


bandwidths(H::ConcreteHilbert{LaurentDirichlet{DD,RR}}) where {DD<:PeriodicLine,RR} =
    (0,0)
rangespace(H::ConcreteHilbert{LaurentDirichlet{DD,RR}}) where {DD<:PeriodicLine,RR} =
    domainspace(H)


function getindex(H::ConcreteHilbert{LaurentDirichlet{PeriodicLine{false,T},RR}},k::Integer,j::Integer) where {T,RR}
    if k==j && iseven(k)
        -T(im)
    elseif k==j && isodd(k) && k > 0
        T(im)
    else
        zero(T)
    end
end
