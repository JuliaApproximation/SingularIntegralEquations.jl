

## cauchy


#TODO: use hilbert instead of cauchy(g,-1)
function cauchy{L<:PeriodicLine,S,T}(f::Fun{MappedSpace{S,L,T}},z::Number,s...)
    g=Fun(f.coefficients,setdomain(space(f).space,Circle()))
    cauchy(g,mappoint(domain(f),Circle(),z),s...)-hilbert(g,-1)/(2im)
    # use im*(C^+ + C^-)=H
    #-cauchy(g,-1)
end


function hilbert{L<:PeriodicLine,S,T}(f::Fun{MappedSpace{S,L,T}},z::Number)
    g=Fun(f.coefficients,setdomain(space(f).space,Circle()))
    hilbert(g,mappoint(domain(f),Circle(),z))-hilbert(g,-1)
    # use im*(C^+ + C^-)=H
    #-cauchy(g,-1)
end



# we use the fact that C^± (z^k + z^(k-1)) = z^k + z^(k-1) and 0
# for k > 0 and
# C^± (z^k + z^(k-1)) = 0 and -z^k - z^(k-1)
# for k < 0, the formula H = im*C^+  +  im*C^-
# and C± 1 = ±1/2  (understood as a PV integral) so that H 1 = 0


bandinds{S,T}(H::Hilbert{PeriodicLineDirichlet{S,T}})=0,0
rangespace{S,T}(H::Hilbert{PeriodicLineDirichlet{S,T}})=domainspace(H)


function addentries!{T}(H::Hilbert{PeriodicLineDirichlet{false,T}},A,kr::Range,::Colon)
    for k=kr
        if iseven(k)  # negative terms
            A[k,k] += -im
        elseif k > 0 # positive terms
            A[k,k] += im
        end
    end
    A
end
