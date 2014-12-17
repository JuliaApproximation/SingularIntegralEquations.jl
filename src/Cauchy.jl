export Cauchy

immutable Cauchy{D<:FunctionSpace,R<:FunctionSpace} <: BandedOperator{Complex{Float64}}
    domainspace::D
    rangespace::R
end

Cauchy(ds::PeriodicDomain,rs::PeriodicDomain)=Cauchy(LaurentSpace(ds),LaurentSpace(rs))

domainspace(C::Cauchy)=C.domainspace
rangespace(C::Cauchy)=C.rangespace

function bandinds(C::Cauchy{LaurentSpace,LaurentSpace})
    ds=domain(domainspace(C));rs=domain(rangespace(C))
    @assert isa(ds,Circle)
    @assert isa(rs,Circle)    
    @assert ds.center==rs.center    
    0,0
end

function addentries!(C::Cauchy{LaurentSpace,LaurentSpace},A::ShiftArray,kr::Range)
    ds=domain(domainspace(C));rs=domain(rangespace(C))
    @assert isa(ds,Circle)
    @assert isa(rs,Circle)    
    @assert ds.center==rs.center    
    s=ds.radius>rs.radius
    if s
        r=rs.radius/ds.radius
    else
        r=ds.radius/rs.radius
    end        
    
    for k=kr
        if isodd(k)==s
            A[k,0]+=(s?1:-1)*r^(div(k,2))
        end
    end
    A
end

    ## Cauchy(s,d)
Cauchy(s::Bool,d)=(s?0.5:-0.5)*I -0.5im*Hilbert(d)
Cauchy(s::Int,d)=Cauchy(s==1,d)
Cauchy(s::Union(Int,Bool))=Cauchy(s,AnySpace())