export Cauchy

immutable Cauchy{D<:FunctionSpace,R<:FunctionSpace} <: BandedOperator{Complex{Float64}}
    domainspace::D
    rangespace::R
end

Cauchy(ds::PeriodicDomain,rs::PeriodicDomain)=Cauchy(Laurent(ds),Laurent(rs))

domainspace(C::Cauchy)=C.domainspace
rangespace(C::Cauchy)=C.rangespace

function bandinds(C::Cauchy{Laurent,Laurent})
    ds=domain(domainspace(C));rs=domain(rangespace(C))
    @assert isa(ds,Circle)
    @assert isa(rs,Circle)
    c2=rs.center;c1=ds.center
    r2=rs.radius;r1=ds.radius   
    
    if c2==c1
        0,0
    else
        mx=abs(c2-c1)/r1+r2/r1    
        @assert mx < 1 ## rs is inside ds    
        b=int(log(100eps())/log(mx))
        0,2b
    end
end

function addentries!(C::Cauchy{Laurent,Laurent},A::ShiftArray,kr::Range)
    ds=domain(domainspace(C));rs=domain(rangespace(C))
    @assert isa(ds,Circle)
    @assert isa(rs,Circle)
    
    c2=rs.center;c1=ds.center
    r2=rs.radius;r1=ds.radius
                
    if isapprox(c2,c1)
        s=r1>r2
        if s
            r=r2/r1
        else
            r=r1/r2
        end        
        
        for k=kr
            if isodd(k)==s
                A[k,0]+=(s?1:-1)*r^(div(k,2))
            end
        end
    else
        mx=abs(c2-c1)/r1+r2/r1     
        b=int(log(100eps())/log(mx))
        @assert mx < 1 ## rs is inside ds, need to implement other direction
        cm=c2-c1
        
        for j=kr[1]:min(2b,kr[end])
            if isodd(j)
                for k=0:2:2b
                    j2=div(j-1,2)
                    k2=div(k,2)+j2                    
                    A[j,k]+=binomial(k2,j2)*cm^(k2-j2)*r2^j2/r1^k2
                end
            end
        end
    end
    A
end

    ## Cauchy(s,d)
Cauchy(s::Bool,d)=(s?0.5:-0.5)*I -0.5im*Hilbert(d)
Cauchy(s::Int,d)=Cauchy(s==1,d)
Cauchy(s::Union(Int,Bool))=Cauchy(s,AnySpace())