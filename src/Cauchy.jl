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
    
    if isapprox(c2,c1)
        0,0   # special form when centers coincide
    elseif r1>r2&&abs(c1-c2)<r1  # we are inside the circle, use Taylor series
        mx=abs(c2-c1)/r1+r2/r1    
        @assert mx < 1 ## rs is inside ds    
        b=int(log(100eps())/log(mx))
        0,2b
        # polys like in the space of polys, so upper triangular
    elseif r1<r2&&abs(c1-c2)<r2 # we surround the domain, use Hardy{False} series
        mx=r2/r1-abs(c1-c2)/r1
        @assert mx>1
        b=int(log(100eps())/log(1/mx))
        -2b,0
        # inverse polys decay at the same rate, so lower triangular      
    else  # There are two disjoint circles, map from Hardy{false}->Taylor
        mx=abs(c2-c1)/r1-r2/r1
        @assert mx > 1
        b=int(log(100eps())/log(1/mx))
        # we have 2bx2b coefficients
        -2b,2b
    end
end

binomialf(x,y)=1.0/((x+1)*beta(x-y+1.0,y+1.0))

function addentries!(C::Cauchy{Laurent,Laurent},A,kr::Range)
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
                A[k,k]+=(s?1:-1)*r^(div(k,2))
            end
        end
    elseif r1>r2&&abs(c1-c2)<r1  # we are inside the circle, use Taylor series
        mx=abs(c2-c1)/r1+r2/r1     
        b=int(log(100eps())/log(mx))
        @assert mx < 1 ## rs is inside ds, need to implement other direction
        cm=c2-c1
        
        for j=kr[1]:min(2b,kr[end])
            if isodd(j)
                for k=0:2:2b
                    j2=div(j-1,2)
                    k2=div(k,2)+j2                    
                    A[j,k+j]+=binomialf(k2,j2)*cm^(k2-j2)*r2^j2/r1^k2
                end
            end
        end
    elseif r1<r2&&abs(c1-c2)<r2 # we surround the domain, use Hardy{False} series
        mx=r2/r1-abs(c2-c1)/r1
        @assert mx>1
        b=int(log(100eps())/log(1/mx))
        cm=c1-c2
        for k=kr[1]:min(2b,kr[end])
            if iseven(k)
                for j=max(-2b,2-k):2:0
                    k2=div(k,2)-1   
                    j2=div(j,2)+k2
                    A[k,k+j]-=binomialf(k2,j2)*cm^(k2-j2)*r1^(j2+1)/r2^(k2+1)
                end
            end
        end            
    else
        mx=abs(c2-c1)/r1-r2/r1
        @assert mx > 1
        b=int(log(100eps())/log(1/mx))
        cm=c1-c2        
        # we have 2bx2b coefficients
        for k=kr[1]:min(2b,kr[end])
            if isodd(k)
                for j=2-k:2:2b-k
                    k2=div(k+1,2)
                    j2=div(j+k,2)
                    A[k,k+j]+=r2^(k2-1)/cm^(k2-1)*(-1)^(j2+1)*r1^j2/cm^j2*binomialf(k2+j2-2,j2-1)
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


