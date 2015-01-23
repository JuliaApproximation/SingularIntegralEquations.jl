export Cauchy

immutable Cauchy{D<:FunctionSpace,R<:FunctionSpace} <: BandedOperator{Complex{Float64}}
    data::BandedMatrix{Complex{Float64}}
    domainspace::D
    rangespace::R
end


    ## Cauchy(s,d)
#C^+-C^- = I
#C^+ + C^- = -im*H
Cauchy(s::Bool,d)=(s?0.5:-0.5)*I +(-0.5im)*Hilbert(d)
Cauchy(s::Int,d)=Cauchy(s==1,d)
Cauchy(s::Union(Int,Bool))=Cauchy(s,UnsetSpace())
Cauchy(ds::PeriodicDomain,rs::PeriodicDomain)=Cauchy(Laurent(ds),Laurent(rs))


domainspace(C::Cauchy)=C.domainspace
rangespace(C::Cauchy)=C.rangespace
bandinds(C::Cauchy)=bandinds(C.data)


function Cauchy(DS::Laurent,RS::Laurent)
    ds=domain(DS);rs=domain(RS)
    @assert isa(ds,Circle)
    @assert isa(rs,Circle)
    
    c2=rs.center;c1=ds.center
    r2=rs.radius;r1=ds.radius
                
    if r1>r2&&abs(c1-c2)<r1  # we are inside the circle, use Taylor series
        M=interior_cauchy(ds,rs)
    elseif r1<r2&&abs(c1-c2)<r2 # we surround the domain, use Hardy{False} series
        M=exterior_cauchy(ds,rs)       
    else
        M=disjoint_cauchy(ds,rs)
    end
    
    Cauchy(M,DS,RS)
end

addentries!(C::Cauchy,A,kr)=addentries!(C.data,A,kr)




## Special cases

function exterior_cauchy(b::Circle,a::Circle)
    c=b.center
    r=b.radius

    S=Fun([0.0,0,1],a)  # Shift to use bandedness
    ret=Array(Fun{Laurent,Complex{Float64}},300)
    ret[1]=Fun(z->(r/(z-c)),a)
    n=1
    m=length(ret[1])-2
    f1=ret[1]*S
    while norm(ret[n].coefficients)>100eps()
        n+=1
        if n > length(ret)
            newret=Array(Fun{Laurent,Complex{Float64}},2length(ret))
            newret[1:length(ret)]=ret
            newret=ret
        end
        ret[n]=chop!(f1*ret[n-1],100eps())
        m=max(m,length(ret[n])-2)
    end

    M=bazeros(Complex{Float64},2n,2n,m,0)
    #j+2k-2≤2n
    #j≤2(n-k)+2
    for k=1:n,j=2:2:min(length(ret[k].coefficients),2(n-k)+2)
        M[j+2k-2,2k]=-ret[k].coefficients[j]
    end
    M
end

function interior_cauchy(a::Circle,b::Circle)
    c=a.center
    r=a.radius


    z=Fun(z->(z-c)/r,b)

    ret=Array(Fun{Laurent,Complex{Float64}},300)
    ret[1]=ones(b)
    n=1
    m=0

    while norm(ret[n].coefficients)>100eps()
        n+=1
        if n > length(ret)
            # double preallocated ret
            newret=Array(Fun{Laurent,Complex{Float64}},2length(ret))
            newret[1:length(ret)]=ret
            ret=newret
        end
        ret[n]=z*ret[n-1]  #will be length 2n-1


        # find bandwidth by checking how many coefficients are zero
        # we jump over negative coefficients
        for j=1:2:2n-1
            if norm(ret[n].coefficients[j])>100eps()
                m=max(m,2n-j-1)
                break
            end
        end
    end

    M=bazeros(Complex{Float64},2n-1,2n-1,0,m)
    for k=1:n,j=max(1,2k-1-m):2:2k-1
        M[j,2k-1]=ret[k].coefficients[j]
    end

    M
end

function disjoint_cauchy(a::Circle,b::Circle)
    c=a.center
    r=a.radius

    f=Fun(z->r/(z-c),b)

        ret=Array(Fun{Laurent,Complex{Float64}},300)
    ret[1]=f
    n=1

    l=length(f)-2   #lower bandwidth
    u=1             #upper bandwidth

    while norm(ret[n].coefficients)>100eps()
        n+=1
        if n > length(ret)
            # double preallocated ret
                newret=Array(Fun{Laurent,Complex{Float64}},2length(ret))
                newret[1:length(ret)]=ret
                newret=ret
        end
    ret[n]=chop!(f*ret[n-1],100eps())  #will be length 2n-1
    u=max(u,length(ret[n])-2n)   # upper bandwidth

        # find bandwidth by checking how many coefficients are zero
        # we jump over negative coefficients
        for j=1:2:length(ret[n])
            if norm(ret[n].coefficients[j])>100eps()
                l=max(l,2n-j)
                break
            end
        end
    end

    M=bazeros(Complex{Float64},2n-1,2n,l,u)
    for k=1:n,j=max(1,2k-u):2:min(length(ret[k]),2n-1)
            M[j,2k]=-ret[k].coefficients[j]
    end
    M
end

