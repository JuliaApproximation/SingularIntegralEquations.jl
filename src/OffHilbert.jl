export OffHilbert


#############
# OffHilbert implements:
#
#       OH f(z) := 1/π\int_Γ f(t)/(t-z) dt,  z ∉ Γ
#
############

immutable OffHilbert{D<:FunctionSpace,R<:FunctionSpace} <: BandedOperator{Complex{Float64}}
    data::BandedMatrix{Complex{Float64}}
    domainspace::D
    rangespace::R
    order::Int
end

addentries!(C::OffHilbert,A,kr)=addentries!(C.data,A,kr)

OffHilbert{D<:FunctionSpace,R<:FunctionSpace}(ds::D,rs::R) = OffHilbert(ds,rs,1)
OffHilbert{B<:BandedMatrix,D<:FunctionSpace,R<:FunctionSpace}(data::B,ds::D,rs::R) = OffHilbert(data,ds,rs,1)

OffHilbert(ds::PeriodicDomain,rs::PeriodicDomain,order)=OffHilbert(Laurent(ds),Laurent(rs),order)
OffHilbert(ds::PeriodicDomain,rs::PeriodicDomain)=OffHilbert(Laurent(ds),Laurent(rs))

domainspace(C::OffHilbert)=C.domainspace
rangespace(C::OffHilbert)=C.rangespace
bandinds(C::OffHilbert)=bandinds(C.data)


## OffHilbert


function OffHilbert(ds::JacobiWeight{Ultraspherical{1}},rs::FunctionSpace,order::Int)
    @assert ds.α==ds.β==0.5
    d = domain(ds)
    C = (.5(d.b-d.a))^(1-order) # probably this is right for all orders ≥ 2. Certainly so for 0,1.

    if order == 0
        z=Fun(identity,rs)
        x=tocanonical(ds,z)
        y=intervaloffcircle(true,x)
        yk,ykp1=y,y*y
        ret=Array(typeof(y),300)
        ret[1]=-.5logabs(2y)+.25real(ykp1)
        n,l,u = 1,length(ret[1])-1,0
        while norm(ret[n].coefficients)>100eps()
            n+=1
            if n > length(ret) resize!(ret,2length(ret)) end  # double preallocated ret
            ykp1*=y
            ret[n]=chop!(.5*(real(yk)/(n+1)-real(ykp1)/(n-1)) ,100eps())  #will be length 2n-1
            yk*=y
            u+=1   # upper bandwidth
            l=max(l,length(ret[n])-n)
        end
    elseif order == 1
        z=Fun(identity,rs)
        x=tocanonical(ds,z)
        y=intervaloffcircle(true,x)
        ret=Array(typeof(y),300)
        ret[1]=-y
        n,l,u = 1,length(ret[1])-1,0
        while norm(ret[n].coefficients)>100eps()
            n+=1
            if n > length(ret) resize!(ret,2length(ret)) end  # double preallocated ret
            ret[n]=chop!(y*ret[n-1],100eps())  #will be length 2n-1
            u+=1   # upper bandwidth
            l=max(l,length(ret[n])-n)
        end
    end

    M=bazeros(Complex{Float64},l+1,n,l,u)
    for k=1:n,j=1:length(ret[k])
        M[j,k]=C*ret[k].coefficients[j]
    end
    OffHilbert(M,ds,rs,order)
end


function OffHilbert(ds::JacobiWeight{ChebyshevDirichlet{1,1}},rs::FunctionSpace,order::Int)
    @assert ds.α==ds.β==-0.5
    d = domain(ds)
    C = (.5(d.b-d.a))^(1-order) # probably this is right for all orders ≥ 2. Certainly so for 0,1.

    if order == 0
        z=Fun(identity,rs)
        x=tocanonical(ds,z)
        y=intervaloffcircle(true,x)
        yk,ykp1=y,y*y
        ret=Array(typeof(y),300)
        ret[1]=-logabs(2y)
        ret[2]=-real(yk)
        ret[3]=chop!(-ret[1]-.5real(ykp1),100eps())
        n,l,u = 3,max(length(ret[1])-1,length(ret[2])-2,length(ret[3])-3),2
        while norm(ret[n].coefficients)>100eps()
            n+=1
            if n > length(ret) resize!(ret,2length(ret)) end  # double preallocated ret
            ykp1*=y
            ret[n]=chop!(real(ykp1)/(n-3)-real(yk)/(n-1),100eps())  #will be length 2n-1
            yk*=y
            u+=1   # upper bandwidth
            l=max(l,length(ret[n])-n)
        end
    elseif order == 1
        z=Fun(identity,rs)
        x=tocanonical(ds,z)
        y=intervaloffcircle(true,x)
        ret=Array(typeof(y),300)
        ret[1]=-1/sqrtx2(x)
        ret[2]=x*ret[1]+1
        ret[3]=2y
        n,l,u = 3,max(length(ret[1])-1,length(ret[2])-2,length(ret[3])-3),2
        while norm(ret[n].coefficients)>100eps()
            n+=1
            if n > length(ret) resize!(ret,2length(ret)) end  # double preallocated ret
            ret[n]=chop!(y*ret[n-1],100eps())  #will be length 2n-1
            u+=1   # upper bandwidth
            l=max(l,length(ret[n])-n)
        end
    end

    M=bazeros(Complex{Float64},l+3,n,l,u)
    for k=1:n,j=1:length(ret[k])
        M[j,k]=C*ret[k].coefficients[j]
    end
    OffHilbert(M,ds,rs,order)
end


function OffHilbert(DS::Laurent,RS::Laurent,order::Int)
    ds=domain(DS);rs=domain(RS)
    @assert isa(ds,Circle)
    @assert isa(rs,Circle)
    @assert order==1

    c2=rs.center;c1=ds.center
    r2=rs.radius;r1=ds.radius

    if r1>r2&&abs(c1-c2)<r1  # we are inside the circle, use Taylor series
        M=interior_cauchy(ds,rs)
    elseif r1<r2&&abs(c1-c2)<r2 # we surround the domain, use Hardy{False} series
        M=exterior_cauchy(ds,rs)
    else
        M=disjoint_cauchy(ds,rs)
    end

    OffHilbert(2im*M,DS,RS)
end



## OffHilbert Functional


function HornerFunctional(y0,sp)
    r=Array(typeof(y0),200)
    r[1]=y0
    k=1
    tol=eps()
    while(abs(r[k])>tol)
        k+=1
        if k>length(r)
            resize!(r,2length(r))
        end
        r[k]=r[k-1]*y0
    end

    CompactFunctional(r[1:k],sp)
end

function OffHilbert(sp::JacobiWeight{Ultraspherical{1}},z::Number)
    if sp.α == sp.β == 0.5
        π*HornerFunctional(intervaloffcircle(true,tocanonical(sp,z)),sp)
    else
        error("Not implemented")
    end
end

function OffHilbert(sp::JacobiWeight{Chebyshev},z::Number)
    if sp.α == sp.β == 0.5
        us=JacobiWeight(0.5,0.5,Ultraspherical{1}(domain(sp)))
        OffHilbert(us,z)*Conversion(sp,us)
    else
        error("Not implemented")
    end
end


