export OffHilbert,OffSingularIntegral,Stieltjes,Cauchy


#############
# OffHilbert implements:
#
#       OH f(z) := 1/π\int_Γ f(t)/(t-z) dt,  z ∉ Γ
#
# OffSingularIntegral implements:
#
#       OSI f(z) := 1/π\int_Γ f(t)/(t-z) ds(t),  z ∉ Γ
#
#############

for Op in (:OffHilbert,:OffSingularIntegral)
    @eval begin
        immutable $Op{D<:FunctionSpace,R<:FunctionSpace,T} <: BandedOperator{T}
            data::BandedMatrix{T}
            domainspace::D
            rangespace::R
            order::Int
        end

        addentries!(C::$Op,A,kr)=addentries!(C.data,A,kr)

        Base.convert{BT<:Operator}(::Type{BT},OH::$Op)=$Op{typeof(OH.domainspace),
                                                           typeof(OH.rangespace),
                                                           eltype(BT)}(OH.data,OH.domainspace,OH.rangespace,OH.order)

        $Op(ds::FunctionSpace,rs::FunctionSpace) = $Op(ds,rs,1)
        $Op(data::BandedMatrix,ds::FunctionSpace,rs::FunctionSpace) = $Op(data,ds,rs,1)

        $Op(ds::PeriodicDomain,rs::PeriodicDomain,order)=$Op(Laurent(ds),Laurent(rs),order)
        $Op(ds::PeriodicDomain,rs::PeriodicDomain)=$Op(Laurent(ds),Laurent(rs))

        domainspace(C::$Op)=C.domainspace
        rangespace(C::$Op)=C.rangespace
        bandinds(C::$Op)=bandinds(C.data)
    end
end

function OffHilbert(ds::FunctionSpace,rs::FunctionSpace,order::Int)
    @assert order==1
    tol=1E-13

    b=Fun([1.],ds);
    v1=chop(Fun(x->-stieltjes(b,x)/π,rs).coefficients,tol)
    b=Fun([0.,1.],ds);
    v2=chop(Fun(x->-stieltjes(b,x)/π,rs).coefficients,tol)
    m=max(length(v1),length(v2))
    C=Array(Complex128,m,1000)

    C[:,1]=pad!(v1,m)
    C[:,2]=pad!(v2,m)

    for k=3:1000
        b=Fun([zeros(k-1);1.],ds)
        cfs=Fun(x->-stieltjes(b,x)/π,rs,m).coefficients
        C[:,k]=cfs
        if norm(cfs)<tol
            return OffHilbert(convert(BandedMatrix,C[:,1:k]),ds,rs,order)
        end
    end
    warn("Max Iteration Reached for OffHilbert from "*string(ds)*" to "*string(rs))
    OffHilbert(convert(BandedMatrix,C),ds,rs,order)
end

## JacobiWeight

for (Op,Len) in ((:OffHilbert,:complexlength),(:OffSingularIntegral,:length))
    @eval begin
        function $Op(ds::JacobiWeight{Ultraspherical{1}},rs::FunctionSpace,order::Int)
            @assert ds.α==ds.β==0.5
            d = domain(ds)
            C = (.5*$Len(d))^(1-order) # probably this is right for all orders ≥ 2. Certainly so for 0,1.

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
                    ret[n]=chop!(.5*(real(ykp1)/(n+1)-real(yk)/(n-1)) ,100eps())  #will be length 2n-1
                    yk*=y
                    u+=1   # upper bandwidth
                    l=max(l,length(ret[n])-n)
                end
            elseif order == 1
                y=Fun(z->intervaloffcircle(true,tocanonical(ds,z)),rs)
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

            M=bazeros(promote_type(typeof(C),eltype(y)),l+1,n,l,u)
            for k=1:n,j=1:length(ret[k])
                M[j,k]=C*ret[k].coefficients[j]
            end
            $Op(M,ds,rs,order)
        end

        function $Op(ds::JacobiWeight{ChebyshevDirichlet{1,1}},rs::PolynomialSpace,order::Int)
            @assert ds.α==ds.β==-0.5
            d = domain(ds)
            C = (.5*$Len(d))^(1-order) # probably this is right for all orders ≥ 2. Certainly so for 0,1.

            if order == 0
                z=Fun(identity,rs)
                x=tocanonical(ds,z)
                y=intervaloffcircle(true,x)
                yk,ykp1=y,y*y
                ret=Array(typeof(y),300)
                ret[1]=-logabs(2y/C)
                ret[2]=-real(yk)
                ret[3]=chop!(-ret[1]-.5real(ykp1),100eps())
                n,l,u = 3,max(length(ret[1])-1,length(ret[2])-2,length(ret[3])-3),2
                while norm(ret[n].coefficients)>100eps()
                    n+=1
                    if n > length(ret) resize!(ret,2length(ret)) end  # double preallocated ret
                    ykp1*=y
                    ret[n]=chop!(real(yk)/(n-3)-real(ykp1)/(n-1),100eps())  #will be length 2n-1
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

            M=bazeros(promote_type(typeof(C),eltype(y)),l+3,n,l,u)
            for k=1:n,j=1:length(ret[k])
                M[j,k]=C*ret[k].coefficients[j]
            end
            $Op(M,ds,rs,order)
        end

    end
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
            resize!(ret,2length(ret))
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
            resize!(ret,2length(ret))
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
            resize!(ret,2length(ret))
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






#############
# Cauchy implements the Cauchy operator corresponding to evaluating the Cauchy transform
#
#       C f(z) := 1/(2πi)\int_Γ f(t)/(t-z) dt
#
# It is given in terms of the Stieltjes operator
#
#       S f(z) := \int_Γ f(t)/(z-t) dt = -2πi*C f(z)
#
# note that the domain of domainspace must be different than the domain of rangespace
#
# The notion of C^± for the left/right limits of the Cauchy operator
# with the domains matching is represented
# using the Hilbert operator and the formulae
#
#    C^+  -  C^- = I
#    C^+  +  C^- = -im*H
#
#   Or for the Stieltjes operator
#
#    S^+ - S^- = -2πi*I
#    S^+ + S^- = -2π*H
#
############
Stieltjes(d,r,order) = (order==0?π:-π)*OffHilbert(d,r,order)
Stieltjes(d,r) = (-π)*OffHilbert(d,r)



Cauchy(s::Bool,d)=(s?0.5:-0.5)*I +(-0.5im)*Hilbert(d)
Cauchy(s::Int,d)=Cauchy(s==1,d)
Cauchy(s::Union(Int,Bool))=Cauchy(s,UnsetSpace())
Cauchy(ds,rs,order)=(1/(2*im))*OffHilbert(ds,rs,order)
Cauchy(ds,rs)=Cauchy(ds,rs,1)






## Stiejles Functional


function hornervector(y0)
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

    r[1:k]
end

HornerFunctional(y0,sp)=CompactFunctional(hornervector(y0),sp)

function OffHilbert(sp::JacobiWeight{Ultraspherical{1}},z::Number)
    if sp.α == sp.β == 0.5
        # this translates the following cauchy to a functional
        #    0.5im*hornersum(cfs,intervaloffcircle(true,tocanonical(u,z)))
        # which consists of multiplying by 2*im
        -HornerFunctional(intervaloffcircle(true,tocanonical(sp,z)),sp)
    else
        # calculate directly
        r=Array(eltype(z),0)
        for k=1:10000
            push!(r,-stieltjes(Fun([zeros(k-1);1.],sp),z)/π)
            if abs(last(r)) < eps()
                break
            end
        end
        CompactFunctional(r,sp)
    end
end

function OffHilbert(sp::JacobiWeight{Chebyshev},z::Number)
    #try converting to Ultraspherical{1}
    us=JacobiWeight(sp.α,sp.β,Ultraspherical{1}(domain(sp)))
    OffHilbert(us,z)*Conversion(sp,us)
end


function OffHilbert(sp::JacobiWeight{ChebyshevDirichlet{1,1}},z::Number)
    if sp.α == sp.β == -0.5
        z=tocanonical(sp,z)

        sx2z=sqrtx2(z)
        sx2zi=1./sx2z

        CompactFunctional([-sx2zi;1-sx2zi;2*hornervector(z-sx2z)],sp)
    else
        # try converting to Canonical
        us=JacobiWeight(sp.α,sp.β,Chebyshev(domain(sp)))
        OffHilbert(us,z)*Conversion(sp,us)
    end
end
