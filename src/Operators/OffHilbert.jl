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
        immutable $Op{D<:Space,R<:Space,T} <: Operator{T}
            data::BandedMatrix{T}
            domainspace::D
            rangespace::R
            order::Int
        end

        getindex(C::$Op,k::Integer,j::Integer) =
            k ≤ size(C.data,1) && j ≤ size(C.data,2) ? C.data[k,j] : zero(eltype(C))

        Base.convert{BT<:Operator}(::Type{BT},OH::$Op) =
            $Op{typeof(OH.domainspace),
                typeof(OH.rangespace),
                eltype(BT)}(OH.data,OH.domainspace,OH.rangespace,OH.order)

        $Op(ds::Space,rs::Space) = $Op(ds,rs,1)
        $Op(data::BandedMatrix,ds::Space,rs::Space) = $Op(data,ds,rs,1)

        $Op(ds::PeriodicDomain,rs::PeriodicDomain,order) = $Op(Laurent(ds),Laurent(rs),order)
        $Op(ds::PeriodicDomain,rs::PeriodicDomain) = $Op(Laurent(ds),Laurent(rs))

        domainspace(C::$Op) = C.domainspace
        rangespace(C::$Op) = C.rangespace
        bandinds(C::$Op) = bandinds(C.data)
        # divide size by blocklengths
        blockbandinds(C::$Op) = (-(size(C.data,1)-1)÷maximum(blocklengths(rangespace(C))),
                                 (size(C.data,2)-1)÷maximum(blocklengths(domainspace(C))))
    end
end

function OffHilbert(ds::Space,rs::Space,order::Int)
    @assert order==1
    tol=1E-13

    vv=Array(Vector{Complex128},0)
    m=100
    for k=1:2:1000
        b=Fun([zeros(k-1);1.],ds)
        v1=Fun(x->-stieltjes(b,x)/π,rs,m)
        b=Fun([zeros(k);1.],ds)
        v2=Fun(x->-stieltjes(b,x)/π,rs,m)
        if abs(v1.coefficients[end-1])>100tol || abs(v1.coefficients[end])>100tol ||
            abs(v2.coefficients[end-1])>100tol || abs(v2.coefficients[end])>100tol
            warn("OffHilbert not resolved with $m rows")
        end
        if norm(v1.coefficients,Inf)<tol &&
            norm(v2.coefficients,Inf)<tol
            C=zeros(Complex128,mapreduce(length,max,vv),length(vv))
            for j=1:length(vv)
                @inbounds C[1:length(vv[j]),j]=vv[j]
            end
            return OffHilbert(convert(BandedMatrix,C),ds,rs,order)
        end

        push!(vv,v1.coefficients)
        push!(vv,v2.coefficients)
    end

    warn("Max Iteration Reached for OffHilbert from "*string(ds)*" to "*string(rs))
    OffHilbert(convert(BandedMatrix,C),ds,rs,order)
end

## JacobiWeight

for (Op,Len) in ((:OffHilbert,:complexlength),(:OffSingularIntegral,:arclength))
    @eval begin
        function $Op{DD<:Interval}(ds::JacobiWeight{Ultraspherical{Int,DD},DD},rs::Space,ord::Int)
            @assert order(ds.space) == 1
            @assert ds.α==ds.β==0.5
            d = domain(ds)
            C = (.5*$Len(d))^(1-ord) # probably this is right for all ords ≥ 2. Certainly so for 0,1.

            if ord == 0
                z=Fun(identity,rs)
                x=mobius(ds,z)
                y=joukowskyinverse(Val{true},x)
                yk,ykp1=y,y*y
                ret=Array(typeof(y),300)
                ret[1]=-.5logabs(2y)+.25real(ykp1)
                n,l,u = 1,ncoefficients(ret[1])-1,0
                while norm(ret[n].coefficients)>100eps()
                    n+=1
                    if n > length(ret) resize!(ret,2length(ret)) end  # double preallocated ret
                    ykp1*=y
                    ret[n]=chop!(.5*(real(ykp1)/(n+1)-real(yk)/(n-1)) ,100eps())  #will be length 2n-1
                    yk*=y
                    u+=1   # upper bandwidth
                    l=max(l,ncoefficients(ret[n])-n)
                end
            elseif ord == 1
                y=Fun(z->joukowskyinverse(Val{true},mobius(ds,z)),rs)
                ret=Array(typeof(y),300)
                ret[1]=-y
                n,l,u = 1,ncoefficients(ret[1])-1,0
                while norm(ret[n].coefficients)>100eps()
                    n+=1
                    if n > length(ret) resize!(ret,2length(ret)) end  # double preallocated ret
                    ret[n]=chop!(y*ret[n-1],100eps())  #will be length 2n-1
                    u+=1   # upper bandwidth
                    l=max(l,ncoefficients(ret[n])-n)
                end
            else
                error("Not implemented for order=$ord")
            end

            M=bzeros(promote_type(typeof(C),eltype(y)),l+1,n,l,u)
            for k=1:n,j=1:ncoefficients(ret[k])
                M[j,k]=C*ret[k].coefficients[j]
            end
            $Op(M,ds,rs,ord)
        end

        function $Op{DD<:Interval}(ds::JacobiWeight{ChebyshevDirichlet{1,1,DD},DD},rs::PolynomialSpace,ord::Int)
            @assert ds.α==ds.β==-0.5
            d = domain(ds)
            C = (.5*$Len(d))^(1-ord) # probably this is right for all orders ≥ 2. Certainly so for 0,1.

            if ord == 0
                z=Fun(identity,rs)
                x=mobius(ds,z)
                y=joukowskyinverse(Val{true},x)
                yk,ykp1=y,y*y
                ret=Array(typeof(y),300)
                ret[1]=-logabs(2y/C)
                ret[2]=-real(yk)
                ret[3]=chop!(-ret[1]-.5real(ykp1),100eps())
                n,l,u = 3,max(ncoefficients(ret[1])-1,ncoefficients(ret[2])-2,ncoefficients(ret[3])-3),2
                while norm(ret[n].coefficients)>100eps()
                    n+=1
                    if n > length(ret) resize!(ret,2length(ret)) end  # double preallocated ret
                    ykp1*=y
                    ret[n]=chop!(real(yk)/(n-3)-real(ykp1)/(n-1),100eps())  #will be length 2n-1
                    yk*=y
                    u+=1   # upper bandwidth
                    l=max(l,ncoefficients(ret[n])-n)
                end
            elseif ord == 1
                z=Fun(identity,rs)
                x=mobius(ds,z)
                y=joukowskyinverse(Val{true},x)
                ret=Array(typeof(y),300)
                ret[1]=-1/sqrtx2(x)
                ret[2]=x*ret[1]+1
                ret[3]=2y
                n,l,u = 3,max(ncoefficients(ret[1])-1,ncoefficients(ret[2])-2,ncoefficients(ret[3])-3),2
                while norm(ret[n].coefficients)>100eps()
                    n+=1
                    if n > length(ret) resize!(ret,2length(ret)) end  # double preallocated ret
                    ret[n]=chop!(y*ret[n-1],100eps())  #will be length 2n-1
                    u+=1   # upper bandwidth
                    l=max(l,ncoefficients(ret[n])-n)
                end
            end

            M=bzeros(promote_type(typeof(C),eltype(y)),l+3,n,l,u)
            for k=1:n,j=1:ncoefficients(ret[k])
                M[j,k]=C*ret[k].coefficients[j]
            end
            $Op(M,ds,rs,ord)
        end

    end
end

function OffHilbert{D1<:Circle,D2<:Circle}(DS::Laurent{D1},RS::Laurent{D2},ord::Int)
    ds=domain(DS);rs=domain(RS)
    @assert ord==1


    # Correct for orrientation
    if !ds.orientation
        rDS=reverseorientation(DS)
        return (-OffHilbert(rDS,RS))*Conversion(DS,rDS)
    end

    if !rs.orientation
        rRS=reverseorientation(RS)
        return Conversion(rRS,RS)*(-OffHilbert(DS,rRS))
    end




    c2=rs.center;c1=ds.center
    r2=rs.radius;r1=ds.radius

    # make positive orientation
    if r1>r2&&abs(c1-c2)<r1  # we are inside the circle, use Taylor series
        M=interior_cauchy(Circle(c1,r1),Circle(c2,r2))
    elseif r1<r2&&abs(c1-c2)<r2 # we surround the domain, use Hardy{False} series
        M=exterior_cauchy(Circle(c1,r1),Circle(c2,r2))
    else
        M=disjoint_cauchy(Circle(c1,r1),Circle(c2,r2))
    end

    OffHilbert(2im*M,DS,RS)
end

function OffHilbert{D1<:Circle,D2<:Circle}(DS::Fourier{D1},RS::Fourier{D2},ord::Int)
    LD=Laurent(domain(DS))
    LR=Laurent(domain(RS))
    Conversion(LR,RS)*OffHilbert(LD,LR,ord)*Conversion(DS,LD)
end




## Special cases

function exterior_cauchy(b::Circle,a::Circle)
    c=b.center
    r=b.radius

    S=Fun([0.0,0,1],a)  # Shift to use bandedness
    ret=Array(Fun{Laurent{typeof(a)},Complex{Float64}},300)
    ret[1]=Fun(z->(r/(z-c)),a)
    n=1
    m=ncoefficients(ret[1])-2
    f1=ret[1]*S
    while norm(ret[n].coefficients)>100eps()
        n+=1
        if n > length(ret)
            resize!(ret,2length(ret))
        end
        ret[n]=chop!(f1*ret[n-1],100eps())
        m=max(m,ncoefficients(ret[n])-2)
    end

    M=bzeros(Complex{Float64},2n,2n,m,0)
    #j+2k-2≤2n
    #j≤2(n-k)+2
    for k=1:n,j=2:2:min(length(ret[k].coefficients),2(n-k)+2)
        M[j+2k-2,2k]=-ret[k].coefficients[j]
    end
    M
end

function interior_cauchy(a::Circle,b::Circle)
    z=mappoint(a,Circle(),Fun(b))

    ret=Array(Fun{Laurent{typeof(b)},Complex{Float64}},300)
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

    M=bzeros(Complex{Float64},2n-1,2n-1,0,m)
    for k=1:n,j=max(1,2k-1-m):2:2k-1
        M[j,2k-1]=ret[k].coefficients[j]
    end

    M
end

function disjoint_cauchy(a::Circle,b::Circle)
    c=a.center
    r=a.radius

    f=Fun(z->r/(z-c),b)

    ret=Array(Fun{Laurent{typeof(b)},Complex{Float64}},300)
    ret[1]=f
    n=1

    l=ncoefficients(f)-2   #lower bandwidth
    u=1             #upper bandwidth

    while norm(ret[n].coefficients)>100eps()
        n+=1
        if n > length(ret)
            # double preallocated ret
            resize!(ret,2length(ret))
        end
    ret[n]=chop!(f*ret[n-1],100eps())  #will be length 2n-1
    u=max(u,ncoefficients(ret[n])-2n)   # upper bandwidth

        # find bandwidth by checking how many coefficients are zero
        # we jump over negative coefficients
        for j=1:2:ncoefficients(ret[n])
            if norm(ret[n].coefficients[j])>100eps()
                l=max(l,2n-j)
                break
            end
        end
    end

    M=bzeros(Complex{Float64},2n-1,2n,l,u)
    for k=1:n,j=max(1,2k-u):2:min(ncoefficients(ret[k]),2n-1)
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
Stieltjes(d,r,ord) = (ord==0?π:-π)*OffHilbert(d,r,ord)
Stieltjes(d,r) = (-π)*OffHilbert(d,r)



Cauchy(s::Bool,d)=(s?0.5:-0.5)*I +(-0.5im)*Hilbert(d)
Cauchy(s::Int,d)=Cauchy(s==1,d)
Cauchy(s::Union{Int,Bool})=Cauchy(s,UnsetSpace())
Cauchy(ds,rs,ord)=(1/(2*im))*OffHilbert(ds,rs,ord)
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


## OffHilbert Functional

HornerFunctional(y0,sp) =
    FiniteOperator(hornervector(y0).',sp,ConstantSpace())


function OffHilbert{DD}(sp::JacobiWeight{Ultraspherical{Int,DD},DD},z::Number)
    @assert order(sp.space) == 1
    if sp.α == sp.β == 0.5
        # this translates the following cauchy to a functional
        #    0.5im*hornersum(cfs,joukowskyinverse(Val{true},mobius(u,z)))
        # which consists of multiplying by 2*im
        -HornerFunctional(joukowskyinverse(Val{true},mobius(sp,z)),sp)
    else
        # calculate directly
        r=Array(eltype(z),0)
        for k=1:10000
            push!(r,-stieltjes(Fun([zeros(k-1);1.],sp),z)/π)
            if abs(last(r)) < eps()
                break
            end
        end
        FiniteOperator(r.',sp,ConstantSpace())
    end
end

function OffHilbert{DD}(sp::JacobiWeight{Chebyshev{DD},DD},z::Number)
    #try converting to Ultraspherical(1)
    us=JacobiWeight(sp.α,sp.β,Ultraspherical(1,domain(sp)))
    OffHilbert(us,z)*Conversion(sp,us)
end


function OffHilbert{DD}(sp::JacobiWeight{ChebyshevDirichlet{1,1,DD},DD},z::Number)
    if sp.α == sp.β == -0.5
        z=mobius(sp,z)

        sx2z=sqrtx2(z)
        sx2zi=1./sx2z

        FiniteOperator([-sx2zi;1-sx2zi;2*hornervector(z-sx2z)].',sp,ConstantSpace())
    else
        # try converting to Canonical
        us=JacobiWeight(sp.α,sp.β,Chebyshev(domain(sp)))
        OffHilbert(us,z)*Conversion(sp,us)
    end
end
