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
        struct $Op{D<:Space,R<:Space,T} <: Operator{T}
            data::BandedMatrix{T,Matrix{T}}
            domainspace::D
            rangespace::R
            order::Int
        end

        getindex(C::$Op,k::Integer,j::Integer) =
            k ≤ size(C.data,1) && j ≤ size(C.data,2) ? C.data[k,j] : zero(eltype(C))

        convert(::Type{Operator{T}},OH::$Op) where {T} =
            $Op{typeof(OH.domainspace),
                typeof(OH.rangespace),
                T}(OH.data,OH.domainspace,OH.rangespace,OH.order)

        $Op(ds::Space,rs::Space) = $Op(ds,rs,1)
        $Op(data::BandedMatrix,ds::Space,rs::Space) = $Op(data,ds,rs,1)

        $Op(ds::PeriodicDomain,rs::PeriodicDomain,order) = $Op(Laurent(ds),Laurent(rs),order)
        $Op(ds::PeriodicDomain,rs::PeriodicDomain) = $Op(Laurent(ds),Laurent(rs))

        $Op(ds::Space,rs::Number,order) = $Op(ds,Space(rs),order)
        $Op(ds::Space,rs::Number) = $Op(ds,ConstantSpace(Point(rs)))

        domainspace(C::$Op) = C.domainspace
        rangespace(C::$Op) = C.rangespace
        bandwidths(C::$Op) = bandwidths(C.data)
        # divide size by blocklengths
        blockbandwidths(C::$Op) = ((size(C.data,1)-1)÷maximum(blocklengths(rangespace(C))),
                                 (size(C.data,2)-1)÷maximum(blocklengths(domainspace(C))))
    end
end

function default_OffHilbert(ds::Space,rs::Space,order::Int)
    @assert order==1
    tol=1E-13

    vv=Vector{Vector{ComplexF64}}()
    m=min(100,dimension(rs))
    for k=1:2:1000
        b=Fun(ds,[zeros(k-1);1.])
        v1=Fun(x->-stieltjes(b,x)/π,rs,m)
        b=Fun(ds,[zeros(k);1.])
        v2=Fun(x->-stieltjes(b,x)/π,rs,m)
        if m ≥ 2 && (abs(v1.coefficients[end-1])>100tol || abs(v1.coefficients[end])>100tol ||
                        abs(v2.coefficients[end-1])>100tol || abs(v2.coefficients[end])>100tol)
            warn("OffHilbert not resolved with $m rows")
        end
        if norm(v1.coefficients,Inf)<tol &&
            norm(v2.coefficients,Inf)<tol
            C=zeros(ComplexF64,mapreduce(length,max,vv),length(vv))
            for j=1:length(vv)
                @inbounds C[1:length(vv[j]),j] = vv[j]
            end
            return OffHilbert(convert(BandedMatrix,C),ds,rs,order)
        end

        push!(vv,v1.coefficients)
        push!(vv,v2.coefficients)
    end

    warn("Max Iteration Reached for OffHilbert from "*string(ds)*" to "*string(rs))
    OffHilbert(convert(BandedMatrix,C),ds,rs,order)
end


function default_OffSingularIntegral(::Type{T}, ds::Space, rs::Space, order::Int) where T
    tol=1E-13

    vv=Vector{Vector{T}}()
    m=100
    for k=1:2:1000
        b=Fun(ds,[zeros(k-1);1.])
        v1=Fun(x->singularintegral(order,b,x),rs,m)
        b=Fun(ds,[zeros(k);1.])
        v2=Fun(x->singularintegral(order,b,x),rs,m)
        if isinf(dimension(rs)) && (abs(v1.coefficients[end-1])>100tol || abs(v1.coefficients[end])>100tol ||
            abs(v2.coefficients[end-1])>100tol || abs(v2.coefficients[end])>100tol)
            warn("OffSingularIntegral not resolved with $m rows")
        end
        if norm(v1.coefficients,Inf)<tol &&
            norm(v2.coefficients,Inf)<tol
            C=zeros(T,mapreduce(length,max,vv),length(vv))
            for j=1:length(vv)
                @inbounds C[1:length(vv[j]),j]=vv[j]
            end
            return OffSingularIntegral(convert(BandedMatrix,C),ds,rs,order)
        end

        push!(vv,v1.coefficients)
        push!(vv,v2.coefficients)
    end

    warn("Max Iteration Reached for OffHilbert from "*string(ds)*" to "*string(rs))
    C=zeros(T,mapreduce(length,max,vv),length(vv))
    for j=1:length(vv)
        @inbounds C[1:length(vv[j]),j]=vv[j]
    end
    OffSingularIntegral(convert(BandedMatrix,C),ds,rs,order)
end

# TODO: SingularInteral( , 0 ) is really different
default_OffSingularIntegral(ds::Space, rs::Space, order::Int) =
    default_OffSingularIntegral(order == 0 ? Float64 : ComplexF64, ds, rs, order)


OffHilbert(ds::Space,rs::Space,order::Int) = default_OffHilbert(ds,rs,order)
OffSingularIntegral(ds::Space,rs::Space,order::Int) = default_OffSingularIntegral(ds,rs,order)

## JacobiWeight

for (Op,Len) in ((:OffHilbert,:complexlength),(:OffSingularIntegral,:arclength))
    @eval begin
        function $Op(ds::JacobiWeight{Ultraspherical{Int,DD,RR},DD},rs::Space,ord::Int) where {DD<:IntervalOrSegment,RR}
            @assert order(ds.space) == 1
            @assert ds.α==ds.β==0.5
            d = domain(ds)
            C = (.5*$Len(d))^(1-ord) # probably this is right for all ords ≥ 2. Certainly so for 0,1.

            if ord == 0
                z=Fun(identity,rs)
                x=mobius(ds,z)
                y=joukowskyinverse(Val{true},x)
                yk,ykp1=y,y*y
                ret=Array{typeof(y)}(undef,300)
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
                ret=Array{typeof(y)}(undef,300)
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

            M=BandedMatrix{promote_type(typeof(C),cfstype(y))}(Zeros(l+1,n), (l,u))
            for k=1:n,j=1:min(l+1,ncoefficients(ret[k]))
                M[j,k]=C*ret[k].coefficients[j]
            end
            $Op(M,ds,rs,ord)
        end

        function $Op(ds::JacobiWeight{ChebyshevDirichlet{1,1,DD,RR},DD},rs::PolynomialSpace,ord::Int) where {DD<:IntervalOrSegment,RR}
            @assert ds.α==ds.β==-0.5
            d = domain(ds)
            C = (.5*$Len(d))^(1-ord) # probably this is right for all orders ≥ 2. Certainly so for 0,1.

            if ord == 0
                z=Fun(identity,rs)
                x=mobius(ds,z)
                y=joukowskyinverse(Val{true},x)
                yk,ykp1=y,y*y
                ret=Array{typeof(y)}(undef,300)
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
                ret=Array{typeof(y)}(undef,300)
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

            M=BandedMatrix{promote_type(typeof(C),cfstype(y))}(Zeros(l+3,n),(l,u))
            for k=1:n,j=1:min(l+3,ncoefficients(ret[k]))
                M[j,k]=C*ret[k].coefficients[j]
            end
            $Op(M,ds,rs,ord)
        end

    end
end

function OffHilbert(DS::Laurent{D1,R1},RS::Laurent{D2,R2},ord::Int) where {D1<:Circle,D2<:Circle,R1,R2}
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

function OffHilbert(DS::Fourier{D1,R1},RS::Fourier{D2,R2},ord::Int) where {D1<:Circle,D2<:Circle,R1,R2}
    LD=Laurent(domain(DS))
    LR=Laurent(domain(RS))
    Conversion(LR,RS)*OffHilbert(LD,LR,ord)*Conversion(DS,LD)
end




## Special cases

function exterior_cauchy(b::Circle,a::Circle)
    c=b.center
    r=b.radius

    S=Fun(a,[0.0,0,1])  # Shift to use bandedness
    ret=Array{Fun{Laurent{typeof(a),ComplexF64},ComplexF64}}(undef,300)
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

    M=BandedMatrix{Complex{Float64}}(Zeros(2n,2n),(m,0))
    #j+2k-2≤2n
    #j≤2(n-k)+2
    for k=1:n,j=2:2:min(length(ret[k].coefficients),2(n-k)+2)
        M[j+2k-2,2k]=-ret[k].coefficients[j]
    end
    M
end

function interior_cauchy(a::Circle,b::Circle)
    z=mappoint(a,Circle(),Fun(b))

    ret=Array{Fun{Laurent{typeof(b),ComplexF64},ComplexF64}}(undef,300)
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

    M=BandedMatrix{Complex{Float64}}(Zeros(2n-1,2n-1),(0,m))
    for k=1:n,j=max(1,2k-1-m):2:2k-1
        M[j,2k-1]=ret[k].coefficients[j]
    end

    M
end

function disjoint_cauchy(a::Circle, b::Circle)
    c=a.center
    r=a.radius

    f=Fun(z->r/(z-c),b)

    ret=Array{Fun{Laurent{typeof(b),ComplexF64},ComplexF64}}(undef,300)
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

    M=BandedMatrix{Complex{Float64}}(Zeros(2n-1,2n),(l,u))
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
Stieltjes(d,r,ord) = (ord==0 ? π : -π)*OffHilbert(d,r,ord)
Stieltjes(d,r) = (-π)*OffHilbert(d,r)



Cauchy(s::Bool,d)=(s ? 0.5 : -0.5)*I + (-0.5im)*Hilbert(d)
Cauchy(s::Int,d)=Cauchy(s==1,d)
Cauchy(s::Union{Int,Bool})=Cauchy(s,UnsetSpace())
Cauchy(ds,rs,ord)=(1/(2*im))*OffHilbert(ds,rs,ord)
Cauchy(ds,rs)=Cauchy(ds,rs,1)






## Stiejles Functional


function hornervector(y0)
    r=Array{typeof(y0)}(undef,200)
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

function HornerFunctional(y0,sp,cs)
    v = hornervector(y0)
    FiniteOperator(transpose(v),sp,cs)
end


function OffHilbert(sp::JacobiWeight{Ultraspherical{Int,DD,RR},DD},cs::ConstantSpace{<:Point},k::Int) where {DD<:IntervalOrSegment,RR}
    z = convert(Number, domain(cs))
    @assert k == 1
    @assert order(sp.space) == 1
    if sp.α == sp.β == 0.5
        # this translates the following cauchy to a functional
        #    0.5im*hornersum(cfs,joukowskyinverse(Val{true},mobius(u,z)))
        # which consists of multiplying by 2*im
        -HornerFunctional(joukowskyinverse(Val{true},mobius(sp,z)),sp,cs)
    else
        # calculate directly
        r=Vector{eltype(z)}()
        for k=1:10000
            push!(r,-stieltjes(Fun(sp,[zeros(k-1);1.]),z)/π)
            if abs(last(r)) < eps()
                break
            end
        end
        FiniteOperator(transpose(r),sp,cs)
    end
end

for Op in (:OffHilbert, :OffSingularIntegral)
    defOp = Meta.parse("default_"*string(Op))
    @eval begin
        function $Op(sp::JacobiWeight{Chebyshev{DD,RR},DD},z::Space,k::Int) where {DD,RR}
            if sp.β == sp.α == 0.5
                #try converting to Ultraspherical(1)
                us=JacobiWeight(sp.β,sp.α,Ultraspherical(1,domain(sp)))
                $Op(us,z,k)*Conversion(sp,us)
            else
                $defOp(sp,z,k)
            end
        end
    end
end


function OffHilbert(sp::JacobiWeight{ChebyshevDirichlet{1,1,DD,RR},DD},cs::ConstantSpace{<:Point},k::Int) where {DD<:IntervalOrSegment,RR}
    z = Number(domain(cs))
    @assert k == 1
    if sp.α == sp.β == -0.5
        z=mobius(sp,z)

        sx2z=sqrtx2(z)
        sx2zi=1/sx2z

        FiniteOperator(transpose([-sx2zi;1-sx2zi;2*hornervector(z-sx2z)]),sp,cs)
    else
        # try converting to Canonical
        us=JacobiWeight(sp.β,sp.α,Chebyshev(domain(sp)))
        OffHilbert(us,cs)*Conversion(sp,us)
    end
end
