# Fractals
# TODO: super-fat Cantor sets, asymmetric Cantor sets...

export cantor, thincantor, thinnercantor, thinnestcantor, smithvolterracantor

# Standard Cantor set removes the middle third at every level

for set in (:cantor,:thincantor,:thinnercantor,:thinnestcantor)
    @eval begin
        $set(d::Domain,n::Int) = $set(d,n,3)

        function $set{T,V}(d::Circle{T,V},n::Int,α::Number)
            c,r = d.center,d.radius
            α = convert(promote_type(real(T),V,typeof(α)),α)
            if n == 0
                return d
            else
                C = $set(Segment{T}(),n,α)
                return UnionDomain(map(d->Arc(c,r,(d.a+1/2α)π,(d.b+1/2α)π),C[1:div(length(C),2)])) ∪ UnionDomain(map(d->Arc(c,r,(d.a-1/2α)π,(d.b-1/2α)π),C[div(length(C),2)+1:length(C)]))
            end
        end

    end
end

# α is width, n is number of levels

function cantor{T}(d::Segment{T},n::Int,α::Number)
    a,b = d.a,d.b
    if n == 0
        return d
    else
        C = Segment{T}(zero(T),one(T))
        for k=n:-1:1
            C = C/α ∪ (α-1+C)/α
        end
        return a+(b-a)*C
    end
end

# Thin Cantor set removes the middle n/(n+2)th at the nth level

function thincantor{T}(d::Segment{T},n::Int,α::Number)
    a,b = d.a,d.b
    if n == 0
        return d
    else
        C = Segment{T}(zero(T),one(T))
        for k=n:-1:1
            C = C/(α+k-1) ∪ (α+k-2+C)/(α+k-1)
        end
        return a+(b-a)*C
    end
end

function thinnercantor{T}(d::Segment{T},n::Int,α::Number)
    a,b = d.a,d.b
    if n == 0
        return d
    else
        C = Segment{T}(zero(T),one(T))
        for k=n:-1:1
            C = C/α^k ∪ (α^k-1+C)/α^k
        end
        return a+(b-a)*C
    end
end

function thinnestcantor{T}(d::Segment{T},n::Int,α::Number)
    a,b = d.a,d.b
    if n == 0
        return d
    else
        C = Segment{T}(zero(T),one(T))
        for k=n:-1:1
            C = C/α^(2^(k-1)) ∪ (α^(2^(k-1))-1+C)/α^(2^(k-1))
        end
        return a+(b-a)*C
    end
end

smithvolterracantor{T}(d::Domain{T},n::Int) = smithvolterracantor(d,n,one(T)/4,one(T)/4)

function smithvolterracantor{T}(d::Segment{T},n::Int,α::Number,β::Number)
    a,b = d.a,d.b
    if n == 0
        return d
    else
        C = Segment{T}(zero(T),one(T))
        ln = (one(T)-β*(one(T)-(2α)^n)/(one(T)-2α))/2^n
        lnm1 = (2ln+β*α^(n-1))
        for k=n:-1:1
            C = (C*ln ∪ (lnm1+(C-1)*ln))/lnm1
            ln,lnm1 = lnm1,(2lnm1+β*α^(k-2))
        end
        return a+(b-a)*C
    end
end
