# Fractals
# TODO: fat Cantor sets, Smith-Volterra-Cantor sets, asymmetric Cantor sets...

export cantor

# α is width, n is number of levels

# Standard Cantor set removes the middle third at every level
cantor{T}(d::Domain{T},n::Int) = cantor(d,n,3)

function cantor{T}(d::Interval{T},n::Int,α::Number)
    a,b = d.a,d.b
    if n == 0
        return d
    else
        d = Interval{T}(zero(T),one(T))
        C = d/α ∪ (α-1+d)/α
        for k=2:n
            C = C/α ∪ (α-1+C)/α
        end
        return a+(b-a)*C
    end
end

function cantor{T,V}(d::Circle{T,V},n::Int,α::Number)
    c,r = d.center,d.radius
    α = convert(promote_type(real(T),V,typeof(α)),α)
    if n == 0
        return d
    else
        C = cantor(Interval{T}(),n,α)
        twon = length(C)
        return ∪(map(d->Arc(c,r,((getfield(d,:a)+1/2α)π,(getfield(d,:b)+1/2α)π)),C[1:div(twon,2)])) ∪ ∪(map(d->Arc(c,r,((getfield(d,:a)-1/2α)π,(getfield(d,:b)-1/2α)π)),C[div(twon,2)+1:twon]))
    end
end
