# Fractal sets of Intervals.
# TODO: fat Cantor sets, Smith-Volterra-Cantor sets, asymmetric Cantor sets...

export cantor

# α is width, n is number of levels

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
