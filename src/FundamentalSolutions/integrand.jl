#
# Evaluates s-plane integrand for u (and possibly ux,uy) given params a,b.
#
function integrand{T<:Union{Float64,Complex{Float64}}}(s::T, a::Float64, b::Float64, x::Float64, y::Float64, derivs::Bool)
    # Precalculations
    es = exp(s)
    ies = 1.0/es
    u = cis(a * ies + b * es - es * es * es / 12.0)

    if derivs
        # calculate prefactors for exponentials in order to do derivatives
        c,d = 0.5im*ies,0.5im*es
        # x deriv computation
        ux = (-c*x)*u
        # y deriv computation
        uy = (-c*y+d)*u
    else
        # make sure that ux, uy aren't going to give us anything funny (or >emach)
        ux = ZIM
        uy = ZIM
    end
    u,ux,uy
end

function integrand!{T<:Union{Float64,Complex{Float64}}}(u::Vector{Complex{Float64}}, ux::Vector{Complex{Float64}}, uy::Vector{Complex{Float64}}, s::Vector{T}, a::Float64, b::Float64, x::Float64, y::Float64, derivs::Bool, nquad::Int)
    # Precalculations
    es = exp(s[1:nquad])
    ies = 1.0./es
    u[1:nquad] = cis(a * ies + b * es - es .* es .* es / 12.0)

    if derivs
        # calculate prefactors for exponentials in order to do derivatives
        c,d = 0.5im*ies,0.5im*es
        # x deriv computation
        ux[1:nquad] = (-c*x).*u[1:nquad]
        # y deriv computation
        uy[1:nquad] = (-c*y+d).*u[1:nquad]
    else
        # make sure that ux, uy aren't going to give us anything funny (or >emach)
        ux[1:nquad] = ZIM
        uy[1:nquad] = ZIM
    end
end
