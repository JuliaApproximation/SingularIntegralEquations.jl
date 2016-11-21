#
# Evaluates s-plane integrand for u (and possibly ux,uy) given params a,b.
#

function integrand!(u::Vector{Complex{Float64}}, ux::Vector{Complex{Float64}}, uy::Vector{Complex{Float64}}, s::Vector{Complex{Float64}}, a::Float64, b::Float64, x::Float64, y::Float64, derivs::Bool, n::Int)
    @simd for i=1:n
        # Precalculations
        @inbounds es = exp(s[i])
        ies = inv(es)
        @inbounds u[i] = cis(a * ies + b * es - es * es * es / 12.0)

        if derivs
            # calculate prefactors for exponentials in order to do derivatives
            c,d = 0.5im*ies,0.5im*es
            # x deriv computation
            @inbounds ux[i] = (-c*x)*u[i]
            # y deriv computation
            @inbounds uy[i] = (-c*y+d)*u[i]
        else
            # make sure that ux, uy aren't going to give us anything funny (or >emach)
            @inbounds ux[i] = ZIM
            @inbounds uy[i] = ZIM
        end
    end
end

# assume if length n not given that vectors are of length 1.

integrand!(u::Vector{Complex{Float64}}, ux::Vector{Complex{Float64}}, uy::Vector{Complex{Float64}}, s::Vector{Complex{Float64}}, a::Float64, b::Float64, x::Float64, y::Float64, derivs::Bool) = integrand!(u, ux, uy, s, a, b, x, y, derivs, 1)
