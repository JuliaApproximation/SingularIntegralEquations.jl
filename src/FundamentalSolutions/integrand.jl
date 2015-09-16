#
# Evaluates s-plane integrand for u (and possibly ux,uy) given params a,b.
#
function integrand!(u::Vector{Complex{Float64}}, ux::Vector{Complex{Float64}}, uy::Vector{Complex{Float64}}, s::Vector{Complex{Float64}}, a::Float64, b::Float64, x::Float64, y::Float64, derivs::Bool)
    # Precalculations
    es = exp(s)
    ies = 1.0./es
    u[:] = cis(a * ies + b * es - es .* es .* es / 12.0)

    if derivs
        # calculate prefactors for exponentials in order to do derivatives
        c,d = complex(0.0,0.5ies),complex(0.0,0.5es)
        # x deriv computation
        ux[:] = (-c*x).*u
        # y deriv computation
        uy[:] = (-c*y+d).*u
    else
        # make sure that ux, uy aren't going to give us anything funny (or >emach)
        ux[:] = 0.0
        uy[:] = 0.0
    end
end
