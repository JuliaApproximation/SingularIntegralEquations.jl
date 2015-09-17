#=
 lhfs
 Description:
    Computes value and possibly source-derivatives of fundamental solution at
    a series of target points and energies, with the source at the origin.
 Parameters:
    x       (input) - array of x coordinates in the cartesian plane
    y       (input) - array of y coordinates in the cartesian plane
    energies(input) - parameter array to fundamental solution, one for each target pt
    derivs  (input) - 1: compute values and derivatives to fundamental solution.
                      0: only compute the fundamental solution values.
    n       (input) - size of x, y arrays
    u       (output) - array of size n of fundamental solution values.
            u[i] is the fundamental solution at the coordinate (x[i],y[i]).
    ux      (output) - array of the x derivative of each fundamental solution value
            (unused if derivs=0)
    uy      (output) - array of the y derivative of each fundamental solution value
            (unused if derivs=0)
    stdquad (input, int)     - global convergence params: # default quad pts per saddle pt
     h (input, double)        - PTR spacing relative to std Gaussian sigma
                           (if h<0, then -h is used and h is also used to
			   scale the parameters maxh and minsaddlen)
     meth (input) - int giving method for choosing # nodes (passed to quad_nodes)
     gamout (input) - 0: no file output (default), 1: diagnostic text to file nodes.dat
=#
function lhfs!(x::Vector{Float64}, y::Vector{Float64}, energies::Vector{Float64}, derivs::Bool, u::Vector{Complex{Float64}}, ux::Vector{Complex{Float64}}, uy::Vector{Complex{Float64}}, stdquad::Int, h::Float64, meth::Int)

    # Allocation
    gam = Vector{Complex{Float64}}(MAXNQUAD)
    gamp = Vector{Complex{Float64}}(MAXNQUAD)
    integ = Vector{Complex{Float64}}(MAXNQUAD)
    integx = Vector{Complex{Float64}}(MAXNQUAD)
    integy = Vector{Complex{Float64}}(MAXNQUAD)
    ts = Vector{Float64}(MAXNQUAD)
    ws = Vector{Float64}(MAXNQUAD)

    n = length(x)
    @assert n == length(y) == length(energies)

    for i=1:n
        a = 0.25(x[i]^2+y[i]^2)
        b = 0.5y[i]+energies[i]

        if a == 0
            u[i] = Inf
            if derivs
                ux[i] = Inf
                uy[i] = Inf
            end
        else
            nquad = quad_nodes!(ts, ws, a, b, x[i], y[i], derivs, stdquad, h, meth)

            contour!(gam, gamp, a, b, ts, nquad)

            integrand!(integ, integx, integy, gam, a, b, x[i], y[i], derivs, nquad)

            u[i] = quadsum(integ,gamp,ws,nquad)
            if derivs
                ux[i] = quadsum(integx,gamp,ws,nquad)
                uy[i] = quadsum(integy,gamp,ws,nquad)
            end
        end
    end
end

function quadsum{S,T,V}(integ::Vector{S}, gamp::Vector{T}, ws::Vector{V}, n::Int)
    ret = integ[1]*gamp[1]*ws[1]
    for k=2:n
        ret += integ[k]*gamp[k]*ws[k]
    end
    ret
end
