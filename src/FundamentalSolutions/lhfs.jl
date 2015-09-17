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
function lhfs(x::Float64, y::Float64, energies::Float64, derivs::Bool, stdquad::Int, h::Float64, meth::Int)
    a = 0.25(x^2+y^2)
    b = 0.5y+energies
    if a == 0
        if derivs
            u = Inf+ZIM
            ux = Inf+ZIM
            uy = Inf+ZIM
            return u,ux,uy
        else
            u = Inf+ZIM
            return u
        end
    else
        nquad = quad_nodes!(ts, ws, a, b, x, y, derivs, stdquad, h, meth)
        for k=1:nquad
            gam[k],gamp[k] = contour(a, b, ts[k])
            integ[k],integx[k],integy[k] = integrand(gam[k], a, b, x, y, derivs)
        end
        if derivs
            u = quadsum(integ,gamp,ws,nquad)
            ux = quadsum(integx,gamp,ws,nquad)
            uy = quadsum(integy,gamp,ws,nquad)
            return u,ux,uy
        else
            u = quadsum(integ,gamp,ws,nquad)
            return u
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
