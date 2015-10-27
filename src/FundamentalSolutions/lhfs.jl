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
        contour!(gam, gamp, a, b, ts, nquad)
        integrand!(integ, integx, integy, gam, a, b, x, y, derivs, nquad)
        if derivs
            u,ux,uy = quadsum3(integ,integx,integy,gamp,ws,nquad)
            return u,ux,uy
        else
            u = quadsum(integ,gamp,ws,nquad)
            return u
        end
    end
end

function lhfs!(u::Vector{Complex{Float64}}, x::Vector{Float64}, y::Vector{Float64}, energies::Vector{Float64}, derivs::Bool, stdquad::Int, h::Float64, meth::Int, n::Int)
    for i=1:n
        a = 0.25(x[i]^2+y[i]^2)
        b = 0.5y[i]+energies[i]
        if a == 0
            u[i] = Inf+ZIM
        else
            nquad = quad_nodes!(ts, ws, a, b, x[i], y[i], derivs, stdquad, h, meth)
            contour!(gam, gamp, a, b, ts, nquad)
            integrand!(integ, integx, integy, gam, a, b, x[i], y[i], derivs, nquad)
            quadsum!(u,i,integ,gamp,ws,nquad)
        end
    end
end

function lhfs!(u::Vector{Complex{Float64}}, ux::Vector{Complex{Float64}}, uy::Vector{Complex{Float64}}, x::Vector{Float64}, y::Vector{Float64}, energies::Vector{Float64}, derivs::Bool, stdquad::Int, h::Float64, meth::Int, n::Int)
    for i=1:n
        a = 0.25(x[i]^2+y[i]^2)
        b = 0.5y[i]+energies[i]
        if a == 0
            u[i] = Inf+ZIM
            ux[i] = Inf+ZIM
            uy[i] = Inf+ZIM
        else
            nquad = quad_nodes!(ts, ws, a, b, x[i], y[i], derivs, stdquad, h, meth)
            contour!(gam, gamp, a, b, ts, nquad)
            integrand!(integ, integx, integy, gam, a, b, x[i], y[i], derivs, nquad)
            quadsum3!(u,ux,uy,i,integ,integx,integy,gamp,ws,nquad)
        end
    end
end

function quadsum!(u::Vector, i::Int, integ::Vector, gamp::Vector, ws::Vector, n::Int)
    @inbounds u[i] = integ[1]*gamp[1]*ws[1]
    @simd for k=2:n
        @inbounds u[i] += integ[k]*gamp[k]*ws[k]
    end
end

function quadsum3!(u::Vector, ux::Vector, uy::Vector, i::Int, integ::Vector, integx::Vector, integy::Vector, gamp::Vector, ws::Vector, n::Int)
    @inbounds temp = gamp[1]*ws[1]
    @inbounds u[i] = integ[1]*temp
    @inbounds ux[i] = integx[1]*temp
    @inbounds uy[i] = integy[1]*temp
    @simd for k=2:n
        @inbounds temp = gamp[k]*ws[k]
        @inbounds u[i] += integ[k]*temp
        @inbounds ux[i] += integx[k]*temp
        @inbounds uy[i] += integy[k]*temp
    end
end

function quadsum(integ::Vector, gamp::Vector, ws::Vector, n::Int)
    @inbounds ret = integ[1]*gamp[1]*ws[1]
    @simd for k=2:n
        @inbounds ret += integ[k]*gamp[k]*ws[k]
    end
    ret
end

function quadsum3(integ::Vector, integx::Vector, integy::Vector, gamp::Vector, ws::Vector, n::Int)
    @inbounds temp = gamp[1]*ws[1]
    @inbounds ret1 = integ[1]*temp
    @inbounds ret2 = integx[1]*temp
    @inbounds ret3 = integy[1]*temp
    @simd for k=2:n
        @inbounds temp = gamp[k]*ws[k]
        @inbounds ret1 += integ[k]*temp
        @inbounds ret2 += integx[k]*temp
        @inbounds ret3 += integy[k]*temp
    end
    ret1,ret2,ret3
end
