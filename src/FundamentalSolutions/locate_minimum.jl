#
# finds local minimum between st1 and st3 along contour
# works by taking minimum over uniform sampling of points and then iterating once.
# returns t value of minimum along contour: tmin
# also returns integrand value at minimum: fmin
#
function locate_minimum!(tmin::Float64, fmin::Float64, st1::Complex{Float64}, st3::Complex{Float64}, a::Float64, b::Float64, x::Float64, y::Float64, derivs::Bool)
    n_t = 20
    ts = Array(Float64,n_t)
    gam = Array(Complex{Float64},n_t)
    integ = Array(Complex{Float64},n_t)
    integx = Array(Complex{Float64},n_t)
    integy = Array(Complex{Float64},n_t)

    t1,t2 = real(st1),real(st3)

    # number of loops
    n_its = 2
    for k=1:n_its

        ts[:] = linspace(t1, t2, n_t)

        # Precalculations
        temp1c=2.0(ts - t2) + W2
        temp2c=-4.0(ts - t1) + V4

        f = (M_1_PI + 0.5) * atan(temp1c) - (M_PI_4 - 0.5)
        g = (M_1_PI + 1.0 / 6.0 ) * atan(temp2c) - (M_PI_4 / 3.0 - 0.5)
        gam[:] = complex(ts,f.*g)

        integrand!(integ,integx,integy,gam,a,b,x,y,derivs)

        # function to be minimized: two norm of u, ux, uy
        #test_min = sqrt(abs(integ)+abs(integx)+abs(integy)) This is how it appears in the C-library
        test_min = sqrt(abs2(integ)+abs2(integx)+abs2(integy))

        fmin,idx = findmin(test_min)
        if idx == 1 || idx == n_t || k == n_its
            tmin = ts[idx]
        else
            t1,t2 = ts[idx-1],ts[idx+1]
        end
    end
end
