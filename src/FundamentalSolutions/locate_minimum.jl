#
# finds local minimum between st1 and st3 along contour
# works by taking minimum over uniform sampling of points and then iterating once.
# returns t value of minimum along contour: tmin
# also returns integrand value at minimum: fmin
#
function locate_minimum(st1::Complex{Float64}, st3::Complex{Float64}, a::Float64, b::Float64, x::Float64, y::Float64, derivs::Bool)
    rst1,rst3 = real(st1),real(st3)
    t1,t2 = rst1,rst3
    tmin,fmin = 0.0,0.0
    # number of loops
    for k=1:n_its
        linspace!(lm_ts,t1,t2,n_t)
        # Precalculations
        for i=1:n_t
            temp1c=2.0(lm_ts[i] - rst3) + W
            temp2c=-4.0(lm_ts[i] - rst1) + V
            f = M_F_1 * atan(temp1c) - M_F_2
            g = M_G_1 * atan(temp2c) - M_G_2
            lm_gam[i] = complex(lm_ts[i],f*g)
        end
        integrand!(lm_integ,lm_integx,lm_integy,lm_gam,a,b,x,y,derivs,n_t)
        # function to be minimized: square of two norm of u, ux, uy
        for i=1:n_t
            test_min[i] = abs2(lm_integ[i])+abs2(lm_integx[i])+abs2(lm_integy[i])
        end

        fmin,idx = findmin(test_min)
        if idx == 1 || idx == n_t || k == n_its
            tmin = lm_ts[idx]
        else
            t1,t2 = lm_ts[idx-1],lm_ts[idx+1]
        end
    end
    return tmin,fmin
end
