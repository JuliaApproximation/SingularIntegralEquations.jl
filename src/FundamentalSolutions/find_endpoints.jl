#
# Computes the endpoints for fundamental solution integral by stepping
# along the contour until it finds a point whose value is almost smaller
# than machine epsilon
#

function find_endpoints(a::Float64, b::Float64, x::Float64, y::Float64, derivs::Bool)
    ϵ = 1e-14
    temp1 = sqrt(b^2-a+ZIM)
    tm = sqrt(2.0(b-temp1))
    tp = sqrt(2.0(b+temp1))
    st1 = log(tp)
    st3 = log(tm)
    if imag(st3) ≥ M_PI_2 st3 -= im*M_PI end

    if b ≤ sqrt(a)
        c1 = real(st1)
        stdist = 0.5abs(c1)
        if stdist == 0 stdist = 1.0 end

        # We start with finding lm1
        t = c1 - stdist
        gammaforbidden!(fe_gam, st3, t)
        dlm1 = 1.0
        integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
        if abs(fe_u[1]) > ϵ || abs(fe_ux[1]) > ϵ || abs(fe_uy[1]) > ϵ
            while abs(fe_u[1]) > ϵ || abs(fe_ux[1]) > ϵ || abs(fe_uy[1]) > ϵ
                dlm1 *= jump_ratio
                t = c1 - dlm1*stdist
                gammaforbidden!(fe_gam, st3, t)
                integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
            end
        else
            while abs(fe_u[1]) < ϵ && abs(fe_ux[1]) < ϵ && abs(fe_uy[1]) < ϵ && dlm1 > ϵ
                dlm1 /= jump_ratio
                t = c1 - dlm1*stdist
                gammaforbidden!(fe_gam, st3, t)
                integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
            end
            if dlm1 < ϵ dlm1 = 0.0 end
            dlm1 *= jump_ratio
        end
        lm1 = c1 - dlm1*stdist

        # lp1 comes next
        t = c1 + stdist
        gammaforbidden!(fe_gam, st3, t)
        dlm1 = 1.0
        integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
        if abs(fe_u[1]) < ϵ && abs(fe_ux[1]) < ϵ && abs(fe_uy[1]) < ϵ
            while abs(fe_u[1]) < ϵ && abs(fe_ux[1]) < ϵ && abs(fe_uy[1]) < ϵ && dlm1 > ϵ
                dlm1 /= jump_ratio
                t = c1 + dlm1*stdist
                gammaforbidden!(fe_gam, st3, t)
                integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
            end
            if dlm1 < ϵ dlm1 = 0.0 end
            dlm1 *= jump_ratio
        else
            while abs(fe_u[1]) > ϵ || abs(fe_ux[1]) > ϵ || abs(fe_uy[1]) > ϵ
                dlm1 *= jump_ratio
                t = c1 + dlm1*stdist
                gammaforbidden!(fe_gam, st3, t)
                integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
            end
        end
        lp1 = c1 + dlm1*stdist

        c2 = c1
        lp2 = 0.0
        lm2 = 0.0
    else
        c1,c2 = real(st1),real(st3)
        if c1 ≥ c2
            c1,c2 = c2,c1
        end
        tmin,fmin = locate_minimum(st1, st3, a, b, x, y, derivs)

        f2 = -real(st3) + W2
        g2 = -real(st1) - V4

        stdist = abs(c1-tmin)
        if stdist == 0 stdist = 1.0 end

        # We start with finding lm1
        t = c1 - stdist

        f = M_F_1 * atan( 2 * ( t + f2)) - M_F_2
        g = M_G_1 * atan( -4 * (t + g2)) - M_G_2
        fe_gam[1] = complex(t,f*g)

        dlm1 = 1.0
        integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
        if abs(fe_u[1]) > ϵ || abs(fe_ux[1]) > ϵ || abs(fe_uy[1]) > ϵ
            while abs(fe_u[1]) > ϵ || abs(fe_ux[1]) > ϵ || abs(fe_uy[1]) > ϵ
                dlm1 *= jump_ratio
                t = c1 - dlm1*stdist
                f = M_F_1 * atan( 2 * ( t + f2)) - M_F_2
                g = M_G_1 * atan( -4 * (t + g2)) - M_G_2
                fe_gam[1] = complex(t,f*g)
                integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
            end
        else
            while abs(fe_u[1]) < ϵ && abs(fe_ux[1]) < ϵ && abs(fe_uy[1]) < ϵ && dlm1 > ϵ
                dlm1 /= jump_ratio
                t = c1 - dlm1*stdist
                f = M_F_1 * atan( 2 * ( t + f2)) - M_F_2
                g = M_G_1 * atan( -4 * (t + g2)) - M_G_2
                fe_gam[1] = complex(t,f*g)
                integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
            end
            if dlm1 < ϵ dlm1 = 0.0 end
            dlm1 *= jump_ratio
        end
        lm1 = c1 - dlm1*stdist

        # lp1 comes next
        t = c1 + stdist

        f = M_F_1 * atan( 2 * ( t + f2)) - M_F_2
        g = M_G_1 * atan( -4 * (t + g2)) - M_G_2
        fe_gam[1] = complex(t,f*g)

        dlm1 = 1.0
        integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
        if abs(fe_u[1]) < ϵ && abs(fe_ux[1]) < ϵ && abs(fe_uy[1]) < ϵ
            while abs(fe_u[1]) < ϵ && abs(fe_ux[1]) < ϵ && abs(fe_uy[1]) < ϵ && dlm1 > ϵ
                dlm1 /= jump_ratio
                t = c1 + dlm1*stdist
                f = M_F_1 * atan( 2 * ( t + f2)) - M_F_2
                g = M_G_1 * atan( -4 * (t + g2)) - M_G_2
                fe_gam[1] = complex(t,f*g)
                integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
            end
            if dlm1 < ϵ dlm1 = 0.0 end
            dlm1 *= jump_ratio
        else
            while abs(fe_u[1]) > ϵ || abs(fe_ux[1]) > ϵ || abs(fe_uy[1]) > ϵ
                dlm1 *= jump_ratio
                t = c1 + dlm1*stdist
                f = M_F_1 * atan( 2 * ( t + f2)) - M_F_2
                g = M_G_1 * atan( -4 * (t + g2)) - M_G_2
                fe_gam[1] = complex(t,f*g)
                integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
            end
        end
        lp1 = c1 + dlm1*stdist

        #
        # Next up: lm2
        #
        stdist = abs(c2-tmin)
        if stdist == 0 stdist = 1.0 end

        t = c2 - stdist

        f = M_F_1 * atan( 2 * ( t + f2)) - M_F_2
        g = M_G_1 * atan( -4 * (t + g2)) - M_G_2
        fe_gam[1] = complex(t,f*g)

        dlm1 = 1.0
        integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
        if abs(fe_u[1]) > ϵ || abs(fe_ux[1]) > ϵ || abs(fe_uy[1]) > ϵ
            while abs(fe_u[1]) > ϵ || abs(fe_ux[1]) > ϵ || abs(fe_uy[1]) > ϵ
                dlm1 *= jump_ratio
                t = c2 - dlm1*stdist
                f = M_F_1 * atan( 2 * ( t + f2)) - M_F_2
                g = M_G_1 * atan( -4 * (t + g2)) - M_G_2
                fe_gam[1] = complex(t,f*g)
                integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
            end
        else
            while abs(fe_u[1]) < ϵ && abs(fe_ux[1]) < ϵ && abs(fe_uy[1]) < ϵ && dlm1 > ϵ
                dlm1 /= jump_ratio
                t = c2 - dlm1*stdist
                f = M_F_1 * atan( 2 * ( t + f2)) - M_F_2
                g = M_G_1 * atan( -4 * (t + g2)) - M_G_2
                fe_gam[1] = complex(t,f*g)
                integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
            end
            if dlm1 < ϵ dlm1 = 0.0 end
            dlm1 *= jump_ratio
        end
        lm2 = c2 - dlm1*stdist

        # Last but not least: lp2
        t = c2 + stdist

        f = M_F_1 * atan( 2 * ( t + f2)) - M_F_2
        g = M_G_1 * atan( -4 * (t + g2)) - M_G_2
        fe_gam[1] = complex(t,f*g)

        dlm1 = 1.0
        integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
        if abs(fe_u[1]) < ϵ && abs(fe_ux[1]) < ϵ && abs(fe_uy[1]) < ϵ
            while abs(fe_u[1]) < ϵ && abs(fe_ux[1]) < ϵ && abs(fe_uy[1]) < ϵ && dlm1 > ϵ
                dlm1 /= jump_ratio
                t = c2 + dlm1*stdist
                f = M_F_1 * atan( 2 * ( t + f2)) - M_F_2
                g = M_G_1 * atan( -4 * (t + g2)) - M_G_2
                fe_gam[1] = complex(t,f*g)
                integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
            end
            if dlm1 < ϵ dlm1 = 0.0 end
            dlm1 *= jump_ratio
        else
            while abs(fe_u[1]) > ϵ || abs(fe_ux[1]) > ϵ || abs(fe_uy[1]) > ϵ
                dlm1 *= jump_ratio
                t = c2 + dlm1*stdist
                f = M_F_1 * atan( 2 * ( t + f2)) - M_F_2
                g = M_G_1 * atan( -4 * (t + g2)) - M_G_2
                fe_gam[1] = complex(t,f*g)
                integrand!(fe_u, fe_ux, fe_uy, fe_gam, a, b, x, y, derivs)
            end
        end
        lp2 = c2 + dlm1*stdist

        # final cleanup
        if lm2 < c1
            if lm1 < lm2
                lm1 = lm2
            end
            lm2 = c2
        end
        if lp1 > c2
            if lp1 < lp2
                lp2 = lp1
            end
            lp1 = c1
        end
    end
    return lm1,lp1,lm2,lp2,c1,c2,tm,tp
end
