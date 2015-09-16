#
# Evaluates quadrature nodes (gam) on contour at real parts given by array ts (of given size),
# and derivatives of contour (gamp), given params a,b.
#
function contour!(gam::Vector{Complex{Float64}}, gamp::Vector{Complex{Float64}}, a::Float64, b::Float64, ts::Vector{Float64})
    # Complex precalculations for stationary points
    temp1 = sqrt(b^2-a+ZIM)
    temp2 = sqrt(a+ZIM)
    # Relevant stationary points of the integrand
    st1 = log(sqrt(2.0(b + temp1)))
    st3 = log(sqrt(2.0(b - temp1)))
    # swap so st3 has smaller real part for contour only
    if real(st3) > real(st1)
        st1,st3 = st3,st1
    end
    # hack to deal with brach of log
    if imag(st3) ≥ M_PI_2 st3 -= im*M_PI end
    # construct gam, gamp
    if b ≤ temp2
        gammaforbidden!(st3,ts,gam,gamp)
    else
        # hack to move ctr just below coalescing saddle at high E:
        # note exp(re(s)) ~ sqrt(E). See also gammaforbidden!()
        imsh = 0.0
        if abs(real(st1)-real(st3)) < 0.1
            imsh = imshack(real(st1),imsh)
        end

        temp1c=2.0(ts - real(st3)) + W2
        temp2c=-4.0(ts - real(st1)) + V4

        f = (M_1_PI + 0.5) * atan(temp1c) - (M_PI_4 - 0.5)
        fp = 2.0(M_1_PI + 0.5) ./ (1.0 + temp1c.^2)

        g = (M_1_PI + 1.0 / 6.0 ) * atan(temp2c) - (M_PI_4 / 3.0 - 0.5)
        gp = -4 * (M_1_PI + 1.0 / 6.0) ./ (1.0 + temp2c.^2)

        gam[:] = complex(ts,f.*g+imsh)
        gamp[:] = complex(1.0,fp.*g+f.*gp)
    end
end
