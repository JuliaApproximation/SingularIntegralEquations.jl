#
# Evaluates quadrature nodes (gam) on contour at real parts given by array ts (of given size),
# and derivatives of contour (gamp), given params a,b.
#

function contour!(gam::Vector{Complex{Float64}}, gamp::Vector{Complex{Float64}}, a::Float64, b::Float64, ts::Vector{Float64}, n::Int)
    # Complex precalculations for stationary points
    temp1 = sqrt(b^2-a+ZIM)
    temp2 = sqrt(a)
    # Relevant stationary points of the integrand
    st1 = 0.5log(2.0(b + temp1))
    st3 = 0.5log(2.0(b - temp1))
    # swap so st3 has smaller real part for contour only
    if real(st3) > real(st1) st1,st3 = st3,st1 end
    # hack to deal with brach of log
    if imag(st3) ≥ M_PI_2 st3 -= im*M_PI end
    # construct gam, gamp
    if b ≤ temp2
        gammaforbidden!(gam,gamp,st3,ts,n)
    else
        # hack to move ctr just below coalescing saddle at high E:
        # note exp(re(s)) ~ sqrt(E). See also gammaforbidden!()
        imsh = 0.0
        if abs(real(st1)-real(st3)) < 0.1
            imsh = imshack(real(st1),imsh)
        end
        @simd for i=1:n
            @inbounds temp1c=2.0(ts[i] - real(st3)) + W
            @inbounds temp2c=-4.0(ts[i] - real(st1)) + V

            f = M_F_1 * atan(temp1c) - M_F_2
            fp = 2.0M_F_1 / (1.0 + temp1c^2)

            g = M_G_1 * atan(temp2c) - M_G_2
            gp = -4 * M_G_1 / (1.0 + temp2c^2)

            @inbounds gam[i] = complex(ts[i],f*g+imsh)
            @inbounds gamp[i] = complex(1.0,fp*g+f*gp)
        end
    end
end
