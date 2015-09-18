#
# Evaluate gamma (and if gamp is passed, evaluate gamma' as well) at the n nodes s
# for the forbidden region (regions 1 and 2). s0 is the lower saddle point.
# Used by find_endpoints() and contour(). No other definitions of this contour in the code.
#

function gammaforbidden!(gam::Vector{Complex{Float64}}, gamp::Vector{Complex{Float64}}, s0::Complex{Float64}, s::Vector{Float64}, n::Int)
    res,ims = reim(s0)
    @simd for i=1:n
        d = s[i] - res
        # region 1 (deep forbidden)
        if ims ≤ -M_PI_3
            @inbounds gam[i] = complex(s[i],THIRD*atan(d) - M_PI_3)
            @inbounds gamp[i] = complex(1.0,THIRD/(1+d*d))
        else
            ims = imshack(res,ims)
            d2 = d^2
            ex = exp(-d2)
            g0 = THIRD*(atan(d)-M_PI)
            @inbounds gam[i] = complex(s[i],ims + (g0 - ims)*(1.0-ex))
            @inbounds gamp[i] = complex(1.0,2.0*(g0 - ims)*d*ex + THIRD / (1.0 + d2) * (1.0-ex))
        end
    end
end

# assume if length n not given that vectors are of length 1 (with a slightly different signature)

function gammaforbidden!(gam::Vector{Complex{Float64}}, s0::Complex{Float64}, s::Float64)
    res,ims = reim(s0)
    d = s - res
    # region 1 (deep forbidden)
    if ims ≤ -M_PI_3
        gam[1] = complex(s,THIRD*atan(d) - M_PI_3)
    else
        ims = imshack(res,ims)
        d2 = d^2
        ex = exp(-d2)
        g0 = THIRD*(atan(d)-M_PI)
        gam[1] = complex(s,ims + (g0 - ims)*(1.0-ex))
    end
end

function imshack(res::Float64,ims::Float64)
    # hack to move ctr just below coalescing saddle at high E. Note exp(res) ~ sqrt(E).
    if ims > -0.1 ims -= max(0.0,min(0.7exp(-res),0.1)) end
    return ims
end
