#=
 quad_nodes
 Description:
    Sets up quadrature nodes and weights to be used to evaluate the fundamental solution integral.
 Parameters:
    ts      (output) - array of size n with quadrature nodes
    ws      (output) - array of size n with quadrature weights
    a       (input) - change of variables coordinate
    b       (input) - change of variables coordinate
    x       (input) - cartesian x coordinate
    y       (input) - cartesian y coordinate
    derivs  (input) - boolean determining whether to evaluate fundamental solution derivatives
     stdquad (input, int)     - global convergence params: # default quad pts per saddle pt
     h (input, double)        - PTR spacing relative to std Gaussian sigma
                         (if h<0, -h used, and also scales maxh & minsaddlen)
     meth (input) - int giving method for choosing # nodes: 0 (old, Brad); 1 (Alex)
=#
function quad_nodes!(ts::Vector{Float64}, ws::Vector{Float64}, a::Float64, b::Float64, x::Float64, y::Float64, derivs::Bool, stdquad::Int, h::Float64, meth::Int)
    n = length(ts)
    @assert n == length(ws)

    maxh = MAXH_STD
    minsaddlen = MINSADDLEN_STD

    if meth == 1 && abs(h) ≤ 0.001
        warn("meth=1: option h must be > 1e-3 in size; setting to 0.3")
        h = 0.3
    end
    lm1,lp1,lm2,lp2,c1,c2,tm,tp = find_endpoints(a,b,x,y,derivs)
    sigm = 1.0/sqrt(abs(a/tm + b*tm - 0.75*tm*tm*tm))
    sigp = 1.0/sqrt(abs(a/tp + b*tp - 0.75*tp*tp*tp))

    if h < 0
        h = -h
        minsaddlen = ceil(Int,MINSADDLEN_TIMES_H/h)
        maxh = MAXH_OVER_H*h
    end

    n1,n2 = 0,0
    if ( lm2 == c2 || lp1 == c1 ) && c1 != c2
        # two maxima, no die-off in middle
        if meth == 0
            lga = log10(a)
            if lga < -0.5
                n2 = round(Int,2stdquad - stdquad*lga)
                if n2 > MAXNQUAD - stdquad n2 = MAXNQUAD - stdquad end
            else
                n2 = 2stdquad
            end
            dist1 = (lp2 - lm1) / ( n2 - 1)
            linspace!(ts,lm1,lp2,n2)
            ws[1:n2] = dist1*M_1_4PI
            n = n2
        elseif meth == 1
            h1 = min(h*min(sigm,sigp),maxh)
            n1 = ceil(Int,(lp2-lm1)/h1)
            if n1<minsaddlen n1=minsaddlen; h1=(lp2-lm1)/(n1-1) end
            if n1 > MAXNQUAD n1=MAXNQUAD end
            linspace!(ts,lm1,lm1+(n1-1)h1,n1)
            ws[1:n1] .= h1*M_1_4PI
            n = n1
        end
    elseif c1 != c2
        # two maxima, die-off in middle
        if meth == 0
            lga = log10(a)
            if lga < -0.5
                n2 = round(Int,stdquad - stdquad*lga)
                if n2 > MAXNQUAD - stdquad n2 = MAXNQUAD - stdquad end
            else
                n2 = stdquad
            end
            dist1 = (lp1-lm1)/(n2-1)
            dist2 = (lp2-lm)/(stdquad-1)
            linspace!(ts,lm1,lp1,n2)
            w[1:n2] = dist1
            n = n2 + stdquad
            linspace!(ts,lm2,lp2,n2,stdquad)
            ws[1+n2:n] .= dist2*M_1_4PI
        elseif meth == 1
            h1 = h*sigm
            if h1>maxh h1=maxh end
            n1 = ceil(Int,(lp1-lm1)/h1)
            if n1>MAXNQUAD/2 n1=div(MAXNQUAD,2) end
            linspace!(ts,lm1,lm1+(n1-1)h1,n1)
            ws[1:n1] = h1*M_1_4PI
            h2 = h*sigp
            if h2>maxh h2=maxh end
            n2 = ceil(Int,(lp2-lm2)/h2)
            if n2>MAXNQUAD/2 n2=div(MAXNQUAD,2) end
            n = n1+n2
            linspace!(ts,lm2,lm2+(n2-1)h2,n1,n2)
            ws[1+n1:n] .= h2*M_1_4PI
        end
    else
        # one maximum
        if meth == 0
            n2 = stdquad
            lp2 = lp1
            dist1 = (lp2-lm1)/(n2-1)
            linspace!(ts,lm1,lp2,n2)
            ws[1:n2] .= dist1*M_1_4PI
            n = n2
        elseif meth == 1
            h1 = min(h*sigp,maxh)
            n1 = ceil(Int,(lp1-lm1)/h1)
            if n1<minsaddlen n1=minsaddlen; h1=(lp1-lm1)/(n1-1) end
            if n1>MAXNQUAD n1=MAXNQUAD end
            linspace!(ts,lm1,lm1+(n1-1)h1,n1)
            ws[1:n1] .= h1*M_1_4PI
            n = n1
        end
    end
    n
end

function linspace!(ts::Vector, start::Real, stop::Real, n::Int)
    h = (stop-start)/(n-1)
    for i=1:n
        ts[i] = start+(i-1)*h
    end
end

function linspace!(ts::Vector, start::Real, stop::Real, n1::Int, n::Int)
    h = (stop-start)/(n-1)
    for i=1:n
        ts[i+n1] = start+(i-1)*h
    end
end
