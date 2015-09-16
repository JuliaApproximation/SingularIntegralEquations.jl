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
#≤≥
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
    sigp = 1.0/sqrt(cabs(a/tp + b*tp - 0.75*tp*tp*tp))

    if h < 0
        h = -h
        minsaddlen = ceil(Int,MINSADDLEN_TIMES_H/h)
        maxh = MAXH_OVER_H*h
    end

    n2 = 0
    if ( ( lm2 == c2 ) || ( lp1 == c1) ) && c1 != c2
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
            ts[:] = linspace(lm1,lp2,n2)
            fill!(ws,dist1)

























    int n2;
    double dist1, dist2;
    if ( ( ( lm2 == c2 ) || ( lp1 == c1) ) && c1 != c2 ) {
        // two maxima, no die-off in middle

      //printf("3, no dieoff\n");
      if (meth==0) {
        double lga = log(a) / log(10);
        if (lga < -0.5) {
            n2 = (int) (2 * stdquad - stdquad * lga);
            if (n2 > MAXNQUAD - stdquad) n2 = MAXNQUAD - stdquad;
        } else {
            n2 = 2 * stdquad;
        }

        dist1 = (lp2 - lm1) / ( n2 - 1);

        linspace(ts, lm1, lp2, n2);
        array_fill(ws, dist1, n2);
	*n = n2;
      } else if (meth==1) {  // use worst-case saddle-pt width and h
	double h1 = fminf(h*fminf(sigm,sigp), maxh);     // why fmin not avail???
	int n1 = ceil((lp2-lm1)/h1);
	// following keeps enough pts at coalescing saddle pts (sig is too wide)...
	if (n1<minsaddlen) { n1=minsaddlen; h1 = (lp2-lm1)/(n1-1); }
	if (n1>MAXNQUAD) n1=MAXNQUAD;           // prevent overflow
        linspace(ts, lm1, lm1+(n1-1)*h1, n1); // exact h1
        array_fill(ws, h1, n1);
	*n = n1;
	//printf("n1=%d h1=%.3g\n",n1,h1);
	//printf("%g %g\n", h*sigm, h*sigp);
	//printf("%.16g\n",M_PI-fmaxf(M_PI,-M_PI)); // shows only single prec
      }


    } else if (c1 != c2) {
        // two maxima, die-off in middle

      //printf("3, dieoff\n");
      if (meth==0) {

        double lga = log(a) / log(10);
        if (lga < -0.5) {
            n2 = (int) (stdquad - stdquad * lga);
            if (n2 > MAXNQUAD - stdquad) n2 = MAXNQUAD - stdquad;
        } else {
            n2 = stdquad;
        }

        // need to check to make sure intervals are non-trivial
        dist1 = (lp1 - lm1) / (n2 - 1);
        dist2 = (lp2 - lm2) / (stdquad - 1);

        linspace(ts, lm1, lp1, n2);
        array_fill(ws, dist1, n2);

        linspace(&(ts[n2]), lm2, lp2, stdquad);
        array_fill(&(ws[n2]), dist2, stdquad);
	*n = n2 + stdquad;

      } else if (meth==1) {  // use saddle-pt widths and h
	double h1 = h*sigm;
	if (h1>maxh) h1=maxh;
	int n1 = ceil((lp1-lm1)/h1);
 	if (n1>MAXNQUAD/2) n1=MAXNQUAD/2;           // prevent overflow
        linspace(ts, lm1, lm1+(n1-1)*h1, n1); // exact h1
        array_fill(ws, h1, n1);
	double h2 = h*sigp;
	if (h2>maxh) h2=maxh;
	int n2 = ceil((lp2-lm2)/h2);
	if (n2>MAXNQUAD/2) n2=MAXNQUAD/2;           // prevent overflow
	linspace(ts+n1, lm2, lm2+(n2-1)*h2, n2); // exact h2
        array_fill(ws+n1, h2, n2);
	*n = n1 + n2;
	//printf("n1=%d h1=%.3g     n2 = %d h2=%.3g\n",n1,h1,n2,h2);
      }

    } else {
        // one maximum
      //printf("1 or 2\n");
      if (meth==0) {
        n2 = stdquad;
        lp2 = lp1;
        dist1 = ( lp2 - lm1 ) / (stdquad - 1);
        linspace(ts, lm1, lp2, stdquad);
        array_fill(ws, dist1, n2);
	*n = stdquad;
      } else if (meth==1) {  // use worst-case saddle-pt width and h
	double h1 = fminf(h*sigp, maxh); // NB sigp only; why fmin not avail???
	int n1 = ceil((lp1-lm1)/h1);
	// following keeps enough pts at coalescing saddle pts (sig is too wide)...
	if (n1<minsaddlen) { n1=minsaddlen; h1 = (lp1-lm1)/(n1-1); }
	if (n1>MAXNQUAD) n1=MAXNQUAD;           // prevent overflow
        linspace(ts, lm1, lm1+(n1-1)*h1, n1); // exact h1
        array_fill(ws, h1, n1);
	*n = n1;
	//printf("n1=%d h1=%.3g\n",n1,h1);
      }

    }

    #ifdef DBTIMING
    tot_quad_pts += *n;
    #endif
    //  printf("# nodes = %d\n",*n);

}
