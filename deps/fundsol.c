/*
A program to evaluate the linear (or gravity) Helmholtz fundamental solution
(and possibly first derivatives with respect to the source) at an array of
target points and energies, with the source located at the
origin. Numerical steepest descent is used.

This version uses the method of steepest descent to evaluate the fundamental solution.
BJN 02/05/13 - created
             - added support for derivatives
BJN 06/28/14 - fundamental solution and its derivatives are computed using the same quadrature nodes.
BJN 09/09/14 - find_endpoints now looks at u, ux, and uy
AHB 9/13/14: replaced lhfs_derivs by lhfs, changed energy to be an array, cut lhfs and
    renamed from lhfs_derivs (note n and derivs swapped in arg list!)
See README for more updates (9/18/14 onwards)
*/

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <complex.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>

// some useful math constants.
#define M_PI_3 1.047197551196598
//#define M_PI_2 1.5707963267948966
//#define M_PI   3.1415926535897932384
//#define M_1_PI 0.31830988618379067
//#define M_PI_4 0.785398163397448309
#define THIRD  0.333333333333333333

#define W 0.363630003348128     // precomputed constant for finding stationary points
#define V -0.534877842831614    // precomputed constant for finding stationary points
#define jump_ratio 1.3          // when finding endpoints, increase distance from, 1.3
	                        // stationary point by this ratio
#define MAXNQUAD 3000           // maximum allowed number of quad points
#define CHUNKSIZE 200           // chunksize for openMP, 300

// default numerical params, and how they scale with h (when h<0 triggers it):
#define MAXH_STD 0.05   // max h spacing on real axis for quad-nodes meth=1, .03
#define MINSADDLEN_STD 43    // min nodes per saddle pt: 40
#define MAXH_OVER_H 0.13         // ratio in mode when h scales things: 0.14
#define MINSADDLEN_TIMES_H 15   // ratio in mode when 1/h scales things: 14


#ifdef DBTIMING
//#define NOPAR // don't want parallelization in this option
// global variables for timing tests
double quad_time;
double ctr_time;
double integ_time;
double tot_quad_pts;
#endif

// function declarations
void lhfs( double * x, double * y, double * energies, int derivs, int n, double complex * u, double complex * ux, double complex * uy, int stdquad, double h, int meth, int gamout, int* nquad);

static void integrand(double complex * u, double complex * ux, double complex * uy, double complex z, double a, double b, double x, double y, int derivs);

static void integral(double complex * integ, double complex * integx, double complex * integy, double complex *  gam, const double a, const double b, const int t_w_size, double x, double y, int derivs);

static void linspace(double * array, const double lowerbd, const double upperbd, const int num);
static void array_fill(double * array, const double fill_num, const int size);
static void ctr(double complex * gam, double complex * gamp, const double a,
                const double b, double * ts, const int size);


static void find_endpoints( double * lm1, double * lp1, double * lm2,
                           double * lp2, double * c1, double * c2, double * tmin,
			    double * fmin, double complex *tm, double complex *tp,
                           const double a, const double b,
                           const double x, const double y, const int derivs);

static void loc_min(double * tmin, double * fmin, double complex st1, double complex st3,
		    const double a, const double b, const double x, const double y,
		    const int derivs);

static void quad_nodes(double * ts, double * ws, int * n, const double a, const double b,
		       const double x, const double y, const int derivs, int stdquad,
		       double h, int meth);
void gamforbid(int n, int deriv, double complex s0, double *s,
	       double complex *gam, double complex *gamp);



/* lhfs
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
*/
void lhfs( double * x, double * y, double *energies, int derivs, int n, double complex * u, double complex * ux, double complex * uy, int stdquad, double h, int meth, int gamout, int *nquad)
{

  // variable declarations;
    double complex * gam;     // contour through complex plane
    double complex * gamp;    // derivative of contour
    double complex * integ;   // value of integrand
    double complex * integx;  // x derivative of integrand
    double complex * integy;  // y derivative of integrand
    double a, b;              // change of variables coordinates
    double * ts;              // paramterization points for contour
    double * ws;              // quadrature weights for each paramaterization point
    int t_w_size;             // number of points that we for quadrature scheme
    int chunk = CHUNKSIZE;    // openMP chunksize
    int i;                    // index variable (declared here for openMP)

    #ifdef DBTIMING
    // we are doing timing tests
    struct timeval start, end;
    double delta;
    omp_set_num_threads(1);
    #endif
    
    *nquad = 0;           // init quad pt counter

  // now to loop through finding the fundamental solution for each coordinate.
#ifdef NOPAR
    // don't do anything if NOPAR set - regular for loop over target points.
#else
    /*
     We use openMP to parallelize evaluation.  We group target points into batches of size CHUNKSIZE, and give those over to separate processes.
     Then for each target point we evaluate the integral which gives us the fundamental solution at that point.  This is done entirely inside an individual process.
     */
#pragma omp parallel for \
  shared(x,y,energies,n,derivs,u,ux,uy,stdquad,h,meth,nquad) \
  private(i,gam,gamp,integ,integx,integy,a,b,ts,ws,t_w_size) \
  schedule(dynamic,chunk)
#endif
  for( i = 0; i < n; i++ ) {

        // allocate memory
      gam  = (double complex *) malloc(MAXNQUAD * sizeof(double complex));
      gamp = (double complex *) malloc(MAXNQUAD * sizeof(double complex));
      integ = (double complex *) malloc(MAXNQUAD * sizeof(double complex));
      integx = (double complex *) malloc(MAXNQUAD * sizeof(double complex));
      integy = (double complex *) malloc(MAXNQUAD * sizeof(double complex));
      ts = (double *) malloc(MAXNQUAD * sizeof(double));
      ws = (double *) malloc(MAXNQUAD * sizeof(double));
    
      a = (x[i] * x[i] + y[i] * y[i]) / 4.0;
      b = y[i] / 2.0 + energies[i];
      
      if( a == 0 ) {
	u[i] = INFINITY;                 // hit the singularity!
	if (derivs) { ux[i] = INFINITY; uy[i] = INFINITY; }
      } else {
          
            #ifdef DBTIMING
            gettimeofday(&start, NULL);
            #endif
          
            quad_nodes(ts, ws, &t_w_size, a, b, x[i], y[i], derivs, stdquad, h, meth);
	    *nquad += t_w_size;    // increment total # quad pts
          
            #ifdef DBTIMING
            gettimeofday(&end, NULL);
            delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
            quad_time += delta;
            #endif
      
            #ifdef DBTIMING
            gettimeofday(&start, NULL);
            #endif
          
            ctr(gam, gamp, a, b, ts, t_w_size); // builds countour
          
            #ifdef DBTIMING
            gettimeofday(&end, NULL);
            delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
            ctr_time += delta;
            #endif
          
            #ifdef DBTIMING
            gettimeofday(&start, NULL);
            #endif
          
            integral(integ, integx, integy, gam, a, b, t_w_size, x[i], y[i], derivs);  // computes integrand

            #ifdef DBTIMING
            gettimeofday(&end, NULL);
            delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
            integ_time += delta;
            #endif

	    if (derivs)                                 // do the quadrature sum
	      for(int k = 0; k < t_w_size; k++ ) {
                u[i] = u[i] + integ[k] * gamp[k] * ws[k];
                ux[i] = ux[i] + integx[k] * gamp[k] * ws[k];
                uy[i] = uy[i] + integy[k] * gamp[k] * ws[k];
	      }
	    else                                  // don't write into ux uy
	      for(int k = 0; k < t_w_size; k++ )
		u[i] = u[i] + integ[k] * gamp[k] * ws[k];
      }

      if (gamout) {
	// write out the nodes of the last contour used in the loop (Alex)
	if (i==n-1) {
	  FILE *out = fopen("nodes.dat","w");      // nodes used
	  if (!out) printf("failed to open output file nodes.dat\n");
	  for (int k=0; k<t_w_size; k++)
	    fprintf(out,"%.16g %.16g %.16g\n", creal(gam[k]), cimag(gam[k]),
		    cabs(integ[k] * gamp[k] * ws[k]));
	  fclose(out);
	}
      }
       // free memory
      free(ts);
      free(ws);
      free(gam);
      free(gamp);
      free(integ);
      free(integx);
      free(integy);      
  }  // end i loop

  if (gamout) {
    i=n-1; a = (x[i] * x[i] + y[i] * y[i]) / 4.0; // last a,b, param values
    b = y[i] / 2.0 + energies[i];
    FILE *out = fopen("ctr.dat","w");       // output full contour at finer nodes
    if (!out) printf("failed to open output file ctr.dat\n");
    n = 1e5;          // gamma vals to output
    gam  = (double complex *) malloc(n * sizeof(double complex));
    gamp = (double complex *) malloc(n * sizeof(double complex));
    ts = (double *) malloc(n * sizeof(double));
    for (int k=0; k<n; k++)
      ts[k] = -10.0 + 0.002*k;   // linear spacing on real axis
    ctr(gam, gamp, a, b, ts, n);
    for (int k=0; k<n; k++)
      fprintf(out,"%.16g %.16g\n", creal(gam[k]), cimag(gam[k]));
    fclose(out);
    free(ts);
    free(gam);
    free(gamp);
  }
  return;
}


/*
 quad_nodes
 Description:
    Sets up quadrature nodes and weights to be used to evaluate the fundamental solution integral.
 Parameters:
    ts      (output) - array of size n with quadrature node locations
    ws      (output) - array of size n with quadrature node weights
    n       (input) - size of ts/ws arrays
    a       (input) - change of variables coordinate
    b       (input) - change of variables coordinate
    x       (input) - cartesian x coordinate
    y       (input) - cartesian y coordinate
    derivs  (input) - boolean determining whether to evaluate fundamental solution derivatives
     stdquad (input, int)     - global convergence params: # default quad pts per saddle pt
     h (input, double)        - PTR spacing relative to std Gaussian sigma
                         (if h<0, -h used, and also scales maxh & minsaddlen)
     meth (input) - int giving method for choosing # nodes: 0 (old, Brad); 1 (Alex)
*/
static void quad_nodes(double * ts, double * ws, int * n, 
                       const double a, const double b, const double x, const double y, const int derivs,
		       int stdquad, double h, int meth) {
 
    double lm1, lp1, lm2, lp2, c1, c2; // endpoint markers
    double tmin, fmin;
    double maxh = MAXH_STD;
    int minsaddlen = MINSADDLEN_STD;
    complex double tm, tp;

    if (meth==1 && fabs(h)<=0.001) {
      fprintf(stderr,"meth=1: option h must be >1e-3 in size; setting to 0.3\n"); h=0.3;
    }

    find_endpoints(&lm1, &lp1, &lm2, &lp2, &c1, &c2, &tmin, &fmin, &tm, &tp, a, b, x, y, derivs);
    //printf("tm = %.16g, tp = %.16g\n",creal(tm),creal(tp));
    double sigm = 1.0/sqrt(cabs(a/tm + b*tm - 0.75*tm*tm*tm));  // saddle widths in s plane
    double sigp = 1.0/sqrt(cabs(a/tp + b*tp - 0.75*tp*tp*tp));  // via phase''(s)
    if (h<0) {  // use h to scale two other convergence params
      h = fabs(h);
      minsaddlen = (int)ceil(MINSADDLEN_TIMES_H/h);
      maxh = MAXH_OVER_H*h;
    }
    //printf("h=%g maxh=%g minsaddlen=%d\n",h,maxh,minsaddlen);


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


void gamforbid(int n, int deriv, double complex s0, double *s,
	       double complex *gam, double complex *gamp)
/* Evaluate gamma (and if deriv=true, gamma') at the n nodes s, for the forbidden
   region (regions 1 and 2). s0 is the lower saddle point. Used by find_endpoints()
   and ctr(). No other definitions of this contour in the code.
   Barnett 9/15/14. Using new region 2 contour, all cleaned up
*/
{
  int j;
  double res = creal(s0), ims = cimag(s0);
  double d, d2, ex, g0;
  
  //printf("deriv=%d, n=%d, s0=%g+%gi\n",deriv,n,res,ims);
  if (ims <= -M_PI_3) {      // region 1 (deep forbidden)
    if (deriv) {
      for(j = 0; j < n; j++) {
	d = s[j] - res;               // x-displacement in s plane from s0
	gam[j] = THIRD * atan(d) - M_PI_3;
	gamp[j] = THIRD / (1 + d*d);
      }
    } else {       // no gamp needed
      for(j = 0; j < n; j++) {
	d = s[j] - res;
	gam[j] = THIRD * atan(d) - M_PI_3;
      }
    }
  } else {  	// region 2 (shallow forbidden); Alex redesigned this contour 9/15/14
    if (ims>-0.1)
      // hack to move ctr just below coalescing saddle at high E:
      ims = ims - fmaxf(0.0,fminf(0.7*exp(-res),0.1));
    // note exp(res) ~ sqrt(E).

    if (deriv) {
      for(j = 0; j < n; j++) {
	d = s[j] - res;
	d2 = d*d;
	ex = exp(-d2);
	g0 = THIRD*(atan(d)-M_PI);
	gam[j] = ims + (g0 - ims)*(1.0-ex);
	gamp[j] = 2.0*(g0 - ims)*d*ex + THIRD / (1.0 + d2) * (1.0-ex);
      }
    } else {    // no gamp needed
      for(j = 0; j < n; j++) {
	d = s[j] - res;
	ex = exp(-d*d);
	g0 = THIRD*(atan(d)-M_PI);
	gam[j] = ims + (g0 - ims)*(1.0-ex);
      }
    }
  }
  if (deriv) {
    for(j = 0; j < n; j++) { // Either region: convert gam from y-coord to pt in C-plane
      gam[j] = s[j] + I * gam[j];
      gamp[j] = 1.0 + I * gamp[j];
    }
  } else {
    for(j = 0; j < n; j++) { // Either region: convert gam from y-coord to pt in C-plane
      gam[j] = s[j] + I * gam[j];
    }
  }
}


/*
Computes the endpoints for fundamental solution integral by stepping
along the contour until it finds a point whose value is smaller
than machine epsilon
*/
static void find_endpoints( double * lm1, double * lp1, double * lm2, 
                           double * lp2, double * c1, double * c2, double * tmin, double * fmin,
			    double complex *tm, double complex *tp,        // passes saddle pts in t plane
                           const double a, const double b,
                           const double x, const double y, const int derivs)
{
  double t, stdist, eps = 1e-14; //DBL_EPSILON;  // abs truncation error param
  double complex gam;
  // compute stat phase (saddle) pts...
  double complex temp1 = csqrt(b * b - a);
  *tm = csqrt(2 * (b - temp1));   // stat phase ts in t
  *tp = csqrt(2 * (b + temp1));
  //if (cabs(*tm) > cabs(*tp)) {    // swap so always log(tm)-st3 has smaller Re part
  // t = *tp; *tp = *tm; *tm = t;
  //}
  double complex st1 = clog(*tp);   // stat phase pts in s
  double complex st3 = clog(*tm);
  if( cimag(st3) >= M_PI_2 )
    st3 = st3 - I * M_PI;           // hack fixing branch cut of clog
  //printf("sm = %.16g, sp = %.16g\n",creal(st3),creal(st1));


  if( b <= sqrt(a) ){  // forbidden region (1 or 2)....................................
    // Alex updated this section using region1+2 function gamforbid 9/15/14
    *c1 = creal(st1);
    stdist = abs(*c1) / 2.0;
    if( stdist == 0) 
      stdist = 1;
      
      // first find Lm1
      t = *c1 - stdist;
      gamforbid(1, 0, st3, &t, &gam, &gam); // reconstructs gam at t
      
      double dlm1 = 1;
      double complex u, ux, uy;
      integrand( &u, &ux, &uy, gam, a, b, x, y, derivs);
      if( (cabs(u) > eps)  || (cabs(ux) > eps) || (cabs(uy) > eps) ) {
          while ((cabs(u) > eps)  || (cabs(ux) > eps) || (cabs(uy) > eps) ) {
              dlm1 *= jump_ratio;
              t = *c1 - dlm1 * stdist;
	      gamforbid(1, 0, st3, &t, &gam, &gam); // rebuild: reconstructs gam at t
              integrand( &u, &ux, &uy, gam, a, b, x, y, derivs);
          }
      } else {
          while (((cabs(u) < eps)  && (cabs(ux) < eps) && (cabs(uy) < eps)) && dlm1 > eps) {
              dlm1 = dlm1 / jump_ratio;
              t = *c1 - dlm1 * stdist;
	      gamforbid(1, 0, st3, &t, &gam, &gam); // rebuild: reconstructs gam at t
              integrand( &u, &ux, &uy, gam, a, b, x, y, derivs);
          }
          if (dlm1 < eps) dlm1 = 0;
          dlm1 *= jump_ratio;
      }
      *lm1 = *c1 - dlm1 * stdist;
      
      
      // now to find *lp1
      t = *c1 + stdist;
      gamforbid(1, 0, st3, &t, &gam, &gam); // reconstructs gam at t
      dlm1 = 1.0;
      integrand( &u, &ux, &uy, gam, a, b, x, y, derivs);
      if( (cabs(u) < eps)  && (cabs(ux) < eps) && (cabs(uy) < eps) ) {
          while (((cabs(u) < eps)  && (cabs(ux) < eps) && (cabs(uy) < eps)) && dlm1 > eps) {
              dlm1 = dlm1 / jump_ratio;
              t = *c1 + dlm1 * stdist;              
	      gamforbid(1, 0, st3, &t, &gam, &gam); // rebuild: reconstructs gam at t    
	      integrand( &u, &ux, &uy, gam, a, b, x, y, derivs);
          }
          if (dlm1 < eps) dlm1 = 0;
          dlm1 *= jump_ratio;
      } else {
          while ((cabs(u) > eps)  || (cabs(ux) > eps) || (cabs(uy) > eps) ) {
              dlm1 *= jump_ratio;
              t = *c1 + dlm1 * stdist;
	      gamforbid(1, 0, st3, &t, &gam, &gam); // rebuild: reconstructs gam at t
	      integrand( &u, &ux, &uy, gam, a, b, x, y, derivs);
          }
      }
      *lp1 = *c1 + dlm1 * stdist;
      
      
      *c2 = *c1;
      *lp2 = 0;
      *lm2 = 0;

      
      
  } else { //region 3 (allowed):  we need to integrate over 2 intervals.............
  // (Brad's code)
      // finds start points of intervals
      if(creal(st1) < creal(st3)) {
          *c1 = creal(st1);
          *c2 = creal(st3);
      } else {
          *c1 = creal(st3);
          *c2 = creal(st1);
      }
      
      // find minimum between stationary points
      
      loc_min(tmin, fmin, st1, st3, a, b, x, y, derivs);
      //tmin = (*c2 + *c1) / 2.0;
      //fmin = 0.0;
      
      
      stdist = fabs(*c1 - *tmin);
      if( stdist == 0) 
          stdist = 1;
      //double stdist = (*c2 - *c1) / 2.0;;
      t = *c1 - stdist;
      
      // precalculations for f, g
      double f1 = (1.0 / M_PI + 0.5);
      double f2 = (-creal(st3) + W / 2);
      double f3 = (-M_PI_4 + 0.5);
      double g1 = (1.0 / M_PI + 1.0 / 6.0);
      double g2 = (-creal(st1) - V / 4);
      double g3 = (-M_PI / 12.0 + 0.5);
      
      double f = f1 * atan( 2 * ( t + f2)) + f3;
      double g = g1 * atan (-4 * (t + g2)) + g3;
      
      gam = t + I * f * g;
      
      double dlm1 = 1;
      double complex u, ux, uy;
      integrand( &u, &ux, &uy, gam, a, b, x, y, derivs);
      if( (cabs(u) > eps)  || (cabs(ux) > eps) || (cabs(uy) > eps) ) {
          while ((cabs(u) > eps)  || (cabs(ux) > eps) || (cabs(uy) > eps) ) {
              dlm1 *= jump_ratio;
              t = *c1 - dlm1 * stdist;
              
              // rebuild contour
              f = f1 * atan( 2 * ( t + f2)) + f3;
              g = g1 * atan (-4 * (t + g2)) + g3;
              gam = t + I * f * g;
              integrand( &u, &ux, &uy, gam, a, b, x, y, derivs);
          }
      } else {
          while (((cabs(u) < eps)  && (cabs(ux) < eps) && (cabs(uy) < eps)) && dlm1 > eps) {
              dlm1 = dlm1 / jump_ratio;
              t = *c1 - dlm1 * stdist;

              // rebuild contour
              f = f1 * atan( 2 * ( t + f2)) + f3;
              g = g1 * atan (-4 * (t + g2)) + g3;
              gam = t + I * f * g;
              integrand( &u, &ux, &uy, gam, a, b, x, y, derivs);
          }
          if (dlm1 < eps) dlm1 = 0;
          dlm1 *= jump_ratio;
      }
      *lm1 = *c1 - dlm1 * stdist;
      
      
      // now to find *lp1
      t = *c1 + stdist;
      
      f = f1 * atan( 2 * ( t + f2)) + f3;
      g = g1 * atan (-4 * (t + g2)) + g3;
      gam = t + I * f * g;
      
      dlm1 = 1.0;
      integrand( &u, &ux, &uy, gam, a, b, x, y, derivs);
      if( (cabs(u) < eps)  && (cabs(ux) < eps) && (cabs(uy) < eps) ) {
          while (((cabs(u) < eps)  && (cabs(ux) < eps) && (cabs(uy) < eps)) && dlm1 > eps) {
              dlm1 = dlm1 / jump_ratio;
              t = *c1 + dlm1 * stdist;
              
              // rebuild contour
              f = f1 * atan( 2 * ( t + f2)) + f3;
              g = g1 * atan (-4 * (t + g2)) + g3;
              gam = t + I * f * g;
              integrand( &u, &ux, &uy, gam, a, b, x, y, derivs);
          }
          if (dlm1 < eps) dlm1 = 0;
          dlm1 *= jump_ratio;
      } else {
          while ((cabs(u) > eps)  || (cabs(ux) > eps) || (cabs(uy) > eps)) {
              dlm1 *= jump_ratio;
              t = *c1 + dlm1 * stdist;
              
              // rebuild contour
              f = f1 * atan( 2 * ( t + f2)) + f3;
              g = g1 * atan (-4 * (t + g2)) + g3;
              gam = t + I * f * g;
              integrand( &u, &ux, &uy, gam, a, b, x, y, derivs);
          }
      }
      *lp1 = *c1 + dlm1 * stdist;
      
      
      // now to find lm2
      stdist = fabs(*c2 - *tmin);
      if( stdist == 0) 
          stdist = 1;
      
      t = *c2 - stdist;
   
      f = f1 * atan( 2 * ( t + f2)) + f3;
      g = g1 * atan (-4 * (t + g2)) + g3;
      gam = t + I * f * g;
      
      dlm1 = 1.0;
      integrand( &u, &ux, &uy, gam, a, b, x, y, derivs);

      if( (cabs(u) > eps)  || (cabs(ux) > eps) || (cabs(uy) > eps) ) {
          while ((cabs(u) > eps)  || (cabs(ux) > eps) || (cabs(uy) > eps) ) {
              dlm1 *= jump_ratio;
              t = *c2 - dlm1 * stdist;
              
              // rebuild contour
              f = f1 * atan( 2 * ( t + f2)) + f3;
              g = g1 * atan (-4 * (t + g2)) + g3;
              gam = t + I * f * g;
              integrand( &u, &ux, &uy, gam, a, b, x, y, derivs);
          }
      } else {
          while (((cabs(u) < eps)  && (cabs(ux) < eps) && (cabs(uy) < eps)) && dlm1 > eps) {
              dlm1 = dlm1 / jump_ratio;
              t = *c2 - dlm1 * stdist;
              
              // rebuild contour
              f = f1 * atan( 2 * ( t + f2)) + f3;
              g = g1 * atan (-4 * (t + g2)) + g3;
              gam = t + I * f * g;
              integrand( &u, &ux, &uy, gam, a, b, x, y, derivs);
          }
          if (dlm1 < eps) dlm1 = 0;
          dlm1 *= jump_ratio;
      }
      *lm2 = *c2 - dlm1 * stdist;

      
      // now to find lp2
      t = *c2 + stdist;
      
      f = f1 * atan( 2 * ( t + f2)) + f3;
      g = g1 * atan (-4 * (t + g2)) + g3;
      gam = t + I * f * g;
      
      dlm1 = 1.0;
      integrand( &u, &ux, &uy, gam, a, b, x, y, derivs);
      if( (cabs(u) > eps)  || (cabs(ux) > eps) || (cabs(uy) > eps) ) {
          while ((cabs(u) > eps)  || (cabs(ux) > eps) || (cabs(uy) > eps) ) {
              dlm1 *= jump_ratio;
              t = *c2 + dlm1 * stdist;
              
              // rebuild contour
              f = f1 * atan( 2 * ( t + f2)) + f3;
              g = g1 * atan (-4 * (t + g2)) + g3;
              gam = t + I * f * g;
              integrand( &u, &ux, &uy, gam, a, b, x, y, derivs);
          }
      } else {
          while (((cabs(u) < eps)  && (cabs(ux) < eps) && (cabs(uy) < eps)) && dlm1 > eps) {
              dlm1 = dlm1 / jump_ratio;
              t = *c2 + dlm1 * stdist;
              
              // rebuild contour
              f = f1 * atan( 2 * ( t + f2)) + f3;
              g = g1 * atan (-4 * (t + g2)) + g3;
              gam = t + I * f * g;
              integrand( &u, &ux, &uy, gam, a, b, x, y, derivs);
          }
          if (dlm1 < eps) dlm1 = 0;
          dlm1 *= jump_ratio;
      }
      *lp2 = *c2 + dlm1 * stdist;
      
      // now for some final clean up
      if(*lm2 < *c1) {
          if(*lm1 < *lm2) {
              *lm1 = *lm2;
          }
          *lm2 = *c2;
      }
      if(*lp1 > *c2) {
          if(*lp1 < *lp2) {
              *lp2 = *lp1;
          }
          *lp1 = *c1;
      }
      
      
  }

  return;

}

// analagous to MATLAB linspace function
static void linspace(double * array, const double lowerbd, const double upperbd, const int num) {

  double space = (upperbd - lowerbd) / (num - 1);
  for( int i = 0; i < num; i++) {  // alex changed to not repeatedly add (rounding err)
    array[i] = lowerbd + space*i;
  }

  return;

}

// fills up array with a constant
static void array_fill(double * array, const double fill_num, const int size) {

  for( int i = 0; i < size; i++) {
    array[i] = fill_num;
  }
 
}

// Evaluates quadr nodes (gam) on contour at real parts given by array ts (of given size),
// and derivatives of contour (gamp), given params a,b.
static void ctr(double complex * gam, double complex * gamp, const double a, const double b, double * ts, const int size)
{
  int i,j;
  //Precalculations for stat pts
  double complex temp1 = csqrt(b * b - a);
  double temp2 = sqrt(a);
  // Relevant stationary points of the integrand
  double complex st1 = clog(csqrt(2 * (b + temp1)));
  double complex st3 = clog(csqrt(2 * (b - temp1)));
  if (creal(st3) > creal(st1)) {    // swap so st3 has smaller Re part for ctr only
    temp1 = st1; st1 = st3; st3 = temp1;
  }
  if(cimag(st3) >= M_PI_2) st3 = st3 - I * M_PI;  // hack to deal with brach of clog
    
  //constructs gam, gamp
  if (b <= temp2) {        // region 1 or 2 ( forbidden): uses Alex's function
    int deriv = 1;
    gamforbid(size, deriv, st3, ts, gam, gamp);
  } else {                       // region 3 (classically allowed)
    double imsh = 0.0;          // imag shift if coal saddles
      // hack to move ctr just below coalescing saddle at high E:
    if (fabs(creal(st1)-creal(st3))<0.1) { // close saddles
	imsh =  -fmaxf(0.0,fminf(0.7*exp(-creal(st1)),0.1));
	// note exp(re(s)) ~ sqrt(E). See also gamforbid()
      }

    double f, fp, g, gp;
    double temp1c, temp2c;
    for(j = 0; j < size ; j++) {
      temp1c=2 * (ts[j] - creal(st3) + W / 2.0);
      temp2c=-4 * (ts[j] - creal(st1) - V / 4.0);
      
      f = (M_1_PI + 0.5) * atan(temp1c) - (M_PI_4 - 0.5);
      fp = 2 * (M_1_PI + 0.5) / (1.0 + temp1c * temp1c);
      
      g = (M_1_PI + 1.0 / 6.0 ) * atan(temp2c) - (M_PI_4 / 3.0 - 0.5);
      gp = -4 * (M_1_PI + 1.0 / 6.0) / (1.0 + temp2c * temp2c);
      
      gam[j] = f * g + imsh;    // Alex: imag shift
      gamp[j] = fp * g + f * gp;
    }
    for(i = 0; i < size; i++) { // region 3: convert gam from y-coord to pt in C-plane 
      gam[i] = ts[i] + I * gam[i];
      gamp[i] = 1.0 + I * gamp[i];
    }
  }
  return;
}


// analagous to integrand in code
static void integral(double complex * integ, double complex * integx, double complex * integy, double complex *  gam, const double a, const double b, const int t_w_size, double x, double y, int derivs)
{

  for( int i = 0; i < t_w_size; i++ ) {
    integrand( &( integ[i] ) , &( integx[i] ), &( integy[i] ), gam[i], a, b, x, y, derivs);
  } 
  
  return;

}


static void integrand(double complex * u, double complex * ux, double complex * uy, double complex z, double a, double b, double x, double y, int derivs) {
  // Evaluates s-plane integrand for u (and possibly ux,uy) given params a,b, at s=z.
    // Precalculations
    double complex cez = cexp(z);
    double complex icez = 1.0/cez;
    double complex exponential = cexp(I * (a * icez + b * cez - cez * cez * cez / 12.0) );
    
    *u = exponential; // value case
    
    if (derivs) {       // just a true/false flag now - AHB
        // calculate prefactors for exponentials in order to do derivatives
        double complex prefactor;
        double complex c = I / 2.0 * icez; // saves another division - AHB
        double complex d = I * cez / 2.0;
        // x deriv computation
        prefactor = -1.0 * c * x;
        *ux = prefactor * exponential;
        // y deriv computation
        prefactor = -c * y + d;
        *uy = prefactor * exponential;
    } else { // make sure that ux, uy aren't going to give us anything funny (or >emach)
        *ux = 0;
        *uy = 0;
    }

    return;

}

// finds local minimum between st1 and st3 along contour
// works by taking minimum over uniform sampling of points and then iterating once.
// returns t value of minimum along contour: tmin
// also returns integrand value at minimum: fmin
static void loc_min(double * tmin, double * fmin,
                    double complex st1, double complex st3, const double a, const double b, const double x, const double y, const int derivs){
    
    int n_t = 20; // zise of t array
    double * ts = malloc(n_t * sizeof(double));
    double complex * gam = malloc(n_t * sizeof(double complex));
    double complex * integ = malloc(n_t * sizeof(double complex));
    double complex * integx = malloc(n_t * sizeof(double complex));
    double complex * integy = malloc(n_t * sizeof(double complex));
    
    double f, g;
    double temp1c, temp2c;
    
    double t1 = creal(st1);
    double t2 = creal(st3);
    
    int n_its = 2; // number of loops
    for( int k = 0; k < n_its; k++){ 
        
        linspace( ts, t1, t2, n_t);
        
        for(int j = 0; j < n_t ; j++) {
            // Precalculations                                                                
            temp1c=2 * (ts[j] - creal(st3) + W / 2.0);
            temp2c=-4 * (ts[j] - creal(st1) - V / 4.0);
            
            f = (M_1_PI + 0.5) * atan(temp1c) - (M_PI_4 - 0.5);
            
            g = (M_1_PI + 1.0 / 6.0 ) * atan(temp2c) - (M_PI_4 / 3.0 - 0.5);
            
            gam[j] = ts[j] + I * f * g;
            
            integx[j] = 0;
            integy[j] = 0;
            
            integrand(&(integ[j]), &(integx[j]), &(integy[j]), gam[j], a, b, x, y, derivs);
            
        }
        
        double current_min = cabs(integ[0]);
        int current_i = 0;
        for(int j = 1; j < n_t; j++) {
            double test_min = sqrt(cabs(integ[j]) + cabs(integx[j]) + cabs(integy[j])); // function to be minimized: two norm of u, ux, uy
            if( test_min < current_min ) {
                current_min = test_min;
                current_i = j;
            }
        }
        
        if( current_i == 0 || current_i == 20-1 || k == n_its-1) {
            *fmin = current_min;
            *tmin = ts[current_i];
            k = n_its;
        } else {
            t1 = ts[current_i - 1];
            t2 = ts[current_i + 1];
        }
        
    }
    
    free(ts);
    free(gam);
    free(integ);
    free(integx);
    free(integy);
    
    
}

//////////////////////////////////////////////////// various MAINs FOR TIMING, etc /////

#ifdef TIMING

int main()
{

  // initialize timing variables
  //  clock_t start_c, diff;
    struct timeval start, end;

  // initialize test variables
    int num_pts = 300000;
    double r;
    double r_min = 0.001;
    double r_max = 20.0;
    double theta;
    double * xs;
    double * ys;
    double *Es;
    int derivs;
    double complex * u;
    double complex * ux;
    double complex * uy;
    
    xs = (double *) malloc(num_pts * sizeof(double));
    ys = (double *) malloc(num_pts * sizeof(double));
    Es = (double *) malloc(num_pts * sizeof(double));
    u = (double complex *) malloc(num_pts * sizeof(double complex));
    ux = (double complex *) malloc(num_pts * sizeof(double complex));
    uy = (double complex *) malloc(num_pts * sizeof(double complex));

    // grid around center at (0,0)

    //initstate(1);
    
    for (int i = 0; i < num_pts; i++) {
        theta = (( ((int) rand()) % 100) / 100.0 ) * (2 * M_PI);
        r = (( ((int) rand()) % 100) /100.0 ) * r_max + r_min;        
        xs[i] = r * cos(theta);
        ys[i] = r * sin(theta);
        Es[i] = 10.0;            // choose energy
    }
   
    // omp_set_num_threads(1);

  // store start time
    gettimeofday(&start, NULL);
    //  start_c = clock();

  // do timing stuff here
    derivs = 1;
    int stdquad=200; // method params
    double h=0.35;
    int meth=1;
    int gamout=0;
    int nquad;
    lhfs( xs, ys, Es, derivs, num_pts, u, ux, uy,stdquad,h,meth,gamout,&nquad);
    
  // get end time
    gettimeofday(&end, NULL);
  // diff = clock() - start_c;
  //  int msec = diff * 1000 / CLOCKS_PER_SEC;
  //  double seconds = (diff * 1000000 / CLOCKS_PER_SEC) / 1000000.0;

    double delta = ((end.tv_sec  - start.tv_sec) * 1000000u + 
		    end.tv_usec - start.tv_usec) / 1.e6;

    //printf("%.6f seconds elapsed, %d milliseconds.\n",seconds, msec);
    //printf("that is %.2f pts evaluated per second.\n",num_pts/seconds);

    printf("%.6f seconds elapsed.\n",delta);
    printf("that is %.2f pts evaluated per second.\n",num_pts/delta);
    
    int num_procs = omp_get_num_procs();
    printf("%d processes used.\n",num_procs);
    printf("%.2f pts/sec/proc.\n",num_pts/delta/num_procs);

    // time cexps
    if(1) {
        double complex a;
        gettimeofday(&start,NULL);
        for (int i = 0; i < num_pts; i++) {
          a = xs[i] + I * ys[i];
          u[i] = cexp(a);
        }
        gettimeofday(&end,NULL);

        delta = ((end.tv_sec  - start.tv_sec) * 1000000u +
                        end.tv_usec - start.tv_usec) / 1.e6;

        printf("%.2f cexp/sec for one process.\n",num_pts/delta);
    }
    free(xs);
    free(ys);
    free(Es);
    free(u);
    free(ux);
    free(uy);
    
  return 1;
}

#endif

#ifdef DBTIMING

int main()
{
    // set up time globals
    quad_time = 0;
    ctr_time = 0;
    integ_time = 0;
    tot_quad_pts = 0;
    
    
    // initialize test variables
    int num_pts = 100000;
    double r;
    double r_min = 0.001;
    double r_max = 20.0;
    double theta;
    double * xs;
    double * ys;
    double * Es;
    int derivs;
    double complex * u;
    double complex * ux;
    double complex * uy;
    
    xs = (double *) malloc(num_pts * sizeof(double));
    ys = (double *) malloc(num_pts * sizeof(double));
    Es = (double *) malloc(num_pts * sizeof(double));
    u = (double complex *) malloc(num_pts * sizeof(double complex));
    ux = (double complex *) malloc(num_pts * sizeof(double complex));
    uy = (double complex *) malloc(num_pts * sizeof(double complex));
    
    // set up coordinates
    for (int i = 0; i < num_pts; i++) {
        theta = (( ((int) rand()) % 100) / 100.0 ) * (2 * M_PI);
        r = (( ((int) rand()) % 100) /100.0 ) * r_max + r_min;
        Es[i] = 10.0;                   // choose energy
        xs[i] = r * cos(theta);
        ys[i] = r * sin(theta);
    }
    
    // do timing stuff here
    derivs = 0;
    int stdquad=200; // method params
    double h=0.35;
    int meth=1;
    int gamout=0;
    int nquad;
    lhfs( xs, ys, Es, derivs, num_pts, u, ux, uy,stdquad,h,meth,gamout,&nquad);
    
    
    // display time data
    double total = quad_time + ctr_time + integ_time;
    printf("%.2f quad pts per eval avg. \n", tot_quad_pts/num_pts);
    printf("%.2f time spent in finding quadrature nodes. %.2f of total time. \n", quad_time, quad_time/total);
    printf("%.2f time spent in constructing contour.  %.2f of total time.\n", ctr_time, ctr_time/total);
    printf("%.2f time spent in evaluating integral.  %.2f of total time.\n", integ_time, integ_time/total);
    printf("%.2f seconds total. %d eval points total.\n", total, num_pts);
    printf("%.2f points/sec average.\n", num_pts/total);
    
    return 1;
    
}

#endif

#ifdef DEBUG_OLD

int main()
{
    
    
    double complex st1, st3;
    double a,b;
    double tmin, fmin;
    
    st1 = 1.0843;
    st3 = 0.54834;
    a = 6.5464;
    b = 2.9349;
    
    
    loc_min(&tmin, &fmin, st1, st3, a, b, 1, 1, 0, 0);
    
    printf("st1 = %f, st3 = %f, a = %f, b = %f\n", creal(st1), creal(st3), a, b);
    printf("tmin = %.16f\n fmin = %.16f\n", tmin, fmin);
    
    st1 = 1.2918;
    st3 = 1.1205;
    a = 31.1373;
    b = 5.6621;
    
    
    loc_min(&tmin, &fmin, st1, st3, a, b, 1, 1, 0, 0);
    
    printf("st1 = %f, st3 = %f, a = %f, b = %f\n", creal(st1), creal(st3), a, b);
    printf("tmin = %.16f\n fmin = %.16f\n", tmin, fmin);
    
    st1 = 1.2171;
    st3 = 0.34348;
    a = 5.6686;
    b = 3.3488;
    
    
    loc_min(&tmin, &fmin, st1, st3, a, b, 1, 1, 0, 0);
    
    printf("st1 = %f, st3 = %f, a = %f, b = %f\n", creal(st1), creal(st3), a, b);
    printf("tmin = %.16f\n fmin = %.16f\n", tmin, fmin);
    
    return 0;
}

#endif

#ifdef DEBUG

int main()
{

    
    double x = -4.429;
    double y = 6.192;
    double energy = 1;
    int derivs = 1;
    int n = 1;
    double complex u, ux, uy;
    
    int stdquad=200; // method params
    double h=0.3;
    int meth=0;
    int gamout=0;
    int nquad;
    lhfs( &x, &y, &energy, derivs, n, &u, &ux, &uy,stdquad,h,meth,gamout,&nquad);
    
    printf("x = %.16f; y= %.16f; E= %.16f\nu = %.16f + %.16fi\n", x, y, energy, creal(u), cimag(u));
    printf("ux = %.16f + %.16fi\n", creal(ux), cimag(ux));
    printf("uy = %.16f + %.16fi\n", creal(uy), cimag(uy));
    printf("nquad = %d\n",nquad);
 



  return 0;
}


#endif

#ifdef DEBUG_OLD

int main()
{
    
    
    double complex gam, gamp;
    double a = 11.8;
    double b = 3.57;
    double t = 1;
    int size = 1;
    
    
    printf("ctr:\n");    
    ctr(&gam, &gamp, a, b, &t, size);
    
    printf("a = %.16f, b = %.16f, t = %.16f\ngam = %.16f + %.16fi\ngamp = %.16f + %.16fi\n\n", a, b, t, creal(gam), cimag(gam), creal(gamp), cimag(gamp) );
    
    printf("find_endpoints:\n");
    double lm1, lp1, lm2, lp2, c1, c2;
    
    find_endpoints(&lm1, &lp1, &lm2, 
                   &lp2, &c1, &c2,
                   a, b);
    
    printf("lm1 = %.16f\nlp1 = %.16f\nlm2 = %.16f\nlp2 = %.16f\nc1 = %.16f\nc2 = %.16f\n\n", lm1, lp1, lm2, lp2, c1, c2);
    
    
    return 0;
}


#endif

