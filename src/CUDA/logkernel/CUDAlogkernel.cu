#include <thrust/complex.h>

using namespace thrust;

extern "C"
{

__global__ void CUDAlogkernel(const double a, const double b, const int nu, const double *u, double *x, double *y, double *ret)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int n = sizeof(x)/sizeof(x[0]);

    const double pi = M_PI;
    const double lengthd = abs(b-a);
    const double C = 0.5*lengthd;

    complex<double> *z,*yv,*yk,*ykp1;

    z = new complex<double>[n];
    yv = new complex<double>[n];
    yk = new complex<double>[n];
    ykp1 = new complex<double>[n];


    z[i] = complex<double>(x[i],y[i]);
    z[i] = (a + b - 2.0*z[i])/(a - b);  // tocanonical(u,z)

    if (z[i].real() <= 1.0 && z[i].real() >= -1.0 && abs(z[i].imag()) <= 2.0e-14) {
        yv[i] = z[i]+complex<double>(0.0,1.0)*sqrt(1.0-z[i])*sqrt(z[i]+1.0);
    }
    else {
        yv[i] = z[i] - sqrt(z[i]-1.0)*sqrt(z[i]+1.0);  // updownjoukowskyinverse(true,z)
    }

    yk[i] = yv[i];
    ykp1[i] = yk[i]*yk[i];


    if ( nu >= 0 ) {
        ret[i] = -u[0]*log(abs(2.0*yk[i]/C));  // -logabs(2y/C)
        if ( nu >= 1 ) {
            ret[i] += -u[1]*yk[i].real();  // -real(yk)
            if ( nu >= 2 ) {
                ret[i] += u[2]*(log(abs(2.0*yk[i]/C))-0.5*ykp1[i].real()); // -ret[1]-.5real(ykp1)
                if ( nu >= 3) {
                    for (int nun = 3; nun<nu; nun++) {
                        ykp1[i] *= yv[i];
                        ret[i] += u[nun]*( yk[i].real()/(nun-2.0)-ykp1[i].real()/(nun-0.0) ); // real(yk)/(n-3)-real(ykp1)/(n-1)
                        yk[i] *= yv[i];
                    }
                }
            }
        }
    }
    ret[i] *= pi*C;
}

} // extern "C"
