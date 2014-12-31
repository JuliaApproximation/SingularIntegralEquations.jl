using ApproxFun, SIE

f=Fun(z->2exp(z.^2),PeriodicLine([-Inf*im,Inf*im]))

erfc2(z)=real(z)>0?-exp(-z^2)*cauchy(f,z):exp(-z^2)*(2-cauchy(f,z))

erfc(1.)-erfc(1.)
