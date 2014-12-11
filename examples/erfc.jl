using ApproxFun, SIE

f=FFun(z->2exp(z.^2),[-Inf*im,Inf*im])

erfc2(z)=real(z)>0?-exp(-z^2)*cauchy(f,z):exp(-z^2)*(2-cauchy(f,z))

erfc(1.)-erfc(1.)
