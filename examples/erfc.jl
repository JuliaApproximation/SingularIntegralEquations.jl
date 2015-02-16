using ApproxFun, SIE

f=Fun(z->2exp(z^2),Space(PeriodicLine(0.,π/2)))

erfc2(z)=real(z)>0?-exp(-z^2)*cauchy(f,z):exp(-z^2)*(2-cauchy(f,z))

erfc2(1.)-erfc(1.)
