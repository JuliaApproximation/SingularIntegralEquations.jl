a=1+10*im;b=2-6*im
d=Curve(Fun(x->1+a*x+b*x^2))


x=Fun(d)
w=sqrt(abs(first(d)-x))*sqrt(abs(last(d)-x))


@test_approx_eq cauchy(w,2.) sum(w/(x-2.))/(2π*im)
