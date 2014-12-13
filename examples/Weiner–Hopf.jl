using ApproxFun, SIE

f=Fun(z->2+cos(z+1/z),Circle());
T=ToeplitzOperator(f)
C=Cauchy(-1,domain(f))
@time u=(I+(1-f)*C)\(f-1)
L=ToeplitzOperator(1/(1+C*u))
U=ToeplitzOperator((1+u+C*u))
(T-U*L)[1:10,1:10]