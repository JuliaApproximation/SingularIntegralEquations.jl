# This file calculates the second application in Y.-S. Chan, A. C. Fannjiang, and G. H. Paulino,
# Integral equations with hypersingular kernels -- theory and applications to fracture mechanics,
# Int. J. Eng. Sci., 41:683--720, 2003.

using ApproxFun,SIE
include("Fractureaux.jl")

x = Fun(identity)
w = 1/sqrt(1-x^2)
d = domain(x)
d2 = d^2
B = dirichlet(d)
H0 = Hilbert(d,0)
H2 = Hilbert(d,2)
S = Σ(d)

β = -2.0
G = exp(β*x)

K = LowRankFun((x,y)->exp(β/2.*(y-x)).*β^2./2.^2.*(FK1(β.*(y-x)./2).*(log(abs(β)/4) + γ) - GK1(β.*(y-x)./2)./2),d2)
K0 = LowRankFun((x,y)->exp(β/2.*(y-x)).*β^2./2.^2.*FK1(β.*(y-x)./2),d2)
K2 = LowRankFun((x,y)->exp(β/2.*(y-x)),d2)

L = H2[K2*w] + H0[K0*w] + S[K*(w/π)]
p = -Fun(one)

uSIE = [B,L]\[zeros(2),p/G]
@time uSIE = [B,L]\[zeros(2),p/G]
println("The rank of K2 is: ",rank(K2)," the rank of K0 is: ",rank(K0),", and the rank of K is: ",rank(K),".")
println("The length of uSIE is: ",length(uSIE))
println("The extrema of uSIE are: ",extrema(uSIE))
println("The normalized Stress Intensity Factors are: ",(uSIE/(1-x^2)/G[1])[-1],"  ",(uSIE/(1-x^2)/G[-1])[1])

using PyCall
pygui(:tk)
using PyPlot

clf();m=length(uSIE)
spy([sparse(B[1][1:m]'),sparse(B[2][1:m]'),L[1:m-2,1:m]])
xlim([-.5,m-.5]);ylim([m-.5,-.5]);xlabel("\$j\$");ylabel("\$i\$")
savefig("Fracture_bspy.pdf")

clf()
semilogy(abs(uSIE.coefficients),"xk")
xlabel("\$n\$");ylabel("\$|u_n|\$")
savefig("Fracture_bcoeffs.pdf")

clf()
len=int(m/2)
err = zeros(len)
Cn = zeros(len)
for i=1:len
	n=2i+3
	An = [sparse(B[1][1:n]');sparse(B[2][1:n]');L[1:n-2,1:n]]
	fn = [0.0,0.0,pad(p.coefficients,n-2)]
	un = An\fn
	n1 = int(ceil(1.01n))
	An1 = [sparse(B[1][1:n1]');sparse(B[2][1:n1]');L[1:n1-2,1:n1]]
	fn1 = [0.0,0.0,pad(p.coefficients,n1-2)]
	un1 = An1\fn1
	err[i] = norm(pad(un,n1)-un1)
	Cn[i] = cond(full(An))
end
ax = gca()
ax[:spines][:"left"][:set_color]("r")
ax[:spines][:"right"][:set_color]("g")
semilogy([5:2:2len+3],err,"-r",label="Cauchy Errors")
tick_params(axis="y",colors="r")
xlabel("\$n\$");ylabel("\$||u_{\\lceil 1.01n\\rceil}-u_n||_2\$")
twinx();
semilogy([5:2:2len+3],Cn,"-.g",label="Condition Numbers")
tick_params(axis="y",colors="g")
ylabel("Condition Number")
xlim([5,2len+3])
savefig("Fracture_bCauchyCond.pdf")

uSIE /= 1-x^2
uSIE *= sqrt(1-x^2)/G[1]
clf();xx = linspace(-1,1,5001)
plot(xx,uSIE[xx],"-k")
xlabel("\$x\$");ylabel("\$w(x,0^+)/G(1)\$")
savefig("Fracture_bplot.pdf")