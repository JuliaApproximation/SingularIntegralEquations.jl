module FundamentalSolutions

#
# This module calculates Φ(x,y,E), the fundamental solution of the Helmholtz equation in a linearly stratified medium:
#
# -(Δ + E + x_2)Φ(x,y,E) = δ(x-y)
#
# also known as the gravity Helmholtz equation.
#
# This is a Julia translation of the C-library described in:
#
# A. H. Barnett, B. J. Nelson, and J. M. Mahoney, High-order boundary integral equation solution of high frequency wave scattering from obstacles in an unbounded linearly stratified medium, J. Comp. Phys., 297:407–426, 2015.
#

export lhelmfs, lhelm_riemann

include("constants.jl")
include("lhelmfs.jl")
include("lhelm_riemann.jl")

end #module
