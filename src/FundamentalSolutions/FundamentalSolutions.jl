module FundamentalSolutions
using InteractiveUtils
#
# This module calculates:
#
# Φ(x,y,E), the fundamental solution of the Helmholtz equation in a linearly stratified medium:
#
# -(Δ + E + x_2)Φ(x,y,E) = δ(x-y)
#
# also known as the gravity Helmholtz equation.
#
# This is a Julia translation of the C-library described in:
#
# A. H. Barnett, B. J. Nelson, and J. M. Mahoney, High-order boundary integral equation solution of high frequency wave scattering from obstacles in an unbounded linearly stratified medium, J. Comp. Phys., 297:407–426, 2015.
#
# ℜ(x,y,E), the Riemann function of the Helmholtz equation in a linearly stratified medium. The bivariate series is described in:
#
# R. M. Slevinsky and S. Olver, A fast and well-conditioned spectral method for singular integral equations, arXiv:1507.00596, 2015.
#

export lhelmfs, lhelm_riemann

include("constants.jl")
include("integrand.jl")
include("gammaforbidden.jl")
include("contour.jl")
include("locate_minimum.jl")
include("find_endpoints.jl")
include("quad_nodes.jl")
include("lhfs.jl")

include("lhelmfs.jl")
include("lhelm_riemann.jl")

end #module

using .FundamentalSolutions

export lhelmfs, lhelm_riemann
