# some useful math constants.
const M_PI_3 = 1.047197551196598
const M_PI_2 = 1.5707963267948966
const M_PI = 3.1415926535897932384
const M_1_PI = 0.31830988618379067
const M_1_4PI = 7.957747154594767e-02
const M_PI_4 = 0.785398163397448309

const M_F_1 = M_1_PI + 0.5
const M_F_2 = M_PI_4 - 0.5
const M_G_1 = M_1_PI + 1.0 / 6.0
const M_G_2 = M_PI_4 / 3.0 - 0.5

const THIRD = 0.333333333333333333
const ZIM = 0.0im

const W = 0.363630003348128         # precomputed constant for finding stationary points
const V = -0.534877842831614        # precomputed constant for finding stationary points
const W2 = 0.5*0.363630003348128    # precomputed constant for finding stationary points
const V4 = -0.25*0.534877842831614  # precomputed constant for finding stationary points
const jump_ratio = 1.3              # when finding endpoints, increase distance from, 1.3
	                                # stationary point by this ratio

const MAXNQUAD = 3000               # maximum allowed number of quad points

# Allocation
const gam = Vector{Complex{Float64}}(undef,MAXNQUAD)
const gamp = Vector{Complex{Float64}}(undef,MAXNQUAD)
const integ = Vector{Complex{Float64}}(undef,MAXNQUAD)
const integx = Vector{Complex{Float64}}(undef,MAXNQUAD)
const integy = Vector{Complex{Float64}}(undef,MAXNQUAD)
const ts = Vector{Float64}(undef,MAXNQUAD)
const ws = Vector{Float64}(undef,MAXNQUAD)

# locate_minimum allocation

const n_its = 2
const n_t = 20
const lm_ts = Vector{Float64}(undef,n_t)
const lm_gam = Vector{Complex{Float64}}(undef,n_t)
const lm_integ = Vector{Complex{Float64}}(undef,n_t)
const lm_integx = Vector{Complex{Float64}}(undef,n_t)
const lm_integy = Vector{Complex{Float64}}(undef,n_t)
const test_min = Vector{Float64}(undef,n_t)

# find_endpoints allocation

const fe_gam = Vector{Complex{Float64}}(undef,1)
const fe_u = Vector{Complex{Float64}}(undef,1)
const fe_ux = Vector{Complex{Float64}}(undef,1)
const fe_uy = Vector{Complex{Float64}}(undef,1)

# default numerical params, and how they scale with h (when h<0 triggers it):
const MAXH_STD = 0.05           # max h spacing on real axis for quad-nodes meth=1, .03
const MINSADDLEN_STD = 43       # min nodes per saddle pt: 40
const MAXH_OVER_H = 0.13        # ratio in mode when h scales things: 0.14
const MINSADDLEN_TIMES_H = 15   # ratio in mode when 1/h scales things: 14
