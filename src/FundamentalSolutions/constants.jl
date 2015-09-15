# some useful math constants.
const M_PI_3 = 1.047197551196598
const M_PI_2 = 1.5707963267948966
const M_PI = 3.1415926535897932384
const M_1_PI = 0.31830988618379067
const M_PI_4 = 0.785398163397448309
const THIRD = 0.333333333333333333

const W = 0.363630003348128     # precomputed constant for finding stationary points
const V = -0.534877842831614    # precomputed constant for finding stationary points
const jump_ratio = 1.3          # when finding endpoints, increase distance from, 1.3
	                            # stationary point by this ratio
const MAXNQUAD = 3000           # maximum allowed number of quad points

# default numerical params, and how they scale with h (when h<0 triggers it):
const MAXH_STD = 0.05           # max h spacing on real axis for quad-nodes meth=1, .03
const MINSADDLEN_STD = 43       # min nodes per saddle pt: 40
const MAXH_OVER_H = 0.13        # ratio in mode when h scales things: 0.14
const MINSADDLEN_TIMES_H = 15   # ratio in mode when 1/h scales things: 14
