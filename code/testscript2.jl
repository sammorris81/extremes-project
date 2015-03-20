
cd()
cd("./repos-git/extremes-project/code")

# packages
using Distributions
using Distances
using ProfileView
using NumericExtensions

# constants
const ns = 200
const nt = 40
const np = 3
# replaces expand.grid from R
const knots_x = linspace(0.01, 0.99, 9)
const knots = hcat(repeat(knots_x, inner=[1], outer=[9]),
             repeat(knots_x, inner=[9], outer=[1]))
const nknots = size(knots)[1]
const s = hcat(rand(ns), rand(ns))
const x = fill(1.0, ns, nt, 3)
for t = 1:nt
  x[:, t, 2] = s[:, 1]
  x[:, t, 3] = s[:, 2]
end

# modules
using AuxFunctions
using UpdateModel

# testing update for beta
# nreps = 2000
# burn = 200
const xi_t = 0.1
const beta_t = [1.0, -1.0, 0.0]
const theta_star_t = fill(1.0, ns, nt)
const alpha_t = 1.0
const rho_t = 1.0
const thresh = 0.0
const y = sample_rarebinary(x, s, knots, beta_t, xi_t, alpha_t, rho_t, thresh)


@time sample_beta!(1, y, theta_star_t, alpha_t, xi_t, x, thresh, ns, nt, np)

@time sample_beta!(2000, y, theta_star_t, alpha_t, xi_t, x, thresh, ns, nt, np)
