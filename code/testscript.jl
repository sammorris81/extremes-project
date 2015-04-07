
cd()
cd("./repos-git/extremes-project/code")

# packages
using Distributions
using Distances
using ProfileView

# constants
const ns = 200
const nt = 40
const np = 3
# replaces expand.grid from R
const knots_x = linspace(0.01, 0.99, 9)
const knots = hcat(repeat(knots_x, inner=[1], outer=[9]),
             repeat(knots_x, inner=[9], outer=[1]))
const nknots = size(knots)[1]

# modules
importall AuxFunctions
importall UpdateModel

const s = hcat(rand(ns), rand(ns))
const x = fill(1.0, ns, nt, 3)
for t = 1:nt
  x[:, t, 2] = s[:, 1]
  x[:, t, 3] = s[:, 2]
end

# testing update for beta
nreps = 2000
burn = 200
xi_t = [0.1]
beta_t = [1.0, -1.0, 0.0]
theta_star_t = fill(1.0, ns, nt)
alpha_t = [1.0]
rho_t = [1.0]
thresh = 0.0
y = sample_rarebinary(x, s, knots, beta_t, xi_t, alpha_t, rho_t, thresh)

# initialization and prior distribution
beta = [0.0, 0.0, 0.0]
beta_m = [0.0, 0.0, 0.0]
beta_s = [10.0, 10.0, 10.0]
x_beta = fill(0.0, ns, nt)
can_x_beta = copy(x_beta)
z = get_z(xi_t, x_beta, thresh)
can_z = copy(z)
cur_lly = logpdf_rarebinary(y, theta_star_t, alpha_t, z)
can_lly = copy(cur_lly)

# MH stuff
att_beta = fill(1, 3)
acc_beta = fill(1, 3)
mh_beta = fill(0.1, 3)
candidate_beta = Normal(0, 1)

@time update_beta!(y, theta_star_t, alpha_t, z, can_z, beta,
               beta_m, beta_s, xi_t, x, x_beta, can_x_beta, cur_lly, can_lly,
               att_beta, acc_beta, mh_beta, thresh, candidate_beta)

#storage
beta_keep = fill(0.0, nreps, 3)
@time for i = 1:nreps
  update_beta!(y, theta_star_t, alpha_t, z, can_z, beta,
               beta_m, beta_s, xi_t, x, x_beta, can_x_beta, cur_lly, can_lly,
               att_beta, acc_beta, mh_beta, thresh, candidate_beta)
  beta_keep[i, :] = beta

  if i < (burn / 2)
    for p = 1:3
      (att_beta[p], acc_beta[p], mh_beta[p]) = mh_update(att_beta[p],
                                                         acc_beta[p],
                                                         mh_beta[p], 50, 0.8,
                                                         1.2)
    end
  end

  if i % 500 == 0
    println("Iter")
  end
end

using Gadfly
plot(x=1:nreps, y=beta_keep[:, 1], Geom.line)
