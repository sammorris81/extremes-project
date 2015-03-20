# Currently still testing to make sure all of the update functions actually work

cd()
cd("./repos-git/extremes-project/code")

# packages
using Distributions
using Distances
using ProfileView

# modules
using AuxFunctions
using UpdateModel

const ns = 200
const nt = 40
s = hcat(rand(ns), rand(ns))
x = fill(1.0, ns, nt, 3)
for t = 1:nt
  x[:, t, 2] = s[:, 1]
  x[:, t, 3] = s[:, 2]
end
const np = 3
# replaces expand.grid from R
knots_x = linspace(0.01, 0.99, 9)
knots = hcat(repeat(knots_x, inner=[1], outer=[9]),
             repeat(knots_x, inner=[9], outer=[1]))
const nknots = size(knots)[1]

# testing update for beta
nreps = 2000
burn = 200
xi_t = 0.1
beta_t = [1.0, -1.0, 0.0]
theta_star_t = fill(1.0, ns, nt)
alpha_t = 1.0
rho_t = 1.0
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

@time sub(x, :, :, 1)
@time slice(x, :, :, 1)
@time x[:, :, 1]

update_beta!(y, theta_star_t, alpha_t, z, can_z, beta,
               beta_m, beta_s, xi_t, x, x_beta, can_x_beta, cur_lly, can_lly,
               att_beta, acc_beta, mh_beta, thresh, candidate_beta)

#storage
beta_keep = fill(0.0, nreps, 3)
for i = 1:nreps
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

@time update_beta!(y, theta_star_t, alpha_t, z, can_z, beta,
               beta_m, beta_s, xi_t, x, x_beta, can_x_beta, cur_lly, can_lly,
               att_beta, acc_beta, mh_beta, thresh, candidate_beta)

@time update_beta_2!(y, theta_star_t, alpha_t, z, can_z, beta,
               beta_m, beta_s, xi_t, x, x_beta, can_x_beta, cur_lly, can_lly,
               att_beta, acc_beta, mh_beta, thresh, candidate_beta)

beta = [0.0, 0.0, 0.0]
cur_lly = logpdf_rarebinary(y, theta_star_t, alpha_t, z)
can_lly = copy(cur_lly)
@time sample_beta!(50, y, theta_star_t, alpha_t, z, beta,
               beta_m, beta_s, xi_t, x, x_beta, cur_lly,
               att_beta, acc_beta, mh_beta, thresh)

# update timing on alpha
nreps = 10000
burn = 7000
xi_t = 0.1
beta_t = [1.0, -1.0, 0.0]
alpha_t = 0.8
rho_t = 1.0
thresh = 0.0
(y, a) = sample_rarebinary(x, s, knots, beta_t, xi_t, alpha_t, rho_t, thresh)
x_beta_t = fill(0.0, ns, nt)
for t in 1:nt
  x_beta_t[:, t] = reshape(x[:, t, :], ns, 3) * beta_t
end
z_t = get_z(xi_t, x_beta_t, thresh)

# initialization and prior distribution
npts = 70
u_beta_q = linspace(0, 1, npts + 1)
u_beta = quantile(Beta(0.5, 0.5), u_beta_q)
mid_points = fill(0.0, 70)
bin_width = fill(0.0, 70)
for i = 1:npts
  mid_points[i] = (u_beta[i] + u_beta[i + 1]) / 2
  bin_width[i] = u_beta[i + 1] - u_beta[i]
end
dw2 = Array(Float64, ns, nknots)
for j = 1:nknots, i = 1:ns
  dw2[i, j] = Distances.sqeuclidean(vec(s[i, :]), vec(knots[j, :]))
end
w = make_w(dw2, rho_t)
std_w!(w)
alpha = 0.5
theta_star = get_thetastar(w, a, alpha)
can_theta_star = copy(theta_star)

cur_llps = fill(0.0, 81, 365)
# seems to be much faster
logpdf_ps!(cur_llps, a, alpha, mid_points, bin_width)
can_llps = copy(cur_llps)

cur_lly = logpdf_rarebinary(y, theta_star, alpha, z_t)
can_lly = copy(cur_lly)

att_alpha = 1
acc_alpha = 1
mh_alpha = 0.1
alpha_keep = fill(0.0, 100)

@time for i = 1:50
  (alpha, att_alpha, acc_alpha) = update_alpha!(y, theta_star, can_theta_star, a, alpha, cur_lly, can_lly,
              cur_llps, can_llps, z_t, w, mid_points, bin_width,
              att_alpha, acc_alpha, mh_alpha)
  alpha_keep[i] = alpha
  if i < (burn / 2)
    (att_alpha, acc_alpha, mh_alpha) = mh_update(att_alpha, acc_alpha, mh_alpha,
                                                50, 0.8, 1.2)
  end
end

@time update_alpha!(y, theta_star, can_theta_star, a, alpha, cur_lly, can_lly,
              cur_llps, can_llps, z_t, w, mid_points, bin_width,
              att_alpha, acc_alpha, mh_alpha)

@time for i = 1:100
logpdf_ps!(cur_llps, a, alpha, mid_points, bin_width)
end

@time logpdf_ps!(cur_llps, a, alpha, mid_points, bin_width)