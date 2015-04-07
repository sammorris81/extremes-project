cd()
cd("./repos-git/extremes-project/code")

# packages
using Distributions
using Distances
using ProfileView

# constants
const ns = 200
const nt = 1
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

# testing update for beta
const nreps = 2000
const burn = 200
xi_t = [0.1]
beta_t = [1.0, -1.0, 0.0]
theta_star_t = fill(1.0, ns, nt)
alpha_t = [1.0]
rho_t = [1.0]
const thresh = 0.0
y = sample_rarebinary(x, s, knots, beta_t, xi_t, alpha_t, rho_t, thresh)

const beta_init = [0.0, 0.0, 0.0]
const beta_m = [0.0, 0.0, 0.0]
const beta_s = [5.0, 5.0, 5.0]
const xi_init = [0.0]
const xi_m = [0.0]
const xi_s = [0.5]
const a_init = fill(1.0, nknots, nt)
const alpha_init = [0.5]
const rho_init = [0.4]
const rho_upper = 1.5
const npts = 70

@time mcmc(y, s, x, knots, beta_init, beta_m, beta_s, xi_init, xi_m, xi_s,
     npts, thresh, 0.01, 0.1, 0.1, 0.1, 1.0, 50, 50, 200, 200, rho_init, rho_upper,
     alpha_init, a_init, nreps, burn, 100, 1)
# time for Julia 18 seconds
# time for R 26 seconds


function mcmc(y::Array{Int64, 2}, s::Array{Float64, 2}, x::Array{Float64, 3},
              knots::Array{Float64, 2},
              beta_init::Array{Float64, 1},
              beta_m::Array{Float64, 1}, beta_s::Array{Float64, 1},
              xi_init::Array{Float64, 1}, xi_m::Array{Float64, 1},
              xi_s::Array{Float64, 1},
              npts::Integer, thresh::Real,
              beta_tune::Real, xi_tune::Real, alpha_tune::Real,
              rho_tune::Real, A_tune::Real,
              beta_attempts::Integer, xi_attempts::Integer,
              alpha_attempts::Integer, rho_attempts::Integer,
              rho_init::Array{Float64, 1}, rho_upper::Real,
              alpha_init::Array{Float64, 1}, a_init::Array{Float64, 2},
              iters::Integer, burn::Integer, update::Integer, thin::Integer
              )

  # Initial setup
  p = size(x)[3]
  ns, nt = size(y)
  nknots = size(knots)[1]

  # get the initial set of weights for the sites and knots
  alpha = copy(alpha_init)
  rho   = copy(rho_init)
  dw2 = Array(Float64, ns, nknots)
  for j = 1:nknots, i = 1:ns
    dw2[i, j] = Distances.sqeuclidean(vec(s[i, :]), vec(knots[j, :]))
  end

  w = make_w(dw2, rho[1])
  std_w!(w)
  can_w = copy(w)
  w_star = w.^(1 / alpha[1])
  can_w_star = copy(w_star)

  # get the initial set of random effects
  a = copy(a_init)
  theta_star = updatethetastar(w_star, a)
  can_theta_star = copy(theta_star)

  # get initial z
  xi = copy(xi_init)
  beta = copy(beta_init)
  x_beta = fill(0.0, ns, nt)
  for t = 1:nt
    x_beta[:, t] = reshape(x[:, t, :], ns, np) * beta
  end
  can_x_beta = copy(x_beta)

  z = updatez(xi, x_beta, thresh)
  can_z = copy(z)
  z_star = z.^(1 / alpha[1])
  can_z_star = copy(z_star)

  quants = linspace(0.0, 1.0, npts + 1)
  u_beta = quantile(Beta(0.5, 0.5), quants)
  mid_points = fill(0.0, npts)
  bin_width = fill(0.0, npts)
  for i = 1:npts
    mid_points[i] = (u_beta[i + 1] + u_beta[i]) / 2
    bin_width[i]  = u_beta[i + 1] - u_beta[i]
  end

  # likelihood values
  cur_lly = logpdf_rarebinary(y, theta_star, z_star)
  can_lly = copy(cur_lly)
  cur_llps = logpdf_ps(a, alpha, mid_points, bin_width)
  can_llps = copy(cur_llps)

  # MH setup
  beta_att = fill(0, p)
  beta_acc = fill(0, p)
  beta_mh  = fill(beta_tune, p)
  beta_candidate = Normal(0, 1)

  xi_att = fill(0, 1)
  xi_acc = fill(0, 1)
  xi_mh  = fill(xi_tune, 1)
  xi_candidate = Normal(0, 1)

  cuts  = exp([-1, 0, 1, 2, 5, 10])
  a_acc = fill(0, 100)
  a_att = fill(0, 100)
  a_mh  = fill(A_tune, 100)

  alpha_att = fill(0, 1)
  alpha_acc = fill(0, 1)
  alpha_mh  = fill(alpha_tune, 1)
  alpha_candidate = Normal(0, 1)

  rho_att = fill(0, 1)
  rho_acc = fill(0, 1)
  rho_mh  = fill(rho_tune, 1)
  rho_candidate = Normal(0, 1)

  # storage
  beta_keepers  = fill(0.0, iters, p)
  xi_keepers    = fill(0.0, iters)
  a_keepers     = fill(0.0, iters, nknots, nt)
  alpha_keepers = fill(0.0, iters)
  rho_keepers   = fill(0.0, iters)

  for iter = 1:iters
    for ttt = 1:thin
      updatebeta!(y, theta_star, alpha, z, can_z, z_star, can_z_star,
                  beta, beta_m, beta_s, xi, x, x_beta, can_x_beta,
                  cur_lly, can_lly, beta_att, beta_acc, beta_mh, thresh,
                  beta_candidate)
      updatemh!(beta_att, beta_acc, beta_mh, beta_attempts)

      updatexi!(y, theta_star, alpha, z, can_z, z_star, can_z_star,
                x_beta, can_x_beta, xi, cur_lly, can_lly,
                xi_att, xi_acc, xi_mh, thresh, xi_candidate)
      updatemh!(xi_att, xi_acc, xi_mh, xi_attempts)

      updatea!(y, theta_star, a, alpha, cur_lly, can_lly, cur_llps, can_llps,
               z_star, w_star, mid_points, bin_width, a_mh, cuts)

      updatealpha!(y, theta_star, can_theta_star, a, alpha,
                   cur_lly, can_lly, cur_llps, can_llps, z, z_star,
                   w, w_star, mid_points, bin_width, alpha_att, alpha_acc,
                   alpha_mh, alpha_candidate)
      updatemh!(alpha_att, alpha_acc, alpha_mh, alpha_attempts)

      updaterho!(y, theta_star, a, alpha, cur_lly, can_lly, z_star, w, w_star,
                 dw2, rho, rho_upper, rho_att, rho_acc, rho_mh, rho_candidate)
      updatemh!(rho_att, rho_acc, rho_mh, rho_attempts)
    end

    beta_keepers[iter, :] = beta
    xi_keepers[iter]      = xi[1]
    a_keepers[iter, :, :] = a
    alpha_keepers[iter]   = alpha[1]
    rho_keepers[iter]     = rho[1]

    if iter % update == 0
      print("  Iter: $iter \n")
    end
  end
end
