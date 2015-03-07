module AuxFunctions
export std_w!,
       make_w,
       logpdf_rarebinary, logpdf_rarebinary!,
       get_z, update_z!,
       get_thetastar,
       sample_pstab,
       sample_rarebinary,
       mh_update

using Distances
using Distributions

# Arguments:
#   w(ns x nknots): kernel weights
# Returns:
#   std_w(ns x nknots): kernel weights standardized to add to 1
function std_w!(w::Array{Float64, 2})
  ns = size(w)[1]
  nknots = size(w)[2]

  total_weight = sum(w, 2)
  for j=1:nknots
    for i=1:ns
      w[i, j] /= total_weight[i]
    end
  end

end

# Arguments:
#   dw2(ns x nknots): distance matrix
#   rho(1): bandwidth parameter
# Return:
#   w(ns x nknots): kernel weights
function make_w(dw2::Array{Float64, 2}, rho::Float64)
  ns = size(dw2)[1]
  nknots = size(dw2)[2]

  w = Array(Float64, ns, nknots)
  for j = 1:nknots, i = 1:ns
    w[i, j] = exp(-0.5 * dw2[i, j] / (rho^2))
  end

  return w
end


# Arguments:
#   y(ns x nt): data matrix (0s and 1s)
#   theta_star(ns x nt): ∑ Aₗ * wₗ^(1/alpha)
#   alpha(1): spatial dependence (0=dependent, 1=independent)
#   z(ns x nt): latent variable with to unit Fréchet margins
# Return:
#   ll_y(ns x nt): loglikelihood matrix
function logpdf_rarebinary(y::Array{Int64, 2}, theta_star::Array{Float64, 2},
                           alpha::Float64, z::Array{Float64, 2})
  ns = size(y)[1]
  nt = size(y)[2]

  ll_y = Array(Float64, ns, nt)
  for j = 1:nt, i = 1:ns
    z_star = -theta_star[i, j] / (z[i, j]^(1 / alpha))
    ll_y[i, j] = (1 - y[i, j]) * z_star +
                 y[i, j] * log(1 - exp(z_star))
  end
  return ll_y
end

function logpdf_rarebinary!(ll::Array{Float64, 2}, y::Array{Int64, 2},
                            theta_star::Array{Float64, 2}, alpha::Float64,
                            z::Array{Float64, 2})
  ns = size(y)[1]
  nt = size(y)[2]

  for j = 1:nt, i = 1:ns
    z_star = -theta_star[i, j] / (z[i, j]^(1 / alpha))
    ll[i, j] = (1 - y[i, j]) * z_star +
                 y[i, j] * log(1 - exp(z_star))
  end
end

# Arguments:
#   w(ns x nknots): kernel weights
#   a(nknots x nt): PS(alpha) random effect
#   alpha(1): spatial dependence (0=dependent, 1=independent)
# Return:
#   theta_star(ns x nt): ∑ Aₗ * wₗ^(1/alpha)
function get_thetastar(w::Array{Float64, 2}, a::Array{Float64, 2},
                       alpha::Float64)
  ns = size(w)[1]
  nknots = size(w)[2]
  nt = size(a)[2]

  w_star = Array(Float64, ns, nknots)
  for j = 1:nknots, i = 1:ns
    w_star[i, j] = w[i, j]^(1 / alpha)
  end
  theta_star = w_star * a

  return theta_star
end

# Arguments:
#   xi(1): shape parameter
#   x_beta(ns x nt): mean function
# Return:
#   z(ns x nt): latent variable with to unit Fréchet margins
function get_z(xi::Float64, x_beta::Array{Float64, 2})
  ns = size(x_beta)[1]
  nt = size(x_beta)[2]

  z = Array(Float64, ns, nt)
  if xi != 0
    for j = 1:nt, i = 1:ns
      z[i, j] = (1 + xi * x_beta[i, j])^(1 / xi)
    end
  else
    for j = 1:nt, i = 1:ns
      z[i, j] = exp(x_beta[i, j])
    end
  end

  return z
end

function update_z!(z::Array{Float64, 2}, xi::Float64,
                  x_beta::Array{Float64, 2})
  ns = size(x_beta)[1]
  nt = size(x_beta)[2]

  if xi != 0
    for j = 1:nt, i = 1:ns
      z[i, j] = (1 + xi * x_beta[i, j])^(1 / xi)
    end
  else
    for j = 1:nt, i = 1:ns
      z[i, j] = exp(x_beta[i, j])
    end
  end

end


# generating PS(alpha) from Stephenson(2003)
# return psrv(n): vector of PS(alpha) random variables
function sample_pstab(n::Int64, alpha::Float64)
  unif = rand(n) * pi
  stdexp = rand(Distributions.Exponential(1), n)

  psrv = Array(Float64, n)
  for i = 1:n
    log_a = (1 - alpha) / alpha * log(sin((1 - alpha) * unif[i])) +
           log(sin(alpha * unif[i])) - (1 - alpha) / alpha * log(stdexp[i]) -
           1 / alpha * log(sin(unif[i]))
    psrv[i] = exp(log_a)
  end

  return psrv
end

# generate dependent rare binary data
# returns y(ns x nt): matrix of observations
function sample_rarebinary(x::Array{Float64, 3}, s::Array{Float64, 2},
                           knots::Array{Float64, 2},
                           beta::Array{Float64, 1},
                           xi::Float64, alpha::Float64,
                           rho::Float64, thresh::Float64)
  ns = size(x)[1]
  nt = size(x)[2]
  np = size(x)[3]
  nknots = size(knots)[1]
  y = Array(Int64, ns, nt)

  x_beta = Array(Float64, ns, nt)
  for t = 1:nt
    x_beta[:, t] = reshape(x[:, t, :], ns, np) * beta
  end

  # get weights
  dw2 = Array(Float64, ns, nknots)
  for j = 1:nknots, i = 1:ns
    dw2[i, j] = Distances.sqeuclidean(vec(s[i, :]), vec(knots[j, :]))
  end

  w = make_w(dw2, rho)
  std_w!(w)

  # get GEV(1, 1, 1) marginally
  # z = u * theta is marginally GEV(1, 1, 1) when
  #   u ~ GEV(1, alpha, alpha)
  #   theta = (sum A_l w_l^(1/alpha))^alpha
  #   A_l ~ PS(alpha)
  if alpha < 1
    a = reshape(sample_pstab(nknots * nt, alpha), nknots, nt)
    theta_star = get_thetastar(w, a, alpha)
    for j = 1:nt, i = 1:ns
      u = rand(Frechet(), 1)[1]^alpha
      z = u * theta_star[i, j]^alpha
      h = x_beta[i, j] + (z^xi - 1) / xi
      y[i, j] = h > thresh ? 1 : 0
    end
  else
    for j = 1:nt, i = 1:ns
      z = rand(Frechet(), 1)
      h = x_beta[i, j] + (z^xi - 1) / xi
      y[i, j] = h > thresh ? 1 : 0
    end
  end

  return y
end


function mh_update(acc::Int64, att::Int64, mh::Float64,
                  nattempts=50, lower=0.8, upper=1.2)
  acc_rate = acc / att
  if att > nattempts
    if acc_rate < 0.25
      mh *= lower
    elseif acc_rate > 0.50
      mh *= upper
    end
    acc = 0
    att = 0
  end

  return (acc, att, mh)
end

end