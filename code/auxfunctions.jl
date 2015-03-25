module AuxFunctions
export std_w!,
       make_w,
       logpdf_rarebinary, logpdf_rarebinary!,
       logpdf_ps!,
       get_z, update_z!,
       get_thetastar,
       sample_pstab,
       sample_rarebinary,
       get_canxbeta!,
       mh_update

using Distances
using Distributions

# Arguments:
#   w(ns x nknots): kernel weights
# Returns:
#   std_w(ns x nknots): kernel weights standardized to add to 1
function std_w!(w::Array{Float64, 2})
  const ns = size(w)[1]
  const nknots = size(w)[2]

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
  const ns = size(dw2)[1]
  const nknots = size(dw2)[2]

  w = Array(Float64, ns, nknots)
  for j = 1:nknots, i = 1:ns
    w[i, j] = exp(-0.5 * dw2[i, j] / (rho^2))
  end

  return w
end

function get_canxbeta!(can_x_beta::Array{Float64, 2}, x_beta::Array{Float64, 2},
                       x::Array{Float64, 2}, can_diff::Float64)
  const ns = size(x_beta)[1]
  const nt = size(x_beta)[2]

  for t = 1:nt, i = 1:ns
    can_x_beta[i, t] = x_beta[i, t] + can_diff * x[i, t]
  end
end

function get_canxbeta!(can_x_beta::Array{Float64, 2}, x_beta::Array{Float64, 2},
                       x::Array{Float64, 2}, can_diff::Float64,
                       ns::Int64, nt::Int64)

  for t = 1:nt, i = 1:ns
    can_x_beta[i, t] = x_beta[i, t] + can_diff * x[i, t]
  end
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
  const ns = size(y)[1]
  const nt = size(y)[2]

  ll = Array(Float64, ns, nt)
  for j = 1:nt, i = 1:ns
    z_star = -theta_star[i, j] / (z[i, j]^(1 / alpha))
    if y[i, j] == 0
      ll[i, j] = z_star
    else
      ll[i, j] = log(1 - exp(z_star))
    end
  end

  return ll
end

function logpdf_rarebinary!(ll::Array{Float64, 2}, y::Array{Int64, 2},
                            theta_star::Array{Float64, 2}, alpha::Float64,
                            z::Array{Float64, 2})
  ns = size(y)[1]
  nt = size(y)[2]

  for i = 1:nt * ns
    z_star = -theta_star[i] / (z[i]^(1 / alpha))
    if y[i] == 0
      ll[i] = z_star
    else
      ll[i] = log(1 - exp(z_star))
    end
  end
end

function logpdf_rarebinary!(ll::Array{Float64, 2}, y::Array{Int64, 2},
                            theta_star::Array{Float64, 2}, alpha::Float64,
                            z::Array{Float64, 2}, ns::Int64, nt::Int64)

  for i = 1:nt * ns
    z_star = -theta_star[i] / (z[i]^(1 / alpha))
    if y[i] == 0
      ll[i] = z_star
    else
      ll[i] = log(1 - exp(z_star))
    end
  end
end

function logpdf_ps!(ll::Array{Float64, 2}, a::Array{Float64, 2}, alpha::Float64,
                    mid_points::Array{Float64, 1}, bin_width::Array{Float64, 1})
  const nt = size(ll)[2]
  const nknots = size(ll)[1]

  # fill!(ll, -Inf)
  for t = 1:nt, k = 1:nknots
    ll[k, t] = log(alpha) - log(1 - alpha) - (1 / (1 - alpha)) * log(a[k, t]) +
               ld(a[k, t], alpha, mid_points, bin_width)
  end
end

function logpdf_ps(a::Float64, alpha::Float64,
                   mid_points::Array{Float64, 1}, bin_width::Array{Float64, 1})

  # fill!(ll, -Inf)
  ll = log(alpha) - log(1 - alpha) - (1 / (1 - alpha)) * log(a) +
               ld(a, alpha, mid_points, bin_width)
  return ll
end

# integral part of density
function ld(a::Float64, alpha::Float64, mid_points::Array{Float64, 1},
             bin_width::Array{Float64, 1})
  nbins = size(mid_points)[1]
  integral = 0.0
  for i = 1:nbins
    psi = pi * mid_points[i]
    c = (sin(alpha * psi) / sin(psi))^(1 / (1 - alpha)) *
        sin((1 - alpha) * psi) / sin(alpha * psi)
    logint = log(c) - c * (1 / a^(alpha  / (1 - alpha)))
    integral += exp(logint) * bin_width[i]
  end

  return log(integral)
end

# Arguments:
#   w(ns x nknots): kernel weights
#   a(nknots x nt): PS(alpha) random effect
#   alpha(1): spatial dependence (0=dependent, 1=independent)
# Return:
#   theta_star(ns x nt): ∑ Aₗ * wₗ^(1/alpha)
function get_thetastar(w::Array{Float64, 2}, a::Array{Float64, 2},
                       alpha::Float64)
  const ns = size(w)[1]
  const nknots = size(w)[2]

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
function get_z(xi::Float64, x_beta::Array{Float64, 2}, thresh::Float64)
  const ns = size(x_beta)[1]
  const nt = size(x_beta)[2]

  z = Array(Float64, ns, nt)
  if xi != 0
    for i = 1:nt * ns
      z[i] = (1 + xi * (thresh - x_beta[i]))^(1 / xi)
    end
  else
    for i = 1:nt * ns
      z[i] = exp(thresh - x_beta[i])
    end
  end

  return z
end

function update_z!(z::Array{Float64, 2}, xi::Float64,
                  x_beta::Array{Float64, 2}, thresh::Float64)
  const ns = size(x_beta)[1]
  const nt = size(x_beta)[2]

  # the loop here is way faster than vectorized code and allocates far
  # less memory
  if xi != 0
    for i = 1:nt * ns
      z[i] = (1 + xi * (thresh - x_beta[i]))^(1 / xi)
    end
  else
    for i = 1:nt * ns
      z[i] = exp(thresh - x_beta[i])
    end
  end

end


function update_z!(z::Array{Float64, 2}, xi::Float64,
                  x_beta::Array{Float64, 2}, thresh::Float64,
                  ns::Int64, nt::Int64)

  # the loop here is way faster than vectorized code and allocates far
  # less memory
  if xi != 0
    for i = 1:nt * ns
      z[i] = (1 + xi * (thresh - x_beta[i]))^(1 / xi)
    end
  else
    for i = 1:nt * ns
      z[i] = exp(thresh - x_beta[i])
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
  const ns = size(x)[1]
  const nt = size(x)[2]
  const np = size(x)[3]
  const nknots = size(knots)[1]

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
    return (y, a)
  else
    for j = 1:nt, i = 1:ns
      z = rand(Frechet(), 1)[1]
      h = x_beta[i, j] + (z^xi - 1) / xi
      y[i, j] = h > thresh ? 1 : 0
    end
    return y
  end

end


function mh_update(att::Int64, acc::Int64, mh::Float64,
                  nattempts::Int64, lower::Float64, upper::Float64)
  # acc_rate = acc / att
  if att > nattempts
    if acc / att < 0.25
      mh *= lower
    elseif acc / att > 0.50
      mh *= upper
    end
    acc = 0
    att = 0
  end

  return (acc, att, mh)
end

function get_level(a::Float64, cuts::Array{Float64, 1})
  lev = 1
  ncuts = size(cuts)[1]
  for i = 1:ncuts
    lev = a > cuts[i] ? i + 1 : lev
  end

  return lev
end

end