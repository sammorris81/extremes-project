module AuxFunctions
export transunif, transprobit,
       std_w!, make_w,
       updatecanxbeta!,
       updatez, updatez!,
       updatethetastar, updatethetastar!,
       updatemh!,
       logpdf_rarebinary, logpdf_rarebinary!,
       logpdf_ps, logpdf_ps!,
       sample_pstab,
       sample_rarebinary

importall Distances
importall Distributions
importall Base.LinAlg.BLAS

function transunif(x::Real, support::Vector, inv::Bool=false)
  @assert isfinite(support[1])
  @assert isfinite(support[2])
  if !inv
    return (x - support[1]) / (support[2] - support[1])
  else
    @assert x >= 0 && x <= 1
    return x * (support[2] - support[1]) + support[1]
  end
end

function transprobit(x::Real, support::Vector, inv::Bool=false)
  @assert isfinite(support[1])
  @assert isfinite(support[2])
  if !inv
    p = transunif(x, support, inv)
    return quantile(Distributions.Normal(0, 1), p)
  else
    p = cdf(Distributions.Normal(0, 1), x)
    return transunif(p, support, inv)
  end
end

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
    w[i, j] = exp(-0.5 * dw2[i, j] / (rho[1]^2))
  end

  return w
end

function make_w(dw2::Array{Float64, 2}, rho::Array{Float64, 1})
  w = make_w(dw2, rho[1])
  return w
end

function updatecanxbeta!(can_x_beta::Array{Float64, 2}, x_beta::Array{Float64, 2},
                       x::Array{Float64, 2}, can_diff::Float64,
                       ns::Int64, nt::Int64)

  for j = 1:nt, i = 1:ns
    can_x_beta[i, j] = x_beta[i, j] + can_diff * x[i, j]
  end
end

function updatecanxbeta!(can_x_beta::Array{Float64, 2}, x_beta::Array{Float64, 2},
                       x::Array{Float64, 2}, can_diff::Float64)
  ns = size(x_beta)[1]
  nt = size(x_beta)[2]

  updatecanxbeta!(can_x_beta, x_beta, x, can_diff, ns, nt)
end

# Arguments:
#   xi(1): shape parameter
#   x_beta(ns x nt): mean function
# Return:
#   z(ns x nt): latent variable with to unit Fréchet margins
function updatez!(z::Array{Float64, 2}, xi::Array{Float64, 1},
                  x_beta::Array{Float64, 2}, thresh::Float64,
                  ns::Int64, nt::Int64)

  # the loop here is way faster than vectorized code and allocates far
  # less memory
  if xi[1] != 0
    for j = 1:nt, i = 1:ns
      z[i, j] = (1 + xi[1] * (thresh - x_beta[i, j]))^(1 / xi[1])
    end
  else
    for j = 1:nt, i = 1:ns
      z[i, j] = exp(thresh - x_beta[i, j])
    end
  end

end

function updatez!(z::Array{Float64, 2}, xi::Array{Float64, 1},
                  x_beta::Array{Float64, 2}, thresh::Float64)
  ns = size(x_beta)[1]
  nt = size(x_beta)[2]

  update_z!(z, xi, x_beta, thresh, ns, nt)
end

function updatez(xi::Array{Float64, 1}, x_beta::Array{Float64, 2}, thresh::Float64)
  ns = size(x_beta)[1]
  nt = size(x_beta)[2]

  z = fill(0.0, ns, nt)
  updatez!(z, xi, x_beta, thresh, ns, nt)

  return z
end

# Arguments:
#   w(ns x nknots): kernel weights
#   a(nknots x nt): PS(alpha) random effect
#   alpha(1): spatial dependence (0=dependent, 1=independent)
# Return:
#   theta_star(ns x nt): ∑ Aₗ * wₗ^(1/alpha)
function updatethetastar!(theta_star::Array{Float64, 2}, w_star::Array{Float64, 2},
                          a::Array{Float64, 2})
  gemm!('N', 'N', 1.0, w_star, a, 0.0, theta_star)
end

function updatethetastar(w_star::Array{Float64, 2}, a::Array{Float64, 2})

  theta_star = fill(0.0, size(w_star)[1], size(a)[2])
  updatethetastar!(theta_star, w_star, a)

  return theta_star
end

function updatemh!(att::Array{Int64, 1}, acc::Array{Int64, 1},
                    mh::Array{Float64, 1},
                    nattempts::Int64, lower::Float64=0.8, upper::Float64=1.2)
  # acc_rate = acc / att
  nupdates = size(att)[1]
  for i = 1:nupdates
    if att[i] > nattempts
      if acc[i] / att[i] < 0.25
        mh[i] *= lower
      elseif acc[i] / att[i] > 0.50
        mh[i] *= upper
      end
      acc[i] = 0
      att[i] = 0
    end
  end
end

# Arguments:
#   y(ns x nt): data matrix (0s and 1s)
#   theta_star(ns x nt): ∑ Aₗ * wₗ^(1/alpha)
#   alpha(1): spatial dependence (0=dependent, 1=independent)
#   z(ns x nt): latent variable with to unit Fréchet margins
# Return:
#   ll_y(ns x nt): loglikelihood matrix
function logpdf_rarebinary!(ll::Array{Float64, 2}, y::Array{Int64, 2},
                            theta_star::Array{Float64, 2},
                            z_star::Array{Float64, 2}, ns::Int64, nt::Int64)

  for j = 1:nt, i = 1:ns
    theta_z_star = -theta_star[i, j] / z_star[i, j]
    if y[i, j] == 0
      ll[i, j] = theta_z_star
    else
      ll[i, j] = log(1 - exp(theta_z_star))
    end
  end
end

function logpdf_rarebinary!(ll::Array{Float64, 2}, y::Array{Int64, 2},
                            theta_star::Array{Float64, 2},
                            z_star::Array{Float64, 2})
  ns, nt = size(y)

  logpdf_rarebinary!(ll, y, theta_star, z_star, ns, nt)
end

function logpdf_rarebinary!(ll::Array{Float64, 1}, y::Array{Int64, 1},
                            theta_star::Array{Float64, 1},
                            z_star::Array{Float64, 1})
  ns = size(y)[1]
  nt = 1

  for i = 1:ns
    theta_z_star = -theta_star[i] ./ z_star[i]
    if y[i] == 0
      ll[i] = theta_z_star
    else
      ll[i] = log(1 - exp(theta_z_star))
    end
  end

end

function logpdf_rarebinary(y::Array{Int64, 2}, theta_star::Array{Float64, 2},
                           z_star::Array{Float64, 2})
  ll = fill(0.0, ns, nt)

  for j = 1:nt, i = 1:ns
    theta_z_star = -theta_star[i, j] / z_star[i, j]
    if y[i, j] == 0  # is easier to read than ? operator
      ll[i, j] = theta_z_star
    else
      ll[i, j] = log(1 - exp(theta_z_star))
    end
  end

  return ll
end

function logpdf_ps!(ll::Array{Float64, 2}, a::Array{Float64, 2},
                    alpha::Float64,
                    mid_points::Array{Float64, 1}, bin_width::Array{Float64, 1})
  nt = size(ll)[2]
  nknots = size(ll)[1]

  # fill!(ll, -Inf)
  for t = 1:nt, k = 1:nknots
    ll[k, t] = log(alpha) - log(1 - alpha) - (1 / (1 - alpha)) * log(a[k, t]) +
               ld(a[k, t], alpha, mid_points, bin_width)
  end
end

function logpdf_ps(a::Array{Float64, 2}, alpha::Array{Float64, 1},
                   mid_points::Array{Float64, 1}, bin_width::Array{Float64, 1})
  nknots, nt = size(a)
  ll = fill(0.0, nknots, nt)
  logpdf_ps!(ll, a, alpha[1], mid_points, bin_width)

  return ll
end

# a is a single day
function logpdf_ps(a::Real, alpha::Real,
                   mid_points::Array{Float64, 1}, bin_width::Array{Float64, 1})
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

function sample_pstab(n::Int64, alpha::Array{Float64, 1})
  sample_pstab(n, alpha[1])
end

# generate dependent rare binary data
# returns y(ns x nt): matrix of observations
function sample_rarebinary(x::Array{Float64, 3}, s::Array{Float64, 2},
                           knots::Array{Float64, 2},
                           beta::Array{Float64, 1},
                           xi::Array{Float64, 1}, alpha::Array{Float64, 1},
                           rho::Array{Float64, 1}, thresh::Float64)
  ns, nt, np = size(x)
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

  w = make_w(dw2, rho[1])
  std_w!(w)

  # get GEV(1, 1, 1) marginally
  # z = u * theta is marginally GEV(1, 1, 1) when
  #   u ~ GEV(1, alpha, alpha)
  #   theta = (sum A_l w_l^(1/alpha))^alpha
  #   A_l ~ PS(alpha)
  if alpha[1] < 1
    a = reshape(sample_pstab(nknots * nt, alpha[1]), nknots, nt)
    theta_star = get_thetastar(w, a, alpha[1])
    for j = 1:nt, i = 1:ns
      u = rand(Frechet(), 1)[1]^alpha[1]
      z = u * theta_star[i, j]^alpha[1]
      h = x_beta[i, j] + (z^xi[1] - 1) / xi[1]
      y[i, j] = h > thresh ? 1 : 0
    end
    return (y, a)
  else
    for j = 1:nt, i = 1:ns
      z = rand(Frechet(), 1)[1]
      h = x_beta[i, j] + (z^xi[1] - 1) / xi[1]
      y[i, j] = h > thresh ? 1 : 0
    end
    return y
  end

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
