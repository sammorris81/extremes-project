module AuxFunctions
export stdW!,
       makeW,
       logLikeY, logLikeY!,
       getZ, updateZ!,
       getThetaStar,
       rPS,
       rRareBinarySpat

# Arguments:
#   w(ns x nknots): kernel weights
# Returns:
#   std_w(ns x nknots): kernel weights standardized to add to 1
function stdW!(w::Array{FloatingPoint, 2})
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
function makeW(dw2::Array{FloatingPoint, 2}, rho::FloatingPoint)
  ns = size(dw2)[1]
  nknots = size(dw2)[2]

  w = Array(FloatingPoint, ns, nknots)
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
function logLikeY(y::Array{Integer, 2}, theta_star::Array{FloatingPoint, 2},
                  alpha::FloatingPoint, z::Array{FloatingPoint, 2})
  ns = size(y)[1]
  nt = size(y)[2]

  ll_y = Array(FloatingPoint, ns, nt)
  for j = 1:nt, i = 1:ns
    z_star = -theta_star[i, j] / (z[i, j]^(1 / alpha))
    ll_y[i, j] = (1 - y[i, j]) * z_star +
                 y[i, j] * log(1 - exp(z_star))
  end
  return ll_y
end

function logLikeY!(ll::Array{FloatingPoint, 2}, y::Array{Integer, 2},
                   theta_star::Array{FloatingPoint, 2}, alpha::FloatingPoint,
                   z::Array{FloatingPoint, 2})
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
function getThetaStar(w::Array{FloatingPoint, 2}, a::Array{FloatingPoint, 2},
                      alpha::FloatingPoint)
  ns = size(w)[1]
  nknots = size(w)[2]
  nt = size(a)[2]

  w_star = Array(FloatingPoint, ns, nknots)
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
function getZ(xi::FloatingPoint, x_beta::Array{FloatingPoint, 2})
  ns = size(x_beta)[1]
  nt = size(x_beta)[2]

  z = Array(FloatingPoint, ns, nt)
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

function updateZ!(z::Array{FloatingPoint, 2}, xi::FloatingPoint,
                  x_beta::Array{FloatingPoint, 2})
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
function rPS(n::Integer, alpha::FloatingPoint)
  unif = rand(n) * pi
  stdexp = rand(Distributions.Exponential(1), n)

  psrv = Array(FloatingPoint, n)
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
function rRareBinarySpat(x::Array{FloatingPoint, 3}, s::Array{FloatingPoint, 2},
                         knots::Array{FloatingPoint, 2},
                         beta::Array{FloatingPoint, 1},
                         xi::FloatingPoint, alpha::FloatingPoint,
                         rho::FloatingPoint, thresh::FloatingPoint)
  ns = size(x)[1]
  nt = size(x)[2]
  np = size(x)[3]
  nknots = size(knots)[1]

  x_beta = Array(FloatingPoint, ns, nt)
  for t = 1:nt
    x_beta[:, t] = reshape(x[:, t, :], ns, np) * beta
  end

  # get weights
  dw2 = Array(FloatingPoint, ns, nknots)
  for j = 1:nknots, i = 1:ns
    dw2[i, j] = sqeuclidean(vec(s[i, :]), vec(knots[j, :]))
  end

  w = makeW(dw2, rho)
  stdW!(w)

  # get random effects and theta_star
  a = reshape(rPS(nknots * nt, alpha), nknots, nt)
  theta_star = getThetaStar(w, a, alpha)


  # get underlying latent variable u ~ GEV(1, alpha, alpha)
  y = Array(Integer, ns, nt)
  for j = 1:nt, i = 1:ns
    u = rand(Frechet(), 1)[1]^alpha
    z = u * theta_star[i, j]^alpha
    h = x_beta[i, j] + (z^xi - 1) / xi
    y[i, j] = h > thresh ? 1 : 0
  end

  return y
end

function mhUpdate(acc::Integer, att::Integer, mh::FloatingPoint,
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