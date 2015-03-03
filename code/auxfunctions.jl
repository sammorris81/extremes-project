module AuxFunctions
export logLikeY, logLikeY!, updateZ!, getThetaStar, makeW, stdW, getZ

# Arguments:
#   w(ns x nknots): kernel weights
# Returns:
#   std_w(ns x nknots): kernel weights standardized to add to 1
function stdW(w::Array{Float64})
  ns = size(w)[1]::Int64
  nknots = size(w)[2]::Int64

  total_weight = sum(w, 2)
  std_w = Array(Float64, ns, nknots)
  for j=1:nknots
    for i=1:ns
      std_w[i, j] = w[i, j] / total_weight[i]
    end
  end

  return std_w
end

# Arguments:
#   y(ns x nt): data matrix (0s and 1s)
#   theta_star(ns x nt): ∑ Aₗ * wₗ^(1/alpha)
#   alpha(1): spatial dependence (0=dependent, 1=independent)
#   z(ns x nt): latent variable with to unit Fréchet margins
# Return:
#   ll_y(ns x nt): loglikelihood matrix
function logLikeY(y::Array{Int64}, theta_star::Array{Float64},
                  alpha::Float64, z::Array{Float64})
  ns = size(y)[1]::Int64
  nt = size(y)[2]::Int64

  ll_y = Array(Float64, ns, nt)
  for j = 1:nt, i = 1:ns
    z_star = -theta_star[i, j] / (z[i, j]^(1 / alpha))
    ll_y[i, j] = (1 - y[i, j]) * z_star +
                 y[i, j] * log(1 - exp(z_star))
  end
  return ll_y
end

function logLikeY!(y::Array{Int64}, theta_star::Array{Float64},
                  alpha::Float64, z::Array{Float64}, ll::Array{Float64})
  ns = size(y)[1]::Int64
  nt = size(y)[2]::Int64
  z_star = 0.0

  for j = 1:nt, i = 1:ns
    z_star = -theta_star[i, j] / (z[i, j]^(1 / alpha))
    ll[i, j] = (1 - y[i, j]) * z_star +
                 y[i, j] * log(1 - exp(z_star))
  end
end

# Arguments:
#   dw2(ns x nknots): distance matrix
#   rho(1): bandwidth parameter
# Return:
#   w(ns x nknots): kernel weights
function makeW(dw2::Array{Float64}, rho::Float64)
  ns = size(dw2)[1]::Int64
  nknots = size(dw2)[2]::Int64

  w = Array(Float64, ns, nknots)
  for j = 1:nknots, i = 1:ns
    w[i, j] = exp(-0.5 * dw2[i, j] / (rho^2))
  end

  return w
end

# Arguments:
#   w(ns x nknots): kernel weights
#   a(nknots x nt): PS(alpha) random effect
#   alpha(1): spatial dependence (0=dependent, 1=independent)
# Return:
#   theta_star(ns x nt): ∑ Aₗ * wₗ^(1/alpha)
function getThetaStar(w::Array{Float64}, a::Array{Float64}, alpha::Float64)
  ns = size(w)[1]::Int64
  nknots = size(w)[2]::Int64
  nt = size(a)[2]::Int64

  w_star = Array(Float64, ns, nt)
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
function updateZ!(xi::Float64, x_beta::Array{Float64}, z::Array{Float64})
  ns = size(x_beta)[1]::Int64
  nt = size(x_beta)[2]::Int64

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

function getZ(xi::Float64, x_beta::Array{Float64})
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

# include("logLikeY.jl")
# include("getZ.jl")
# include("getThetaStar.jl")
# include("makeW.jl")
# include("stdW.jl")

end
