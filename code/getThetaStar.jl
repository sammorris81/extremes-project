# Arguments:
#   w(ns x nknots): kernel weights
#   a(nknots x nt): PS(alpha) random effect
#   alpha(1): spatial dependence (0=dependent, 1=independent)
# Return:
#   theta_star(ns x nt): ∑ Aₗ * wₗ^(1/alpha)
function getThetaStar(w::Array{Float64}, a::Array{Float64}, alpha::Float64)
  ns = size(w)[1]
  nknots = size(w)[2]
  nt = size(a)[2]

  w_star = fill(0.0, ns, nt)
  for j=1:nknots
    for i=1:ns
      w_star[i, j] = w[i, j]^(1 / alpha)
    end
  end
  theta_star = w_star * a

  return theta_star
end