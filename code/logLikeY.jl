# Arguments:
#   y(ns x nt): data matrix (0s and 1s)
#   theta_star(ns x nt): ∑ Aₗ * wₗ^(1/alpha)
#   alpha(1): spatial dependence (0=dependent, 1=independent)
#   z(ns x nt): latent variable with to unit Fréchet margins
# Return:
#   ll_y(ns x nt): loglikelihood matrix
function logLikeY(y::Array{Int64}, theta_star::Array{Float64},
                  alpha::Float64, z::Array{Float64})
  ns = size(y)[1]
  nt = size(y)[2]

  ll_y = fill(0.0, ns, nt)
  for j = 1:nt
    for i = 1:ns
      z_star = -theta_star[i, j] / (z[i, j]^(1 / alpha))
      ll_y[i, j] = (1 - y[i, j]) * z_star +
                   y[i, j] * log(1 - exp(z_star))
    end
  end
  return ll_y
end

# # Arguments:
# #   y(ns x nt): data matrix (0s and 1s)
# #   theta_star(ns x nt): ∑ Aₗ * wₗ^(1/alpha)
# #   alpha(1): spatial dependence (0=dependent, 1=independent)
# #   z(ns x nt): latent variable with to unit Fréchet margins
# # Return:
# #   ll_y(ns x nt): loglikelihood matrix
# function logLikeY2(y::Array{Int64}, theta_star::Array{Float64},
#                   alpha::Float64, z::Array{Float64})
#   ns = size(y)[1]
#   nt = size(y)[2]

#   ll_y = fill(0.0, ns, nt)
#   z_star = -theta_star ./ (z.^(1 / alpha))
#   for j = 1:nt
#     for i = 1:ns
#       ll_y[i, j] = (1 - y[i, j]) * z_star[i, j] +
#                    y[i, j] * log(1 - exp(z_star[i, j]))
#     end
#   end
#   return ll_y
# end

# ns = 1000
# nt = 365
# y = fill(1, ns, nt)
# theta_star = fill(1.0, ns, nt)
# z = fill(1.0, ns, nt)
# for j = 1:nt
#   for i = 1:ns
#     y[i, j] = rand(Binomial(1, 0.1))
#     z[i, j] = rand(Chisq(2))
#     theta_star[i, j] = rand(Chisq(3))
#   end
# end

# alpha = 0.5

# @elapsed logLikeY(y, theta_star, alpha, z)
# @elapsed logLikeY2(y, theta_star, alpha, z)

