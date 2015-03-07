module UpdateModel
export update_beta!, sample_beta! # sampleBeta, updateBeta

using Distributions
using AuxFunctions


# TODO: Rewrite to loop over parameter updates
function update_beta!(y::Array{Int64, 2}, theta_star::Array{Float64, 2},
                     alpha::Float64, z::Array{Float64, 2},
                     can_z::Array{Float64, 2},
                     beta::Array{Float64, 1},
                     can_beta::Array{Float64, 1},
                     beta_m::Array{Float64, 1},
                     beta_s::Array{Float64, 1},
                     xi::Float64, x::Array{Float64, 3},
                     x_beta::Array{Float64, 2},
                     can_x_beta::Array{Float64, 2},
                     cur_lly::Array{Float64, 2},
                     can_lly::Array{Float64, 2},
                     acc::Array{Int64, 1}, att::Array{Int64, 1},
                     mh::Array{Float64, 1},
                     candidate::Distribution)
  np = size(x)[3]
  ns = size(y)[1]
  nt = size(y)[2]

  # get candidate draws
  for p = 1:np
    att[p] += 1
    can_beta = rand(Normal(beta[p], mh[p]), 1)[1]

    # faster than reshape because it doesn't need to make a copy
    for t = 1:nt, i = 1:ns
      can_x_beta[i, t] += x_beta[i, t] + x[i, t, p] * (can_beta - beta[p])
    end

    update_z!(can_z, xi, can_x_beta)
    logpdf_rarebinary!(can_lly, y, theta_star, alpha, can_z)

    prior = Distributions.Normal(beta_m[p], beta_s[p])
    R = sum(can_lly - cur_lly) +
        logpdf(prior, can_beta) - logpdf(prior, beta[p])

    if (length(R) > 1)
      error("R has length greater than 1")
    end

    if (log(rand(1))[1] < R)  # rand generates a vector
      beta[p] = can_beta
      x_beta = can_x_beta
      z = can_z
      cur_lly = can_lly
      acc[p] += 1
    end
  end
end

# for testing timing of update_beta!
function sample_beta!(nreps::Int64, y::Array{Int64, 2},
                      theta_star::Array{Float64, 2}, alpha::Float64,
                      z::Array{Float64, 2}, beta::Array{Float64, 1},
                      beta_m::Array{Float64, 1},
                      beta_s::Array{Float64, 1}, xi::Float64,
                      x::Array{Float64, 3},
                      x_beta::Array{Float64, 2},
                      cur_lly::Array{Float64, 2},
                      acc_beta::Array{Int64, 1}, att_beta::Array{Int64, 1},
                      mh_beta::Array{Float64, 1})
  np = size(x)[3]
  ns = size(y)[1]
  nt = size(y)[2]
  beta_keep = fill(0.0, nreps, np)
  beta_can_dis = Normal(0, 1)
  can_beta = Array(Float64, np)
  # println(pointer(can_beta))
  can_x_beta = Array(Float64, ns, nt)
  # println(pointer(can_x_beta))
  can_z = Array(Float64, ns, nt)
  # println(pointer(can_z))
  can_lly = Array(Float64, ns, nt)
  # println(pointer(can_lly))
  for i = 1:nreps
    beta_update = update_beta!(y, theta_star, alpha, z, can_z,
                              beta, can_beta, beta_m, beta_s, xi,
                              x, x_beta, can_x_beta, cur_lly, can_lly,
                              acc_beta, att_beta, mh_beta,
                              beta_can_dis)
    beta_keep[i, :] = beta
    if i % 500 == 0
      println(i)
    end
    # println(pointer(can_beta))
    # println(pointer(can_x_beta))
    # println(pointer(can_z))
    # println(pointer(can_lly))
  end
  return beta_keep
end

# function updateBeta(y::Array{Int64, 2}, theta_star::Array{Float64, 2},
#                      alpha::Float64, z::Array{Float64, 2},
#                      beta::Array{Float64, 1},
#                      beta_m::Array{Float64, 1},
#                      beta_s::Array{Float64, 1}, xi::Float64,
#                      x::Array{Float64, 3},
#                      x_beta::Array{Float64, 2},
#                      cur_lly::Array{Float64, 2},
#                      acc::Int64, att::Int64, mh::Array{Float64, 1})
#   np = size(x)[3]
#   ns = size(y)[1]
#   nt = size(y)[2]
#   att += 1

#   # get candidate draws
#   can_beta = Array(Float64, 3)
#   for i = 1:np
#     can_beta[i] = rand(Distributions.Normal(beta[i], mh[i]))
#   end

#   can_x_beta = Array(Float64, ns, nt)
#   for t = 1:nt
#     can_x_beta[:, t] = reshape(x[:, t, :], ns, 3) * can_beta
#   end

#   can_z = getZ(xi, can_x_beta)
#   can_lly = logLikeY(y, theta_star, alpha, can_z)

#   R = sum(can_lly - cur_lly)
#   for i = 1:np
#     prior = Distributions.Normal(beta_m[i], beta_s[i])
#     R += logpdf(prior, can_beta[i]) - logpdf(prior, beta[i])
#   end

#   if (length(R) > 1)
#     error("R has length greater than 1")
#   end

#   if (log(rand(1))[1] < R)  # rand generates a vector
#     beta = can_beta
#     x_beta = can_x_beta
#     z = can_z
#     cur_lly = can_lly
#     acc += 1
#   end

#   results = {"beta" => beta, "x_beta" => x_beta, "z" => z,
#              "cur_lly" => cur_lly, "att" => att, "acc" => acc}

#   return results
# end

# function sampleBeta(nreps::Int64, y::Array{Int64, 2},
#                     theta_star::Array{Float64, 2}, alpha::Float64,
#                     z::Array{Float64, 2}, beta::Array{Float64, 1},
#                     beta_m::Array{Float64, 1},
#                     beta_s::Array{Float64, 1}, xi::Float64,
#                     x::Array{Float64, 3}, x_beta::Array{Float64, 2},
#                     cur_lly::Array{Float64, 2},
#                     acc_beta::Array{Int64, 1}, att_beta::Array{Int64, 1},
#                     mh_beta::Array{Float64, 1})
#   np = size(x)[3]
#   beta_keep = fill(0.0, nreps, np)
#   for i = 1:nreps
#     beta_update = updateBeta(y, theta_star, alpha, z, beta, beta_m, beta_s, xi,
#                              x, x_beta, cur_lly, acc_beta, att_beta, mh_beta)
#     beta = beta_update["beta"]
#     x_beta = beta_update["x_beta"]
#     z = beta_update["z"]
#     cur_lly = beta_update["cur_lly"]
#     att_beta = beta_update["att"]
#     acc_beta = beta_update["acc"]

#     beta_keep[i, :] = beta
#     if i % 500 == 0
#       println(i)
#     end
#   end
#   return beta_keep
# end

end