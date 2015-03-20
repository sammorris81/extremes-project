module UpdateModel
export update_beta!, sample_beta!, update_beta_2!, update_alpha! # sampleBeta, updateBeta

using Distributions
using AuxFunctions


# TODO: Rewrite to loop over parameter updates
function update_beta!(y::Array{Int64, 2}, theta_star::Array{Float64, 2},
                     alpha::Float64, z::Array{Float64, 2},
                     can_z::Array{Float64, 2},
                     beta::Array{Float64, 1},
                     beta_m::Array{Float64, 1},
                     beta_s::Array{Float64, 1},
                     xi::Float64, x::Array{Float64, 3},
                     x_beta::Array{Float64, 2},
                     can_x_beta::Array{Float64, 2},
                     cur_lly::Array{Float64, 2},
                     can_lly::Array{Float64, 2},
                     att::Array{Int64, 1}, acc::Array{Int64, 1},
                     mh::Array{Float64, 1}, thresh::Float64,
                     candidate::Distribution)

  const ns = size(y)[1]
  const nt = size(y)[2]
  const np = size(x)[3]

  # get candidate draws
  for p = 1:np
    att[p] += 1
    can_beta = beta[p] + mh[p] * rand(candidate, 1)[1]

    # faster than reshape because it doesn't need to make a copy
    can_diff = can_beta - beta[p]
    # @time for t = 1:nt, i = 1:ns
    #   can_x_beta[i, t] = x_beta[i, t] + can_diff * x[i, t, p]
    # end
    # get_canxbeta!(can_x_beta, x_beta, slice(x, :, :, p), can_diff, ns, nt)
    # get_canxbeta!(can_x_beta, x_beta, sub(x, :, :, p), can_diff, ns, nt)
    get_canxbeta!(can_x_beta, x_beta, x[:, :, p], can_diff, ns, nt)
    update_z!(can_z, xi, can_x_beta, thresh, ns, nt)
    logpdf_rarebinary!(can_lly, y, theta_star, alpha, can_z, ns, nt)

    prior = Distributions.Normal(beta_m[p], beta_s[p])
    # > 64k memory for summing over can_lly and cur_lly
    R = sum(can_lly - cur_lly) +
        logpdf(prior, can_beta) - logpdf(prior, beta[p])

    if (length(R) > 1)
      error("R has length greater than 1")
    end

    if (log(rand(1)[1]) < R)  # rand generates a vector
      beta[p] = can_beta
      copy!(x_beta, can_x_beta)
      copy!(z, can_z)
      copy!(cur_lly, can_lly)
      acc[p] += 1
    end
  end

end

function update_beta!(y::Array{Int64, 2}, theta_star::Array{Float64, 2},
                     alpha::Float64, z::Array{Float64, 2},
                     can_z::Array{Float64, 2},
                     beta::Array{Float64, 1},
                     beta_m::Array{Float64, 1},
                     beta_s::Array{Float64, 1},
                     xi::Float64, x::Array{Float64, 3},
                     x_beta::Array{Float64, 2},
                     can_x_beta::Array{Float64, 2},
                     cur_lly::Array{Float64, 2},
                     can_lly::Array{Float64, 2},
                     att::Array{Int64, 1}, acc::Array{Int64, 1},
                     mh::Array{Float64, 1}, thresh::Float64,
                     candidate::Distribution, ns::Int64, nt::Int64, np::Int64)

  # get candidate draws
  for p = 1:np
    att[p] += 1
    can_beta = beta[p] + mh[p] * rand(candidate, 1)[1]

    # faster than reshape because it doesn't need to make a copy
    can_diff = can_beta - beta[p]
    # @time for t = 1:nt, i = 1:ns
    #   can_x_beta[i, t] = x_beta[i, t] + can_diff * x[i, t, p]
    # end
    get_canxbeta!(can_x_beta, x_beta, slice(x, :, :, p), can_diff, ns, nt)
    # get_canxbeta!(can_x_beta, x_beta, sub(x, :, :, p), can_diff, ns, nt)
    # get_canxbeta!(can_x_beta, x_beta, x[:, :, p], can_diff, ns, nt)
    update_z!(can_z, xi, can_x_beta, thresh, ns, nt)
    logpdf_rarebinary!(can_lly, y, theta_star, alpha, can_z, ns, nt)

    prior = Distributions.Normal(beta_m[p], beta_s[p])
    # > 64k memory for summing over can_lly and cur_lly
    R = sum(can_lly - cur_lly) +
        logpdf(prior, can_beta) - logpdf(prior, beta[p])

    if (length(R) > 1)
      error("R has length greater than 1")
    end

    if (log(rand(1)[1]) < R)  # rand generates a vector
      beta[p] = can_beta
      copy!(x_beta, can_x_beta)
      copy!(z, can_z)
      copy!(cur_lly, can_lly)
      acc[p] += 1
    end
  end

end



# function update_beta_1!(y::Array{Int64, 2}, theta_star::Array{Float64, 2},
#                      alpha::Float64, z::Array{Float64, 2},
#                      can_z::Array{Float64, 2},
#                      beta::Array{Float64, 1},
#                      beta_m::Array{Float64, 1},
#                      beta_s::Array{Float64, 1},
#                      xi::Float64, x::Array{Float64, 3},
#                      x_beta::Array{Float64, 2},
#                      can_x_beta::Array{Float64, 2},
#                      cur_lly::Array{Float64, 2},
#                      can_lly::Array{Float64, 2},
#                      att::Array{Int64, 1}, acc::Array{Int64, 1},
#                      mh::Array{Float64, 1}, thresh::Float64,
#                      candidate::Distribution)
#   np = size(x)[3]
#   ns = size(y)[1]
#   nt = size(y)[2]

#   # get candidate draws
#   for p = 1:np
#     att[p] += 1
#     can_beta = beta[p] + mh[p] * rand(candidate, 1)[1]

#     # faster than reshape because it doesn't need to make a copy
#     # > 1Mb memory
#     for t = 1:nt, i = 1:ns
#       can_x_beta[i, t] = x_beta[i, t] + x[i, t, p] * (can_beta - beta[p])
#     end

#     update_z_2!(can_z, xi, can_x_beta, thresh)
#     logpdf_rarebinary!(can_lly, y, theta_star, alpha, can_z)

#     prior = Distributions.Normal(beta_m[p], beta_s[p])
#     # > 64k memory for summing over can_lly and cur_llys
#     R = sum(can_lly - cur_lly) +
#         logpdf(prior, can_beta) - logpdf(prior, beta[p])

#     if (length(R) > 1)
#       error("R has length greater than 1")
#     end

#     if (log(rand(1))[1] < R)  # rand generates a vector
#       beta[p] = can_beta
#       copy!(x_beta, can_x_beta)
#       copy!(z, can_z)
#       copy!(cur_lly, can_lly)
#       acc[p] += 1
#     end
#   end
# end

function update_beta_2!(y::Array{Int64, 2}, theta_star::Array{Float64, 2},
                     alpha::Float64, z::Array{Float64, 2},
                     can_z::Array{Float64, 2},
                     beta::Array{Float64, 1},
                     beta_m::Array{Float64, 1},
                     beta_s::Array{Float64, 1},
                     xi::Float64, x::Array{Float64, 3},
                     x_beta::Array{Float64, 2},
                     can_x_beta::Array{Float64, 2},
                     cur_lly::Array{Float64, 2},
                     can_lly::Array{Float64, 2},
                     att::Array{Int64, 1}, acc::Array{Int64, 1},
                     mh::Array{Float64, 1}, thresh::Float64,
                     candidate::Distribution)
  np = size(x)[3]
  ns = size(y)[1]
  nt = size(y)[2]

  # get candidate draws
  for p = 1:np
    att[p] += 1
    can_beta = beta[p] + mh[p] * rand(candidate, 1)[1]

    get_canxbeta!(can_x_beta, x_beta, x, beta, can_beta, p)

    update_z!(can_z, xi, can_x_beta, thresh)
    logpdf_rarebinary!(can_lly, y, theta_star, alpha, can_z)

    prior = Distributions.Normal(beta_m[p], beta_s[p])
    # > 64k memory for summing over can_lly and cur_llys
    R = sum(can_lly - cur_lly) +
        logpdf(prior, can_beta) - logpdf(prior, beta[p])

    if (length(R) > 1)
      error("R has length greater than 1")
    end

    if (log(rand(1))[1] < R)  # rand generates a vector
      beta[p] = can_beta
      copy!(x_beta, can_x_beta)
      copy!(z, can_z)
      copy!(cur_lly, can_lly)
      acc[p] += 1
    end

  end
end


# for testing timing of update_beta!
function sample_beta!(nreps::Int64, y::Array{Int64, 2},
                      theta_star::Array{Float64, 2}, alpha::Float64,
                      xi::Float64, x::Array{Float64, 3},
                      thresh::Float64, ns::Int64, nt::Int64, np::Int64)
  beta_keep = fill(0.0, nreps, np)

  beta = [0.0, 0.0, 0.0]
  can_beta = copy(beta)
  x_beta = fill(0.0, ns, nt)
  can_x_beta = copy(x_beta)
  z = get_z(xi_t, x_beta, thresh)
  can_z = copy(z)
  cur_lly = logpdf_rarebinary(y, theta_star_t, alpha_t, z)
  can_lly = copy(cur_lly)

  beta_m = [0.0, 0.0, 0.0]
  beta_s = [10.0, 10.0, 10.0]
  beta_can_dis = Normal(0, 1)

  att_beta = fill(1, 3)
  acc_beta = fill(1, 3)
  mh_beta = fill(0.1, 3)

  for i = 1:nreps
    beta_update = update_beta!(y, theta_star, alpha, z, can_z,
                              beta, beta_m, beta_s, xi,
                              x, x_beta, can_x_beta, cur_lly, can_lly,
                              acc_beta, att_beta, mh_beta, thresh,
                              beta_can_dis, ns, nt, np)
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

# will update cur_lly, cur_llps, theta_star
function update_alpha!(y::Array{Int64, 2}, theta_star::Array{Float64, 2},
                       can_theta_star::Array{Float64, 2},
                       a::Array{Float64, 2}, alpha::Float64,
                       cur_lly::Array{Float64, 2}, can_lly::Array{Float64, 2},
                       cur_llps::Array{Float64, 2}, can_llps::Array{Float64, 2},
                       z::Array{Float64, 2}, w::Array{Float64, 2},
                       mid_points::Array{Float64, 1},
                       bin_width::Array{Float64, 1}, att::Int64, acc::Int64,
                       mh::Float64)
  nt = size(y)[2]
  nknots = size(a)[1]

  att += 1

  cur_alpha_star = (alpha - 0.000001) / (0.999999 - 0.000001)
  cur_alpha_star = quantile(Normal(0, 1), cur_alpha_star)[1]
  can_alpha_star = rand(Normal(cur_alpha_star, mh))[1]
  can_alpha = cdf(Normal(0, 1), can_alpha_star)
  can_alpha = can_alpha * (0.999999 - 0.000001) + 0.000001
  can_theta_star = get_thetastar(w, a, can_alpha)

  logpdf_rarebinary!(can_lly, y, theta_star, alpha, z)
  logpdf_ps!(can_llps, a, can_alpha, mid_points, bin_width)

  R = sum(can_lly - cur_lly) + sum(can_llps - cur_llps) +
      logpdf(Normal(0, 1), can_alpha_star) -
      logpdf(Normal(0, 1), cur_alpha_star)

  if (log(rand(1))[1] < R)
    alpha = can_alpha
    copy!(theta_star, can_theta_star)
    copy!(cur_lly, can_lly)
    copy!(cur_llps, can_llps)
    acc += 1
  end

  return (alpha, att, acc)
end

function update_a!(y::Array{Int64, 2}, theta_star::Array{Float64, 2},
                   a::Array{Float64, 2}, alpha::Float64,
                   cur_lly::Array{Float64, 2}, can_lly::Array{Float64, 2},
                   cur_llps::Array{Float64, 2}, can_llps::Array{Float64, 2},
                   z::Array{Float64, 2}, w::Array{Float64, 2},
                   mid_points::Array{Float64, 1}, bin_width::Array{Float64, 1},
                   mh::Array{Float64, 1}, cuts::Array{Float64, 1})
  nt = size(y)[2]
  nknots = size(a)[1]

  for t = 1:nt
    cur_lly_t = cur_lly[:, t]
    can_lly_t = can_lly[:, t]
    for k = 1:nknots
      cur_a = a[k, t]
      l1 = get_level(cur_a, cuts)
      can_a = exp(rand(Normal(log(cur_a), mh[l1]), 1)[1])
      l2 = get_level(can_a, cuts)
      www = w[:, k].^(1 / alpha)
      # can_theta_star only changes at a site when it's near the knot
      can_theta_star = theta_star[:, t] + www .* (can_a - cur_a)
      can_llps[k, t] = logpdf_ps(can_a, alpha, mid_points, bin_width)
      logpdf_rarebinary!(can_lly_t, y[:, t], can_theta_star, alpha, z[:, t])

      R = sum(can_lly_t - cur_lly_t) + can_llps[k, t] - cur_llps[k, t] +
          logpdf(LogNormal(cur_a, can_a, mh[l2])) -
          logpdf(LogNormal(can_a, cur_a, mh[l1]))

      if (log(rand(1))[1] < R)
        a[k, t] = can_a
        theta_star[:, t] = can_theta_star
        cur_lly_t = can_lly_t
        cur_llps[k, t] = can_llps[k, t]
      end
    end
  end

  logpdf_rarebinary!(cur_lly, y, theta_star, alpha, z)
end

end