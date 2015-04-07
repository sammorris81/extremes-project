module UpdateModel
export updatebeta!,
       updatexi!,
       updatealpha!

importall Distributions
importall AuxFunctions

# TODO: Rewrite to loop over parameter updates
function updatebeta!(y::Array{Int64, 2}, theta_star::Array{Float64, 2},
                     alpha::Array{Float64, 1}, z::Array{Float64, 2},
                     can_z::Array{Float64, 2},
                     z_star::Array{Float64, 2},
                     can_z_star::Array{Float64, 2},
                     beta::Array{Float64, 1},
                     beta_m::Array{Float64, 1},
                     beta_s::Array{Float64, 1},
                     xi::Array{Float64, 1}, x::Array{Float64, 3},
                     x_beta::Array{Float64, 2},
                     can_x_beta::Array{Float64, 2},
                     cur_lly::Array{Float64, 2},
                     can_lly::Array{Float64, 2},
                     att::Array{Int64, 1}, acc::Array{Int64, 1},
                     mh::Array{Float64, 1}, thresh::Float64,
                     candidate::Distribution)

  ns, nt = size(y)
  np = size(x)[3]
  alpha_inv = 1 / alpha[1]

  # get candidate draws
  for p = 1:np
    att[p] += 1
    can_beta = beta[p] + mh[p] * rand(candidate, 1)

    # faster than reshape because it doesn't need to make a copy
    can_diff = can_beta[1] - beta[p]
    # @time for t = 1:nt, i = 1:ns
    #   can_x_beta[i, t] = x_beta[i, t] + can_diff * x[i, t, p]
    # end
    # get_canxbeta!(can_x_beta, x_beta, slice(x, :, :, p), can_diff, ns, nt)
    # get_canxbeta!(can_x_beta, x_beta, sub(x, :, :, p), can_diff, ns, nt)
    updatecanxbeta!(can_x_beta, x_beta, x[:, :, p], can_diff, ns, nt)
    updatez!(can_z, xi, can_x_beta, thresh, ns, nt)
    can_z_star = can_z.^alpha_inv

    # update the likelihood`
    logpdf_rarebinary!(can_lly, y, theta_star, can_z_star, ns, nt)

    prior = Distributions.Normal(beta_m[p], beta_s[p])
    # > 64k memory for summing over can_lly and cur_lly
    R = sum(can_lly) - sum(cur_lly) +
        logpdf(prior, can_beta[1]) - logpdf(prior, beta[p])

    if (length(R) > 1)
      error("R has length greater than 1")
    end

    if (log(rand(1)[1]) < R)  # rand generates a vector
      beta[p] = can_beta[1]
      copy!(x_beta, can_x_beta)
      copy!(z, can_z)
      copy!(cur_lly, can_lly)
      acc[p] += 1
    end
  end

end

function updatexi!(y::Array{Int64, 2}, theta_star::Array{Float64, 2},
                   alpha::Array{Float64, 1}, z::Array{Float64, 2},
                   can_z::Array{Float64, 2},
                   z_star::Array{Float64, 2},
                   can_z_star::Array{Float64, 2},
                   x_beta::Array{Float64, 2},
                   can_x_beta::Array{Float64, 2},
                   xi::Array{Float64, 1},
                   cur_lly::Array{Float64, 2},
                   can_lly::Array{Float64, 2},
                   att::Array{Int64, 1}, acc::Array{Int64, 1},
                   mh::Array{Float64, 1}, thresh::Float64,
                   candidate::Distribution)
  nt = size(y)[2]
  att[1] += 1
  alpha_inv = 1 / alpha[1]

  can_xi = xi[1] + mh[1] *rand(candidate, 1)
  updatez!(can_z, xi, can_x_beta, thresh, ns, nt)
  can_z_star = can_z.^alpha_inv

  logpdf_rarebinary!(can_lly, y, theta_star, can_z_star, ns, nt)

  prior = Distributions.Normal(xi_m[1], xi_s[1])
  R = sum(can_lly) - sum(cur_lly) +
        logpdf(prior, can_xi[1]) - logpdf(prior, xi[1])

  if (log(rand(1)[1]) < R)
    xi[1] = can_xi[1]
    copy!(z, can_z)
    copy!(z_star, can_z_star)
    copy!(cur_lly, can_lly)
    acc[1] += 1
  end

end

function updatea!(y::Array{Int64, 2}, theta_star::Array{Float64, 2},
                  a::Array{Float64, 2}, alpha::Array{Float64, 1},
                  cur_lly::Array{Float64, 2}, can_lly::Array{Float64, 2},
                  cur_llps::Array{Float64, 2}, can_llps::Array{Float64, 2},
                  z_star::Array{Float64, 2}, w_star::Array{Float64, 2},
                  mid_points::Array{Float64, 1}, bin_width::Array{Float64, 1},
                  mh::Array{Float64, 1}, cuts::Array{Float64, 1})
  nt = size(y)[2]
  nknots = size(a)[1]

  for t = 1:nt
    cur_lly_t = cur_lly[:, t]
    can_lly_t = can_lly[:, t]
    for k = 1:nknots
      cur_a = a[k, t]
      l1    = get_level(cur_a, cuts)
      can_a = exp(rand(Normal(log(cur_a), mh[l1]), 1)[1])
      l2    = get_level(can_a, cuts)

      # can_theta_star only changes at a site when it's near the knot
      can_theta_star = theta_star[:, t] + w_star[:, k] .* (can_a - cur_a)
      if sum(can_theta_star .<= 0) > 0
        can_llps[k, t] = -Inf
        can_lly_t[k] = -Inf
      else
        can_llps[k, t] = logpdf_ps(can_a, alpha[1], mid_points, bin_width)
        logpdf_rarebinary!(can_lly_t, y[:, t], can_theta_star, z_star[:, t])
      end

      R = sum(can_lly_t) - sum(cur_lly_t) +
          can_llps[k, t] - cur_llps[k, t] +
          logpdf(LogNormal(can_a, mh[l2]), cur_a) -
          logpdf(LogNormal(cur_a, mh[l1]), can_a)

      if (log(rand(1))[1] < R)
        a[k, t] = can_a
        theta_star[:, t] = can_theta_star
        cur_lly_t = can_lly_t
        cur_llps[k, t] = can_llps[k, t]
      end
    end
  end

  logpdf_rarebinary!(cur_lly, y, theta_star, z_star)
end

# will update cur_lly, cur_llps, theta_star
function updatealpha!(y::Array{Int64, 2}, theta_star::Array{Float64, 2},
                      can_theta_star::Array{Float64, 2},
                      a::Array{Float64, 2}, alpha::Array{Float64, 1},
                      cur_lly::Array{Float64, 2}, can_lly::Array{Float64, 2},
                      cur_llps::Array{Float64, 2}, can_llps::Array{Float64, 2},
                      z::Array{Float64, 2}, z_star::Array{Float64, 2},
                      w::Array{Float64, 2}, w_star::Array{Float64, 2},
                      mid_points::Array{Float64, 1}, bin_width::Array{Float64, 1},
                      att::Array{Int64, 1}, acc::Array{Int64, 1},
                      mh::Array{Float64, 1}, candidate::Distribution)
  nt = size(y)[2]
  nknots = size(a)[1]

  att[1] += 1

  # the candidate stays a scalar
  alpha_star = transprobit(alpha[1], [0.000001, 0.999999], false)
  can_alpha_star = alpha_star + mh[1] * rand(candidate, 1)[1]
  can_alpha = transprobit(can_alpha_star, [0.000001, 0.999999], true)
  can_alpha_inv = 1 / can_alpha
  can_w_star = w.^can_alpha_inv
  can_z_star = z.^can_alpha_inv
  can_theta_star = updatethetastar(can_w_star, a)

  logpdf_rarebinary!(can_lly, y, can_theta_star, can_z_star)
  logpdf_ps!(can_llps, a, can_alpha, mid_points, bin_width)

  R = sum(can_lly) - sum(cur_lly) + sum(can_llps) - sum(cur_llps) +
      logpdf(Normal(0, 1), can_alpha_star) -
      logpdf(Normal(0, 1), alpha_star)

  if (log(rand(1))[1] < R)
    alpha[1] = can_alpha
    copy!(theta_star, can_theta_star)
    copy!(cur_lly, can_lly)
    copy!(cur_llps, can_llps)
    acc[1] += 1
  end

end

function updaterho!(y::Array{Int64, 2}, theta_star::Array{Float64, 2},
                    a::Array{Float64, 2}, alpha::Array{Float64, 1},
                    cur_lly::Array{Float64, 2}, can_lly::Array{Float64, 2},
                    z_star::Array{Float64, 2},
                    w::Array{Float64, 2}, w_star::Array{Float64, 2},
                    dw2::Array{Float64, 2}, rho::Array{Float64, 1},
                    rho_upper::Float64,
                    att::Array{Int64, 1}, acc::Array{Int64, 1},
                    mh::Array{Float64, 1}, candidate::Distribution)
  nt = size(y)[2]
  nknots = size(a)[1]

  att[1] += 1
  alpha_inv = 1 / alpha[1]

  rho_star = transprobit(rho[1], [0.000001, rho_upper], false)
  can_rho_star = rho_star + mh[1] * rand(candidate, 1)[1]
  can_rho = transprobit(can_rho_star, [0.000001, rho_upper], true)
  can_w = make_w(dw2, can_rho)
  std_w!(can_w)
  can_w_star = can_w.^(alpha_inv)
  can_theta_star = updatethetastar(can_w_star, a)

  logpdf_rarebinary!(can_lly, y, can_theta_star, z_star)

  R = sum(can_lly) - sum(cur_lly) +
      logpdf(Normal(0, 1), can_rho_star) -
      logpdf(Normal(0, 1), rho_star)

  if (log(rand(1))[1] < R)
    rho[1] = can_rho
    copy!(w, can_w)
    copy!(w_star, can_w_star)
    copy!(theta_star, can_theta_star)
    copy!(cur_lly, can_lly)
    acc[1] += 1
  end
end




# function update_beta_2!(y::Array{Int64, 2}, theta_star::Array{Float64, 2},
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

#     get_canxbeta!(can_x_beta, x_beta, x, beta, can_beta, p)

#     update_z!(can_z, xi, can_x_beta, thresh)
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


# for testing timing of update_beta!
# function sample_beta!(nreps::Int64, y::Array{Int64, 2},
#                       theta_star::Array{Float64, 2}, alpha::Float64,
#                       xi::Float64, x::Array{Float64, 3},
#                       thresh::Float64, ns::Int64, nt::Int64, np::Int64)
#   beta_keep = fill(0.0, nreps, np)

#   beta = [0.0, 0.0, 0.0]
#   can_beta = copy(beta)
#   x_beta = fill(0.0, ns, nt)
#   can_x_beta = copy(x_beta)
#   z = get_z(xi, x_beta, thresh)
#   can_z = copy(z)
#   cur_lly = logpdf_rarebinary(y, theta_star, alpha, z)
#   can_lly = copy(cur_lly)

#   beta_m = [0.0, 0.0, 0.0]
#   beta_s = [10.0, 10.0, 10.0]
#   beta_can_dis = Normal(0, 1)

#   att_beta = fill(1, 3)
#   acc_beta = fill(1, 3)
#   mh_beta = fill(0.1, 3)

#   for i = 1:nreps
#     beta_update = update_beta!(y, theta_star, alpha, z, can_z,
#                               beta, beta_m, beta_s, xi,
#                               x, x_beta, can_x_beta, cur_lly, can_lly,
#                               acc_beta, att_beta, mh_beta, thresh,
#                               beta_can_dis, ns, nt, np)
#     beta_keep[i, :] = beta
#     if i % 500 == 0
#       println(i)
#     end
#     # println(pointer(can_beta))
#     # println(pointer(can_x_beta))
#     # println(pointer(can_z))
#     # println(pointer(can_lly))
#   end
#   return beta_keep
# end

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
