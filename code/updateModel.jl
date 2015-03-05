module UpdateModel
export updateBeta, updateBeta!, sampleBeta, sampleBeta!

using Distributions
using AuxFunctions
function updateBeta(y::Array{Int64}, theta_star::Array{Float64},
                     alpha::Float64, z::Array{Float64}, beta::Array{Float64},
                     beta_m::Array{Float64}, beta_s::Array{Float64}, xi::Float64,
                     x::Array{Float64}, x_beta::Array{Float64}, cur_lly::Array{Float64},
                     acc::Int64, att::Int64, mh::Array{Float64})
  np = size(x)[3]
  ns = size(y)[1]
  nt = size(y)[2]
  att += 1

  # get candidate draws
  can_beta = Array(Float64, 3)
  for i = 1:np
    can_beta[i] = rand(Distributions.Normal(beta[i], mh[i]))
  end

  can_x_beta = Array(Float64, ns, nt)
  for t = 1:nt
    can_x_beta[:, t] = reshape(x[:, t, :], ns, 3) * can_beta
  end

  can_z = getZ(xi, can_x_beta)
  can_lly = logLikeY(y, theta_star, alpha, can_z)

  R = sum(can_lly - cur_lly)
  for i = 1:np
    prior = Distributions.Normal(beta_m[i], beta_s[i])
    R += logpdf(prior, can_beta[i]) - logpdf(prior, beta[i])
  end

  if (length(R) > 1)
    error("R has length greater than 1")
  end

  if (log(rand(1))[1] < R)  # rand generates a vector
    beta = can_beta
    x_beta = can_x_beta
    z = can_z
    cur_lly = can_lly
    acc += 1
  end

  results = {"beta" => beta, "x_beta" => x_beta, "z" => z,
             "cur_lly" => cur_lly, "att" => att, "acc" => acc}

  return results
end

function updateBeta!(y::Array{Int64}, theta_star::Array{Float64}, alpha::Float64,
                     z::Array{Float64}, can_z::Array{Float64},
                     beta::Array{Float64}, can_beta::Array{Float64},
                     beta_m::Array{Float64}, beta_s::Array{Float64},
                     xi::Float64, x::Array{Float64},
                     x_beta::Array{Float64}, can_x_beta::Array{Float64},
                     cur_lly::Array{Float64}, can_lly::Array{Float64},
                     acc::Int64, att::Int64, mh::Array{Float64}, candidate)
  np = size(x)[3]::Int64
  ns = size(y)[1]::Int64
  nt = size(y)[2]::Int64
  R = 0.0
  att += 1

  # get candidate draws
  for i = 1:np
    can_beta[i] = beta[i] + mh[i] * rand(candidate)
  end

  # loop from the last to the first
  fill!(can_x_beta, 0.0)
  for k = 1:np, j = 1:nt
    can_x_beta[i, j] += x[i, j, k] * can_beta[k]
  end

  updateZ!(xi, can_x_beta, can_z)
  logLikeY!(y, theta_star, alpha, can_z, can_lly)

  R = sum(can_lly - cur_lly)
  for i = 1:np
    prior = Distributions.Normal(beta_m[i], beta_s[i])
    R += logpdf(prior, can_beta[i]) - logpdf(prior, beta[i])
  end

  if (length(R) > 1)
    error("R has length greater than 1")
  end

  if (log(rand(1))[1] < R)  # rand generates a vector
    beta = can_beta
    x_beta = can_x_beta
    z = can_z
    cur_lly = can_lly
    acc += 1
  end

  return
end

function sampleBeta(nreps::Int64, y::Array{Int64}, theta_star::Array{Float64},
                    alpha::Float64, z::Array{Float64}, beta::Array{Float64},
                    beta_m::Array{Float64}, beta_s::Array{Float64}, xi::Float64,
                    x::Array{Float64}, x_beta::Array{Float64}, cur_lly::Array{Float64},
                    acc_beta::Int64, att_beta::Int64, mh_beta::Array{Float64})
  np = size(x)[3]
  beta_keep = fill(0.0, nreps, np)
  for i = 1:nreps
    beta_update = updateBeta(y, theta_star, alpha, z, beta, beta_m, beta_s, xi,
                             x, x_beta, cur_lly, acc_beta, att_beta, mh_beta)
    beta = beta_update["beta"]
    x_beta = beta_update["x_beta"]
    z = beta_update["z"]
    cur_lly = beta_update["cur_lly"]
    att_beta = beta_update["att"]
    acc_beta = beta_update["acc"]

    beta_keep[i, :] = beta
    if i % 500 == 0
      println(i)
    end
  end
  return beta_keep
end

function sampleBeta!(nreps::Int64, y::Array{Int64}, theta_star::Array{Float64},
                    alpha::Float64, z::Array{Float64}, beta::Array{Float64},
                    beta_m::Array{Float64}, beta_s::Array{Float64}, xi::Float64,
                    x::Array{Float64}, x_beta::Array{Float64}, cur_lly::Array{Float64},
                    acc_beta::Int64, att_beta::Int64, mh_beta::Array{Float64})
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
    beta_update = updateBeta!(y, theta_star, alpha, z, can_z,
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

end
