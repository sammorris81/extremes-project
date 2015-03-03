cd()
cd("./repos-git/extremes-project/code")

using Distributions
using AuxFunctions
using UpdateModel
using ProfileView


beta = fill(0.0, 3)
beta_m = fill(0.0, 3)
beta_s = fill(10.0, 3)
alpha = 0.5

ns = 100
nt = 50
y = fill(1, ns, nt)
theta_star = Array(Float64, ns, nt)
z = Array(Float64, ns, nt)
x_beta = Array(Float64, ns, nt)
x = Array(Float64, ns, nt, 3)
xi = 0.1
for j = 1:nt
  for i = 1:ns
    y[i, j] = rand(Binomial(1, 0.1))
    z[i, j] = rand(Chisq(2))
    theta_star[i, j] = rand(Chisq(3))
    x[i, j, 2:3] = rand(2)
    x_beta[i, j] = rand(Normal(0, 1))
  end
end

cur_lly = logLikeY(y, theta_star, alpha, z)
acc_beta = 1
att_beta = 1
mh_beta = fill(1.0, 3)

nreps = 5000
beta_keep = Array(Float64, nreps, 3)
candidate = Normal(0, 1)

@time getZ(xi, x_beta)
@time updateZ!(xi, x_beta, z)



# initialize
Profile.clear()
sampleBeta(1, y, theta_star, alpha, z, beta, beta_m, beta_s, xi,
           x, x_beta, cur_lly, acc_beta, att_beta, mh_beta)
@time sampleBeta(500, y, theta_star, alpha, z, beta, beta_m, beta_s, xi,
                 x, x_beta, cur_lly, acc_beta, att_beta, mh_beta)

@elapsed sampleBeta(5, y, theta_star, alpha, z, beta, beta_m, beta_s, xi,
                    x, x_beta, cur_lly, acc_beta, att_beta, mh_beta)

@time updateBeta(y, theta_star, alpha, z, beta, beta_m, beta_s, xi,
                    x, x_beta, cur_lly, acc_beta, att_beta, mh_beta)

@time updateBeta!(y, theta_star, alpha, z, beta, beta_m, beta_s, xi,
                     x, x_beta, cur_lly, acc_beta, att_beta, mh_beta, candidate)
sampleBeta!(2, y, theta_star, alpha, z, beta, beta_m, beta_s, xi,
            x, x_beta, cur_lly, acc_beta, att_beta, mh_beta)
@time sampleBeta!(2000, y, theta_star, alpha, z, beta, beta_m, beta_s, xi,
                  x, x_beta, cur_lly, acc_beta, att_beta, mh_beta)
Array(Float64, 3, 2)
@elapsed for i = 1:1000
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

@elapsed for i = 1:1000
  beta_update = updateBeta!(y, theta_star, alpha, z, beta, beta_m, beta_s, xi,
                           x, x_beta, cur_lly, acc_beta, att_beta, mh_beta)

  beta_keep[i, :] = beta
  if i % 500 == 0
    println(i)
  end
end

test = rand(2)
length(test) > 1
length(x)

rand(1)[1] .< 0.5
# you can use fill to create a matrix/array initialized at a specific value
y = fill(1, 40, 50)
x = fill(1.0, 40, 50, 3)
for i = 1:40
  for j = 1:50
    x[i, j, 2:3] = rand(Normal(0, 1), 2)
  end
end
x

np = size(x)[3]
ns = size(y)[1]
nt = size(y)[2]

@time rand(Normal(0, 1), 1)
temp = Normal(0, 1)
@time rand(temp, 1)
can_x_beta = Array(Float64, ns, nt)
t = 1
typeof(y)

# reshape allows you to change from a 3D array to 2D matrix
reshape(x[:, t, :], ns, 3)
for t = 1:nt
  can_x_beta[:, t] = reshape(x[:, t, :], ns, 3) * beta
end

can_x_beta
exp(can_x_beta)
can_x_beta.^2
typeof(y)
logLikeY(y, can_x_beta, 0.5, x)

sum(can_x_beta, 2)
1 + can_x_beta
# Julia uses a hash table in place of lists in R
results = {"x" => x, "y" => y, "can_x_beta" => can_x_beta}
results["can_x_beta"]

# rows of a column are contiguous
# cols of a row are not
rand(1)

ns = 1000
nt = 300
nknots = 50
dw2 = Array(Float64, ns, nknots)
@time w = makeW(dw2, 1.0)
@time stdW(w)

x_beta
z
xi
@time for i = 1:20
  getZ(xi, x_beta)
end

@time for i = 1:20
  updateZ!(xi, x_beta, z)
end

sampleBeta!(2, y, theta_star, alpha, z, beta, beta_m, beta_s, xi,
            x, x_beta, cur_lly, acc_beta, att_beta, mh_beta)

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

pointer(x)

pointer(x[:, 1, :])

pointer(sub(x, :, 1, :))

beta
reshape(x[:, 1, :], 1000, 3) * beta
reshape(sub(x, :, 1, :), 1000, 3) * beta

pointer(reshape(sub(x, :, 1, :), 1000, 3))


subx = x[1:3, 1:3, 1:3]
reshape(subx, 27, 1)
pointer(x)
fill!(x, 0.0)
pointer(x)

x = fill(1.0, 4, 5, 3)
for i = 1:4
  for j = 1:5
    x[i, j, 2:3] = rand(Normal(0, 1), 2)
  end
end

np = 3
nt = 5
ns = 4
can_x_beta = fill(0.0, ns, nt)
beta = [1.0, 2.0, 3.0]
for k = 1:np, j = 1:nt, i = 1:ns
  can_x_beta[i, j] += x[i, j, k] * beta[k]
end

can_x_beta2 = Array(Float64, ns, nt)
for t = 1:nt
  can_x_beta2[:, t] = reshape(x[:, t, :], ns, 3) * beta
end

x
beta
beta = fill(0.0, 3, 1)
broadcast(*, x[:, 1, :], beta')
