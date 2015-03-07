cd()
cd("./repos-git/extremes-project/code")

# packages
using Distributions
using Distances
using ProfileView

# modules
using AuxFunctions
using UpdateModel


beta = fill(0.0, 3)
beta_m = fill(0.0, 3)
beta_s = fill(10.0, 3)
alpha = 0.5

ns = 100
nt = 50
nknots = 100
y = fill(1, ns, nt)
theta_star = Array(Float64, ns, nt)
z = Array(Float64, ns, nt)
x_beta = Array(Float64, ns, nt)
x = Array(Float64, ns, nt, 3)
fill!(x, 1.0)
xi = 0.1
for j = 1:nt
  for i = 1:ns
    y[i, j] = rand(Binomial(1, 0.1))
    z[i, j] = rand(Chisq(2))
    theta_star[i, j] = rand(Chisq(3))
    if j ==1
      x[i, j, 2:3] = rand(2)
    else
      x[i, j, 2:3] = x[i, 1, 2:3]
    end
    x_beta[i, j] = rand(Normal(0, 1))
  end
end

# get spatial locations
s = reshape(x[:, 1, 2:3], ns, 2)

# replaces expand.grid from R
knots_x = linspace(0.01, 0.99, 10)
knots = hcat(repeat(knots_x, inner=[1], outer=[10]),
             repeat(knots_x, inner=[10], outer=[1]))

thresh = 0.0

beta = fill(10.0, 3)
beta[2] = 0
beta[3] = 0
for t = 1:nt
  x_beta[:, t] = reshape(x[:, t, :], ns, 3) * beta
end
rho = 1.0
y = sample_rarebinary(x, s, knots, beta, xi, alpha, rho, 10.0)





cur_lly = logpdf_rarebinary(y, theta_star, alpha, z)
acc_beta = fill(1, 3)
att_beta = fill(1, 3)
mh_beta = fill(1.0, 3)

nreps = 5000
beta_keep = Array(Float64, nreps, 3)
candidate = Normal(0, 1)

@time get_z(xi, x_beta)
@time update_z!(z, xi, x_beta)

(acc, att, mh) = mh_update(acc, att, mh)

@code_llvm mh_update(acc, att, mh)

# initialize
# sample_beta(1, y, theta_star, alpha, z, beta, beta_m, beta_s, xi,
#            x, x_beta, cur_lly, acc_beta, att_beta, mh_beta)
# @time sampleBeta(500, y, theta_star, alpha, z, beta, beta_m, beta_s, xi,
#                  x, x_beta, cur_lly, acc_beta, att_beta, mh_beta)

# @elapsed sampleBeta(5, y, theta_star, alpha, z, beta, beta_m, beta_s, xi,
#                     x, x_beta, cur_lly, acc_beta, att_beta, mh_beta)

# @time updateBeta(y, theta_star, alpha, z, beta, beta_m, beta_s, xi,
#                     x, x_beta, cur_lly, acc_beta, att_beta, mh_beta)

@time update_beta!(y, theta_star, alpha, z, beta, beta_m, beta_s, xi,
                     x, x_beta, cur_lly, acc_beta, att_beta, mh_beta, candidate)
sample_beta!(2, y, theta_star, alpha, z, beta, beta_m, beta_s, xi, x, x_beta,
            cur_lly, acc_beta, att_beta, mh_beta)


@time sample_beta!(2000, y, theta_star, alpha, z, beta, beta_m, beta_s, xi,
                  x, x_beta, cur_lly, acc_beta, att_beta, mh_beta)


np = size(x)[3]
ns = size(y)[1]
nt = size(y)[2]


can_x_beta = Array(Float64, ns, nt)

t = 1
typeof(y)

logpdf_rarebinary(y, can_x_beta, 0.5, x)

sum(can_x_beta, 2)
1 + can_x_beta



rand(1)

ns = 1000
nt = 300
nknots = 50
dw2 = Array(Float64, ns, nknots)
@time w = make_w(dw2, 1.0)
@time std_w(w)

x_beta
z
xi
@time for i = 1:20
  get_z(xi, x_beta)
end

@time for i = 1:20
  update_z!(xi, x_beta, z)
end

sample_beta!(1, y, theta_star, alpha, z, beta, beta_m, beta_s, xi,
             x, x_beta, cur_lly, acc_beta, att_beta, mh_beta)






