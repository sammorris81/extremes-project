using Distributions
import Base.LinAlg.BLAS.gemv!
using DataTransformations
using MetropolisUpdaters

# Simulated example - Logistic regression
# model P(Y = 1) = exp(Xβ) / (1 + exp(Xβ))
#                = 1 / (1 + exp(-Xβ))
#   Xβ = β₀ + β₁x₁ + β₂x₂
# priors:
#   β ~ N(0, 100) i.i.d.

# generate covariate data and simulated observations
n = 200
X = reshape(rand(Normal(), n*2), n, 2)
X = hcat(ones(n), X)
βₐ = [0.5, -1., 1.]
Xβₐ = X * βₐ
π = 1 ./ (1 + exp(-Xβₐ))
y = [rand(Binomial(1, π[i]), 1)[1] for i = 1:n]

# functions to update calculated values
function updatexbeta!(Xβ::CalculatedValuesVector, X::Matrix, β::MetropolisVector)
  gemv!('N', 1.0, X, activevalue(β), 0.0, activevalue(Xβ))
end

function updatelly!(ll::CalculatedValuesVector, y::Vector,
                    Xβ::CalculatedValuesVector)
  activeXβ = activevalue(Xβ)
  for i = 1:ll.length
    if y[i] == 1
      activevalue(ll)[i] = -log(1 + exp(-activeXβ[i]))
    else
      activevalue(ll)[i] = -log(1 + exp(activeXβ[i]))
    end
  end
end

β = createmetropolis(3, prior=Distributions.Normal(0, 100))
Xβ = createcalculatedvalues(n, updater=updatexbeta!,
                            requires=(X, β))
ll = createcalculatedvalues(n, updater=updatelly!,
                            requires=(y, Xβ))
β.ll = [ll]  # assign the actual likelihood values to the parameter
β.impacts = [Xβ]

niters = 10000
burn   = 2000

β_keep = fill(0.0, niters, β.length)
@time for iter = 1:niters
  updatemhseq!(β)

  if iter < (burn / 2)
    updatestepsize!(β)
  end

  β_keep[iter, :] = β.cur
end

using Gadfly
plot(x=1:niters, y=β_keep[:, 1], Geom.line)
plot(x=1:niters, y=β_keep[:, 2], Geom.line)
plot(x=1:niters, y=β_keep[:, 3], Geom.line)
