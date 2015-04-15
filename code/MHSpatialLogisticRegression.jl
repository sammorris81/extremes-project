using Distributions
import Base.LinAlg.BLAS.gemv!
push!(LOAD_PATH, homedir()"/repos-git/extremes-project/code/types")
using DataTransformations
using MCMCCalculated
using MHParameters
using SpatialCorrelation
using Distances

# Simulated example - Spatial Logistic Regression
# model P(Y = 1) = exp(Xβ + w(s)) / (1 + exp(Xβ + w(s)))
#                = 1 / (1 + exp(-(Xβ + w(s)))
#     Xβ = β₀ + β₁x₁ + β₂x₂ + β₃long + β₄lat
#   w(s) ~ MVN(0, Σ)
#      s ∈ [0, 1] x [0, 1]
#      Σ = σ² * exp(-h / ρ)
# priors:
#   β ~ N(0, 100) i.i.d.
#  σ² ~ IG(0.1)

# covariate information
n = 200
s = hcat(rand(n), rand(n))
X = reshape(rand(Normal(), n*2), n, 2)
X = hcat(ones(n), X)
βₐ = [0.5, -1., 1.]
Xβₐ = X * βₐ

# spatial
d = fill(0.0, n, n)
for j = 1:n, i = 1:n
  d[i, j] = Distances.euclidean(vec(s[i, :]), vec(s[j, :]))
end
σ²ₐ = 1
ϕₐ = 0.2
Σₐ = exponential(d, σ²ₐ, ϕₐ)
Σₐ_ch = chol(Σₐ, :L)
wₐ = Σₐ_ch * rand(Normal(0, 1), n)

# generate data
π = 1 ./ (1 + exp(-(Xβₐ + wₐ)))
y = [rand(Binomial(1, π[i]), 1)[1] for i = 1:n]

# functions to update calculated values
function updatexbeta!(Xβ::CalculatedValuesVector, X::Matrix, β::MetropolisVector)
  gemv!('N', 1.0, X, activevalue(β), 0.0, activevalue(Xβ))
end

function updatez!(z::CalculatedValuesVector, Xβ::CalculatedValuesVector,
                  w::MetropolisVector)
  activeXβ = activevalue(Xβ)
  activew  = activevalue(w)

  for i = 1:z.length
    z[i] = activeXβ[i] + activew[i]
  end
end

function updatecov!(Σ::CalculateValuesMatrix, d::Matrix, σ²::MetropolisVector,
                    ϕ::MetropolisVector)
  exponential!(activevalue(Σ), d, activevalue(σ²), activevalue(ϕ))
end

function updatellw!(ll:CalculatedValuesMatrix, w::MetropolisMatrix,
                    Σ::CalculatedValuesMatrix)

end

function updatelly!(ll::CalculatedValuesVector, y::Vector, z::CalculatedValuesVector)
  activez = activevalue(z)
  activell = activevalue(ll)
  for i = 1l.length
    if y[i] == 1
      activell[i] = -log(1 + exp(-activez[i]))
    else
      activell[i] = -log(1 + exp(activez[i]))
    end
  end
end

# mean function
β = createmetropolis(3, prior = Distributions.Normal(0, 100))

Xβ = createcalculatedvalues(n, updater = updatexbeta!, requires = (X, β))

# spatial dependence
σ² = createmetropolis(1, prior = Distributions.InverseGamma(0.1, 0.1),
                      support = [0, Inf], transform = translog)

ϕ  = createmetropolis(1, prior = Distributions.Normal(0, 1),
                      support = [0.000001, 1.2],
                      transform = transunif, usestar = true)

Σ  = createcalculatedvalues(n, n, updater = updatecov!, requires = (d, σ², ϕ))

w  = createmetropolis(n, n, prior = Distributions.MvNormal(zeros(n), activevalue(Σ)),
                      support = [-Inf, Inf])

# likelihood
llw = createcalculatedvalues(n, n, updater = updatellw!, requires = (w, Σ))
lly = createcalculatedvalues(n, updater = updatelly!, requires = (y, z))
