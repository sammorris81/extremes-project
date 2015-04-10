using Distributions
import Base.LinAlg.BLAS.gemv!
push!(LOAD_PATH, homedir()"/repos-git/extremes-project/code/types")
using DataTransformations
using MCMCCalculated
using MHParameters
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



n = 200
s = hcat(rand(n), rand(n))
X = reshape(rand(Normal(), n*2), n, 2)
X = hcat(ones(n), X)
βₐ = [0.5, -1., 1.]
Xβₐ = X * βₐ
π = exp(Xβₐ) ./ (1 + exp(Xβₐ))
y = [rand(Binomial(1, π[i]), 1)[1] for i = 1:n]