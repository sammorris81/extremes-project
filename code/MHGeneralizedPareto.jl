using Distributions
import Base.LinAlg.BLAS.gemv!
using DataTransformations
using ExtremeValueDistributions
using MetropolisUpdaters

# Simulated example - GEV Data
#   Z ~ GeneralizedPareto(0.0, 1.0, 0.2)
# priors:
#   μ ~ Normal(0, 100)
#   σ ~ InvGamma(0.1, 0.1)
#   ξ ~ Normal(0, 0.5)
# generate covariate data and simulated observations
n = 1000
μₐ = 0.0
σₐ = 2.0
ξₐ = 0.2
z = rand(GeneralizedPareto(μₐ, σₐ, ξₐ), n)

# using Gadfly
# plot(layer(x=z, Geom.density, Theme(default_color=color("red"))),
#      layer(x->pdf(GeneralizedExtremeValue(μₐ[1], σₐ[1], ξₐ[1]), x), -5, 10))

# functions to update calculated values
function updatelly!(ll::CalculatedValuesVector, y::Vector,
                    μ::Real, σ::MetropolisVector, ξ::MetropolisVector)
  activeσ = activevalue(σ)[1]
  activeξ = activevalue(ξ)[1]
  datall = GeneralizedPareto(μ, activeσ, activeξ)

  for i = 1:ll.length
    activevalue(ll)[i] = logpdf(datall, y[i])
  end
end

μ = 0.0
σ = createmetropolis(1, prior=Distributions.InverseGamma(0.1, 0.1), initial=2.0,
                     transform=translog, support=[0, Inf])
ξ = createmetropolis(1, prior=Distributions.Normal(0.0, 0.5))

ll = createcalculatedvalues(n, updater=updatelly!,
                            requires=(z, μ, σ, ξ))

σ.ll = [ll]  # assign the actual likelihood values to the parameter
ξ.ll = [ll]

niters = 10000
burn   = 2000

σ_keep = fill(0.0, niters, σ.length)
ξ_keep = fill(0.0, niters, ξ.length)
@time for iter = 1:niters
  updatemhseq!(σ)
  updatemhseq!(ξ)

  if iter < (burn / 2)
    updatestepsize!(σ)
    updatestepsize!(ξ)
  end

  σ_keep[iter, :] = σ.cur
  ξ_keep[iter, :] = ξ.cur
end

using Gadfly
plot(x=1:niters, y=σ_keep[:, 1], Geom.line)
plot(x=1:niters, y=ξ_keep[:, 1], Geom.line)

quantile(σ_keep[2001:10000, 1], [0.025, 0.975])
quantile(ξ_keep[2001:10000, 1], [0.025, 0.975])