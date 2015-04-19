using Distributions
import Base.LinAlg.BLAS.gemv!
using DataTransformations
using MetropolisUpdaters

# Simulated example - Simple Linear Regression
# model y = Xβ + e
#   e ~ N(0, 1) i.i.d.
# priors:
#   β ~ N(0, 100) i.i.d.

n = 200
X = reshape(rand(Normal(), n), n)
X = hcat(ones(n), X)
βₐ = [3., -1.]
Xβₐ = X * βₐ
y = Xβₐ + rand(Normal(), n)

# functions to update calculated values
function updatexbeta!(Xβ::CalculatedValuesVector, X::Matrix, β::MetropolisVector)
  gemv!('N', 1.0, X, activevalue(β), 0.0, activevalue(Xβ))
end

function updatelly!(ll::CalculatedValuesVector, y::Vector,
                    Xβ::CalculatedValuesVector)
  for i = 1:ll.length
    activevalue(ll)[i] = logpdf(Normal(activevalue(Xβ)[i], 1), y[i])
  end
end

β = createmetropolis(2)
Xβ = createcalculatedvalues(n, updater=updatexbeta!,
                            requires=(X, β))
ll = createcalculatedvalues(n, updater=updatelly!,
                            requires=(y, Xβ))
β.ll = [ll]  # assign the actual likelihood values to the parameter
β.impacts = [Xβ]

niters = 10000
burn   = 2000

β_keep = fill(0.0, niters, β.length)
for iter = 1:niters
  updatemhseq!(β)

  if iter < (burn / 2)
    updatestepsize!(β)
  end

  β_keep[iter, :] = β.cur
end

using Gadfly
plot(x=1:niters, y=β_keep[:, 1], Geom.line)
plot(x=1:niters, y=β_keep[:, 2], Geom.line)

# updater(requires...) requires... unpacks the array named requires
