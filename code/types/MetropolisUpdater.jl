using Distributions
import Base.LinAlg.BLAS.gemv!
push!(LOAD_PATH, homedir()"/repos-git/extremes-project/code/types")
import DataTransformations
import MCMCCalculated
import MHParameters

# testing

# generate covariates and testing data
# simple example
# model y = XB + e
# e ~ N(0, 1)
# priors:
#   beta ~ N(0, 100)


x = reshape(rand(Normal(), 45), 45)
x = hcat(ones(45), x)
beta_t = [3., -1.]
x_beta_t = x * beta_t
y = x_beta_t + rand(Normal(), 45)

# functions to update calculated values
function updatexbeta!(xβ::CalculatedValuesVector, x::Matrix, β::MetropolisVector)
  gemv!('N', 1.0, x, activevalue(β), 0.0, activevalue(xβ))
end

function updatelly!(ll::CalculatedValuesVector, y::Vector,
                    xβ::CalculatedValuesVector)
  for i = 1:ll.length
    activevalue(ll)[i] = logpdf(Normal(activevalue(xβ)[i], 1), y[i])
  end
end

β = createmetropolis(2)
xβ = createcalculatedvalues(45, updater=updatexbeta!,
                            requires=(x, β))
ll = createcalculatedvalues(45, updater=updatelly!,
                            requires=(y, xβ))
β.ll = [ll]  # assign the actual likelihood values to the parameter
β.impacts = [xβ]

niters = 10000
burn   = 2000

β_keep = fill(0.0, niters, β.length)
for iter = 1:niters
  updatemh!(β)

  if iter < (burn / 2)
    updatestepsize!(β)
  end

  β_keep[iter, :] = β.cur
end


# updater(requires...) requires... unpacks the array named requires