using Distributions
push!(LOAD_PATH, homedir()"/repos-git/extremes-project/code/types")
using DataTransformations

abstract ParameterType

abstract MetropolisParameter <: ParameterType

# would be nice to have more flexibility in acceptance and rejections
# currently parameter support must be the same for all rvs in a parameter
# would be nice to have non-normal candidates

# originally had MetropolisScalar, but am going to use MetropolisVector
# where it's a vector of length 1. This will allow me to iterate over
# the elements using one set of updates for scalar and vectors

# type MetropolisScalar <: MetropolisParameter
#   can::Real   # candidate value
#   cur::Real   # current value
#   can_ll::Array  # candidate values log likelihood
#   cur_ll::Array  # current values for loglikelihood
#   support::Vector  # parameter support
#   length::Integer  # for consistency with vectors

#   canddist::Normal     # candidate distribution
#   prior::Distribution  # prior distribution
#   transform::Function  # should there be a transformation
#   requires::Array      # array of everything the updatefxns requires

#   # want to store the name of elements the parameter impacts
#   # need both candidate and current versions
#   # also want to know which functions we use to update impacted elements
#   can_impacts::Array
#   cur_impacts::Array
#   updatefxns::Array

#   acc::Integer    # number of acceptances
#   att::Integer    # number of attempts
#   mh::Real   # current metropolis standard deviation

#   updating::Bool

#   MetropolisScalar() = new()
# end

type MetropolisVector <: MetropolisParameter
  can::Vector      # candidate value
  cur::Vector      # current value
  ll::Array        # values of log likelihood
  # cur_ll::Array  # values of log likelihood
  support::Vector  # parameter support
  length::Integer

  canddist::Normal     # candidate distribution
  prior::Distribution  # prior distribution
  transform::Function  # should there be a transformation
  impacts::Array       # array of everything the update impacts

  acc::Vector  # number of acceptances
  att::Vector  # number of attempts
  mh::Vector   # current metropolis standard deviation

  updating::Bool

  MetropolisVector() = new()
end

type MetropolisMatrix <: MetropolisParameter
  can::Matrix      # candidate value
  cur::Matrix      # current value
  ll::CalculatedValues
  # can_ll::Array    # values of log likelihood
  # cur_ll::Array    # values of log likelihood
  support::Vector  # parameter support
  nrows::Integer
  ncols::Integer

  canddist::Normal     # candidate distribution
  prior::Distribution  # prior distribution
  transform::Function  # should there be a transformation
  impacts       # array of everything the update impacts

  acc::Matrix  # number of acceptances
  att::Matrix  # number of attempts
  mh::Matrix   # current metropolis standard deviation

  updating::Bool

  MetropolisMatrix() = new()
end

function fillcan!(obj::MetropolisParameter, fill_with::Real)
  fill!(obj.can, fill_with)
end

function fillcur!(obj::MetropolisParameter, fill_with::Real)
  fill!(obj.cur, fill_with)
end

function initialize!(obj::MetropolisVector, fill_with::Real)
  obj.can = fill(fill_with, obj.length)
  obj.cur = fill(fill_with, obj.length)

  obj.acc = fill(0, obj.length)
  obj.att = fill(0, obj.length)
  return
end

function initialize!(obj::MetropolisMatrix, fill_with::Real)
  obj.can = fill(fill_with, obj.nrows, obj.ncols)
  obj.cur = fill(fill_with, obj.nrows, obj.ncols)

  obj.acc = fill(0, obj.nrows, obj.ncols)
  obj.att = fill(0, obj.nrows, obj.ncols)
  return
end

# create a metropolis parameter that's a scalar or vector
# to simplify the update, we treat scalars as arrays of length 1
function createmetropolis(length::Integer; initial::Real=0.0,
                          prior::Distribution=Distributions.Normal(),
                          support::Vector=[-Inf, Inf],
                          transform::Function=transident, tune::Real=1.0)

  this = MetropolisVector()
  this.can = fill(convert(FloatingPoint, initial), length)
  this.cur = fill(convert(FloatingPoint, initial), length)

  this.support    = support
  this.length     = length
  this.canddist   = Distributions.Normal()
  this.prior      = prior
  this.transform  = transform
  this.acc        = fill(0, length)
  this.att        = fill(0, length)
  this.mh         = fill(tune, length)
  this.updating   = false

  return this
end

# create a metropolis parameter that's a matrix
function createmetropolis(nrows::Integer, ncols::Integer; initial::Real=0.0,
                          prior::Distribution=Distributions.Normal(),
                          support::Vector=[-Inf, Inf],
                          transform::Function=transident, tune::Real=1.0)
  this = MetropolisMatrix()
  this.can = fill(convert(FloatingPoint, initial), nrows, ncols)
  this.cur = fill(convert(FloatingPoint, initial), nrows, ncols)

  this.support    = support
  this.nrows      = nrows
  this.ncols      = ncols
  this.canddist   = Distributions.Normal()
  this.prior      = prior
  this.transform  = transform
  this.acc        = fill(0, nrows, ncols)
  this.att        = fill(0, nrows, ncols)
  this.mh         = fill(tune, nrows, ncols)
  this.updating   = false

  return this
end

# get either the current or candidate values from a metropolis parameter
# used in the update functions
function activevalue(obj::MetropolisParameter)
  if obj.updating
    return obj.can
  else
    return obj.cur
  end
end

function addlikelihood!(obj::MetropolisParameter, ll::CalculatedValues)
  obj.can_ll = ll.can
  obj.cur_ll = ll.cur
end

# update the candidate standard deviation
function updatemh!(obj::MetropolisVector, nattempts=50, lower=0.8, higher=1.2)
  for i = 1:obj.length
    if obj.att[i] > nattempts
      acc_rate = obj.acc[i] / obj.att[i]
      if acc_rate < 0.25
        obj.mh[i] *= lower
      elseif acc_rate > 0.50
        obj.mh[i] *= higher
      end
      obj.acc[i] = 0
      obj.att[i] = 0
    end
  end
end

function updatemh!(obj::MetropolisMatrix, nattempts=50, lower=0.8, higher=1.2)
  for j = 1:obj.ncols, i = 1:obj.nrows
    if obj.att[i, j] > nattempts  # avoid divide by 0
      acc_rate = obj.acc[i, j] / obj.att[i, j]
      if acc_rate < 0.25
        obj.mh[i, j] *= lower
      elseif acc_rate > 0.50
        obj.mh[i, j] *= higher
      end
      obj.acc[i, j] = 0
      obj.att[i, j] = 0
    end
  end
end

# draw from the candidate distribution
function drawcandidate!(obj::MetropolisVector, i::Integer)
  if obj.transform == transident
    cur_star = obj.cur[i]
  else  # transform candidate
    cur_star = obj.transform(obj.cur[i], obj.support, false)
  end

  can_star = cur_star + obj.mh[i] * rand(obj.canddist, 1)[1]

  if obj.transform == transident
    obj.can[i] = can_star
  else  # transform candidate
    obj.can[i] = obj.transform(can_star, obj.support, true)
  end
end

function drawcandidate!(obj::MetropolisMatrix, i::Integer, j::Integer)
  if obj.transform == transident
    cur_star = obj.cur[i, j]
  else  # transform candidate
    cur_star = obj.transform(obj.cur[i, j], obj.support, false)
  end

  can_star = cur_star + obj.mh[i, j] * rand(obj.canddist, 1)[1]

  if obj.transform == transident
    obj.can[i, j] = can_star
  else  # transform candidate
    obj.can[i, j] = obj.transform(can_star, obj.support, true)
  end

  obj.att[i, j] += 1
end

# update the parameter
function updatecurrent!(obj::MetropolisVector, i::Integer)
  obj.cur[i] = obj.can[i]
  obj.acc[i] += 1
end

function updatecurrent!(obj::MetropolisMatrix, i::Integer, j::Integer)
  obj.cur[i, j] = obj.can[i, j]
  obj.acc[i, j] += 1
end

# running the update for the metropolis parameter
function updatestep!(obj::MetropolisVector)
  obj.updating = true  # tell the object it's currently updating

  for i = 1:obj.length
    obj.att[i] += 1
    drawcandidate!(obj, i)  # get candidate value

    for j = 1:length(obj.impacts)
      this_impact = obj.impacts[j]  # get the name of the object impacted
      this_impact.updating = true
      this_impact.updater(this_impact, this_impact.requires...)  # candidate
    end

    # get candidate ll
    for j = 1:length(obj.ll)
      this_ll = obj.ll[j]
      this_ll.updating = true
      this_ll.updater(this_ll, this_ll.requires...)
      this_ll.updating = false
    end

    # log ratio for acceptance
    R = 0
    for j = 1:length(obj.ll)
      R += sum(obj.ll[j].can) - sum(obj.ll[j].cur)
    end

    # update the acceptance ratio with the logprior values
    R += logpdf(obj.prior, obj.can[i]) - logpdf(obj.prior, obj.cur[i])

    # decide whether or not to accept or reject
    if (log(rand(1)[1]) < R)
      obj.acc[i] += 1
      copy!(obj.cur, obj.can)  # to copy over the candidate to current values

      # update the likelihood storage
      for j = 1:length(obj.ll)
        copy!(obj.ll[j].cur, obj.ll[j].can)
      end

      # update the elements that the parameter impacts
      for j = 1:length(obj.impacts)
        copy!(obj.impacts[j].cur, obj.impacts[j].can)
      end

      for j = 1:length(obj.impacts)
        obj.impacts[j].updating = false
      end
    end

  end
  obj.updating = false
end

# testing

# generate covariates and testing data
# simple example
# model y = XB + e
# e ~ N(0, 1)
# priors:
#   beta ~ N(0, 100)
using Distributions
import Base.LinAlg.BLAS.gemv!

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
  updatestep!(β)

  if iter < (burn / 2)
    updatemh!(β)
  end

  β_keep[iter, :] = β.cur
end


# updater(requires...) requires... unpacks the array named requires



