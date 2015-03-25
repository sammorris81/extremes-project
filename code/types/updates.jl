using Distributions

abstract ParameterType

abstract MetropolisParameter <: ParameterType

# would be nice to have more flexibility in acceptance and rejections
# currently parameter support must be the same for all rvs in a parameter
# would be nice to have non-normal candidates

type MetropolisScalar <: MetropolisParameter
  can::Float64  # candidate value
  cur::Float64  # current value
  support::Vector  # parameter support

  canddist::Type{Normal}  # candidate distribution
  priordist::Distribution  # prior distribution
  transform::Function      # should there be a transformation

  acc::Int64    # number of acceptances
  att::Int64    # number of attempts
  mh::Float64   # current metropolis standard deviation
end

type MetropolisVector <: MetropolisParameter
  can::Vector  # candidate value
  cur::Vector  # current value
  support::Vector  # parameter support

  canddist::Type{Normal}   # candidate distribution
  priordist::Distribution  # prior distribution
  transform::Function      # should there be a transformation

  acc::Vector  # number of acceptances
  att::Vector  # number of attempts
  mh::Vector   # current metropolis standard deviation
end

type MetropolisMatrix <: MetropolisParameter
  can::Matrix  # candidate value
  cur::Matrix  # current value
  support::Vector  # parameter support

  canddist::Type{Normal}   # candidate distribution
  priordist::Distribution  # prior distribution
  transform::Function      # should there be a transformation

  acc::Matrix  # number of acceptances
  att::Matrix  # number of attempts
  mh::Matrix   # current metropolis standard deviation
end


function updatemh!(obj::MetropolisScalar, nattempts=50, lower=0.8, higher=1.2)
  if obj.att > nattempts
    acc_rate = obj.acc / obj.att
    if acc_rate < 0.25
      obj.mh *= lower
    elseif acc_rate > 0.50
      obj.mh *= higher
    end
    obj.acc = 0
    obj.att = 0
  end
end

function updatemh!(obj::MetropolisVector, nattempts=50, lower=0.8, higher=1.2)
  for i = 1:length(obj.can)
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
  nrows = size(obj)[1]
  ncols = size(obj)[2]
  for j = 1:ncols, i = 1:nrows
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
function drawcandidate!(obj::MetropolisScalar)
  if obj.transform == transident
    cur_star = obj.cur
  else  # transform candidate
    cur_star = obj.transform(obj.cur, obj.support, false)
  end

  can_star = rand(obj.canddist(cur_star, obj.mh), 1)[1]

  if obj.transform == transident
    obj.can = can_star
  else  # transform candidate
    obj.can = obj.transform(can_star, obj.support, true)
  end

  obj.att += 1
end

function drawcandidate!(obj::MetropolisVector, i::Integer)
  if obj.transform == transident
    cur_star = obj.cur[i]
  else  # transform candidate
    cur_star = obj.transform(obj.cur[i], obj.support, false)
  end

  can_star = rand(obj.canddist(cur_star, obj.mh[i]), 1)[1]

  if obj.transform == transident
    obj.can[i] = can_star
  else  # transform candidate
    obj.can[i] = obj.transform(can_star, obj.support, true)
  end

  obj.att[i] += 1
end

function drawcandidate!(obj::MetropolisMatrix, i::Integer, j::Integer)
  if obj.transform == transident
    cur_star = obj.cur[i, j]
  else  # transform candidate
    cur_star = obj.transform(obj.cur[i, j], obj.support, false)
  end

  can_star = rand(obj.canddist(cur_star, obj.mh[i, j]), 1)[1]

  if obj.transform == transident
    obj.can[i, j] = can_star
  else  # transform candidate
    obj.can[i, j] = obj.transform(can_star, obj.support, true)
  end

  obj.att[i, j] += 1
end

# update the parameter
function updatecurrent!(obj::MetropolisScalar)
  obj.cur = obj.can
  obj.acc += 1
end

function updatecurrent!(obj::MetropolisVector, i::Integer)
  obj.cur[i] = obj.can[i]
  obj.acc[i] += 1
end

function updatecurrent!(obj::MetropolisMatrix, i::Integer, j::Integer)
  obj.cur[i, j] = obj.can[i, j]
  obj.acc[i, j] += 1
end


# testing
using Distributions
canddist = Distributions.Normal
prior = Distributions.Uniform(0, 10)
rho = MetropolisScalar(5.0, 5.0, [0, 10],
                        canddist, prior, translogit,
                        10, 100, 0.1)
rho.transform(rho.can, rho.support, false)
@time drawcandidate!(rho)
updatemh!(rho)

