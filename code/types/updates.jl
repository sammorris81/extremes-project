using Distributions

abstract ParameterType

abstract MetropolisParameter <: ParameterType

# would be nice to have more flexibility in acceptance and rejections
# currently parameter support must be the same for all rvs in a parameter
# would be nice to have non-normal candidates

type MetropolisScalar <: MetropolisParameter
  can::Real   # candidate value
  cur::Real   # current value
  can_ll::Array  # candidate values log likelihood
  cur_ll::Array  # current values for loglikelihood
  support::Vector  # parameter support
  length::Integer  # for consistency with vectors

  canddist::Type{Normal}  # candidate distribution
  prior::Distribution     # prior distribution
  transform::Function     # should there be a transformation
  requires::Array         # array of everything the updatefxns requires

  # want to store the name of elements the parameter impacts
  # need both candidate and current versions
  # also want to know which functions we use to update impacted elements
  can_impacts::Array
  cur_impacts::Array
  updatefxns::Array

  acc::Integer    # number of acceptances
  att::Integer    # number of attempts
  mh::Real   # current metropolis standard deviation

  updating::Bool

  MetropolisScalar() = new()
end

type MetropolisVector <: MetropolisParameter
  can::Vector    # candidate value
  cur::Vector    # current value
  can_ll::Array  # candidate values log likelihood
  cur_ll::Array  # current values for loglikelihood
  support::Vector  # parameter support
  length::Integer

  canddist::Type{Normal}  # candidate distribution
  prior::Distribution     # prior distribution
  transform::Function     # should there be a transformation
  requires::Array         # array of everything the update requires
  updatefxns::Array       # array of function names for each update

  acc::Vector  # number of acceptances
  att::Vector  # number of attempts
  mh::Vector   # current metropolis standard deviation

  updating::Bool

  MetropolisVector() = new()
end

type MetropolisMatrix <: MetropolisParameter
  can::Matrix    # candidate value
  cur::Matrix    # current value
  can_ll::Array  # candidate values log likelihood
  cur_ll::Array  # current values for loglikelihood
  support::Vector  # parameter support
  nrows::Integer
  ncols::Integer

  canddist::Type{Normal}  # candidate distribution
  prior::Distribution     # prior distribution
  transform::Function     # should there be a transformation
  requires::Array         # array of everything the update requires
  updatefxns::Array       # array of function names for each update

  acc::Matrix  # number of acceptances
  att::Matrix  # number of attempts
  mh::Matrix   # current metropolis standard deviation

  updating::Bool

  MetropolisMatrix() = new()
end

function createmetropolis(length::Integer)
  if length == 1
    this = MetropolisScalar()
  else
    this = MetropolisVector()
  end

  this.length = length
  return this
end

function createmetropolis(nrows::Integer, ncols::Integer)
  this = MetropolisMatrix()
  this.nrows = nrows
  this.ncols = ncols

  return this
end

function fillcan!(obj::MetropolisParameter, fill_with::Real)
  fill!(obj.can, fill_with)
end

function fillcur!(obj::MetropolisParameter, fill_with::Real)
  fill!(obj.cur, fill_with)
end

function initialize!(obj::MetropolisParameter, fill_with::Real)
  obj.can = fill(fill_with, obj.nrows, obj.ncols)
  obj.cur = fill(fill_with, obj.nrows, obj.ncols)

  return
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

function updatemh!(obj::MetropolisScalar)
  obj.updating = true
  drawcandidate!(obj)

  for i = 1:length(obj.impacts)
    can_impact = obj.can_impacts[i]  # get the name of the object impacted
    can_impact.updater(can_impact.requires)  # update the item
  end

  R = 0
  for i = 1:length(obj.can_ll)
    R += sum(obj.can_ll[i]) - sum(obj.cur_ll[i])
  end
  R += logpdf(obj.prior, obj.can) - logpdf(obj.prior, obj.cur)

  if (log(rand(1)[1]) < R)
    updatecurrent!(obj)
    for i = 1:length(obj.can_ll)
      copy!(obj.cur_ll[i], obj.can_ll[i])
    end
    for i = 1:length(obj.impacts)
      copy!(obj.cur_impacts[i], obj.can_impacts[i])
    end
  end

  obj.updating = false
end

# testing
using Distributions
lly_only = {lly}
llps_only = {llps}
ll_both = {lly, llps}

canddist = Distributions.Normal

beta = createmetropolis(3)
xi = createmetropolis(1)
A = createmetropolis(10, 12)
alpha = createmetropolis(1)
rho = createmetropolis(1)

xi_can_impacts = {can_z_star}
xi_cur_impacts = {z_star}
xi_update_fxns = {getz, getzstar, logpdf_rarebinary!}



xi = MetropolisScalar(0.1, 0.1, [-1, 1],
                      0, 0, 0.1,
                      Distributions.Normal, Distributions.Normal(0, 1),
                      translogit,
                      can_lly_only, cur_lly_only,
                      xi_can_impacts, xi_cur_impacts,
                      xi_update_fxns, xi_requires
                      )
xi.transform(xi.can, xi.support, false)
@time drawcandidate!(xi)
updatemh!(xi)


