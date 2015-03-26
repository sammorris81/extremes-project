abstract LikelihoodValues

# these are the building blocks for the updated likelihood
# need the updater function
abstract CalculatedValues <: LikelihoodValues

type CalculatedValuesScalar <: CalculatedValues
  can::Real
  cur::Real
  updater::Function
  requires
  length::Integer

  CalculatedValuesScalar() = new()
end

type CalculatedValuesVector <: CalculatedValues
  can::Vector
  cur::Vector
  updater::Function
  requires
  length::Integer

  CalculatedValuesVector() = new()
end

type CalculatedValuesMatrix <: CalculatedValues
  can::Matrix
  cur::Matrix
  updater::Function
  requires
  nrows::Integer
  ncols::Integer

  CalculatedValuesMatrix() = new()
end

# initializes object and give size
function createcalculatedvalues(length::Integer)
  if length == 1
    this = CalculatedValuesScalar()
  else
    this = CalculateValuesVector()
  end
  this.length = length

  return this
end

function createcalculatedvalues(nrows::Integer, ncols::Integer)
  this = CalculatedValuesMatrix()
  this.nrows, this.ncols = nrows, ncols

  return this
end

# fill candidate with a single value
function fillcan!(obj::CalculatedValuesMatrix, fill_with::Real)
  fill!(fill_with, obj.can)

  return
end

# fill current with a single value
function fillcur!(obj::CalculatedValuesMatrix, fill_with::Real)
  fill!(fill_with, obj.cur)

  return
end

function initialize!(obj::CalculatedValuesMatrix, fill_with::Real)
  obj.can = fill(fill_with, obj.nrows, obj.ncols)
  obj.cur = fill(fill_with, obj.nrows, obj.ncols)

  return
end

# these don't change throughout the MCMC
type DataValues <: LikelihoodValues

end


function getzcan!(z::CalculatedValuesMatrix, xi::MetropolisScalar,
                  beta::MetropolisVector, x_beta::CalculatedValuesMatrix,
                  thresh=0.0)
  if xi.updating
    this_xi = xi.can
  else
    this_xi = xi.cur
  end

  if beta.updating
    this_xbeta = x_beta.can
  else
    this_xbeta = x_beta.cur
  end

  if this_xi != 0
    for j = 1:z.nrows, i = 1:z.cols
      z.can[i, j] = (1 + this_xi * (thresh - this_xbeta[i, j]))^(1 / this_xi)
    end
  else
    for j = 1:z.nrows, i = 1:z.cols
      z.can[i, j] = exp(thresh - this_xbeta[i, j])
    end
  end
end

function getzstarcan!(z_star::CalculatedValuesMatrix, z::CalculatedValuesMatrix,
                      alpha::MetropolisScalar, beta::MetropolisVector,
                      xi::MetropolisScalar)
  if alpha.updating
    this_alpha_inv = 1 / alpha.can
  else
    this_alpha_inv = 1 / alpha.cur
  end

  if beta.updating || xi.updating
    this_z = z.can
  else
    this_z = z.cur
  end

  for j = 1:z_star.nrows, i = 1:z_star.ncols
    z_star.can[i, j] = this_z[i, j]^(this_alpha_inv)
  end

end

# testing
x_beta = createcalculatedvalues(10, 15)
initialize!(x_beta, 0.0)
z      = createcalculatedvalues(10, 15)
initialize!(z, 0.0)
z_star = createcalculatedvalues(10, 15)
initialize!(z_star, 0.0)
lly    = createcalculatedvalues(10, 15)
initialize!(lly, 0.0)
llps   = createcalculatedvalues(10, 15)
initialize!(llps, 0.0)

  # can::Matrix
  # cur::Matrix
  # updater::Function
  # requires
  # rows::Int64
  # cols::Int64