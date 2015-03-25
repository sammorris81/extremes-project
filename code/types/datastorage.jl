abstract LikelihoodValues

# these are the building blocks for the updated likelihood
# need the updater function
abstract CalculatedValues <: LikelihoodValues

type CalculatedValuesScalar <: CalculatedValues
  can::Float64
  cur::Float64
  updater::Function
  requires
  length::Int64

  CalculatedValuesScalar() = new()
end

type CalculatedValuesVector <: CalculatedValues
  can::Vector
  cur::Vector
  updater::Function
  requires
  length::Int64_

  CalculatedValuesVector() = new()
end

type CalculatedValuesMatrix <: CalculatedValues
  can::Matrix
  cur::Matrix
  updater::Function
  requires
  rows::Int64
  cols::Int64

  CalculatedValuesMatrix() = new()
end

function calculatedvalues(initial::Float64, updater::Function, requires)

  CalculatedValuesScalar(initial, updater, requires, 1)
end

function calculatedvalues(initial::Vector, updater::Function, requires)

  CalculatedValuesVector(initial, updater, requires, size(value)[1])
end

function calculatedvalues(initial::Matrix, updater::Function, requires)

  CalculatedValuesMatrix(initial, updater, initialize, requires,
                         size(value)[1], size(value)[2])
end

function fill!(obj::CalculatedValuesMatrix, fill_with::Matrix)
  obj.can = copy(fill_with)
  obj.cur = copy(fill_with)
  obj.nrows, obj.ncols = size(fill_with)
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
x_beta = CalculatedValuesMatrix()
z      = CalculatedValuesMatrix()
z_star = CalculatedValuesMatrix()
lly    = CalculatedValuesMatrix()
llps   = CalculatedValuesMatrix()

  # can::Matrix
  # cur::Matrix
  # updater::Function
  # requires
  # rows::Int64
  # cols::Int64