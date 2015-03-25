using Distributions
# importall

function transident(x::Real)
  x
end

function transunif(x::Real, support::Vector, inv::Bool=false)
  @assert isfinite(support[1])
  @assert isfinite(support[2])
  if !inv
    return (x - support[1]) / (support[2] - support[1])
  else
    @assert x >= 0 && x <= 1
    return x * (support[2] - support[1]) + support[1]
  end
end

function translogit(x::Real, support::Vector, inv::Bool=false)
  @assert isfinite(support[1])
  @assert isfinite(support[2])
  if !inv
    p = transunif(x, support, inv)
    return log(p / (1 - p))
  else
    p = exp(x) / (1 + exp(x))
    return transunif(p, support, inv)
  end
end

function transprobit(x::Real, support::Vector, inv::Bool=false)
  @assert isfinite(support[1])
  @assert isfinite(support[2])
  if !inv
    p = transunif(x, support, inv)
    return quantile(Distributions.Normal(0, 1), p)
  else
    p = cdf(Distributions.Normal(0, 1), x)
    return transunif(p, support, inv)
  end
end

function transcopula(x::Real, dens::Distribution, inv::Bool=false)
  if !inv
    p = cdf(dens, x)
    return quantile(Distributions.Normal(0, 1), p)
  else
    p = cdf(Distributions.Normal(0, 1), x)
    return quantile(dens, p)
  end
end

