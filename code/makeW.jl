# Arguments:
#   dw2(ns x nknots): distance matrix
#   rho(1): bandwidth parameter
# Return:
#   w(ns x nknots): kernel weights
function makeW(dw2::Array{Float64}, rho::Float64)
  ns = size(dw2)[1]
  nknots = size(dw2)[2]

  w = fill(0.0, ns, nknots)
  for j=1:nknots
    for i=1:ns
      w[i, j] = exp(-0.5 * dw2[i, j] / (rho^2))
    end
  end

  return w
end