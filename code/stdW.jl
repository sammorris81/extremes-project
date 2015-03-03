# Arguments:
#   w(ns x nknots): kernel weights
# Returns:
#   std_w(ns x nknots): kernel weights standardized to add to 1
function stdW(w::Array{Float64})
  ns = size(x)[1]
  nknots = size(x)[2]

  total_weight = sum(w, 2)
  std_w = fill(1.0, ns, nknots)
  for j=1:nknots
    for i=1:ns
      std_w[i, j] = w[i, j] / total_weight[i]
    end
  end

  return std_w
end
