# Arguments:
#   xi(1): shape parameter
#   x_beta(ns x nt): mean function
# Return:
#   z(ns x nt): latent variable with to unit Fréchet margins
function getZ(xi::Float64, x_beta::Array{Float64})
  ns = size(x_beta)[1]
  nt = size(x_beta)[2]

  z = fill(0.0, ns, nt)
  if xi != 0
    for j=1:nt
      for i=1:ns
        z[i, j] = (1 + xi * x_beta[i, j])^(1 / xi)
      end
    end
  else
    for j=1:nt
      for i=1:ns
        z[i, j] = exp(x_beta[i, j])
      end
    end
  end

  return z
end