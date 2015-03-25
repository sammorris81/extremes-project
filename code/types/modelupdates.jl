function updatecanxbeta!(obj::CalculatedValuesMatrix)
  for j = 1:obj.cols, i = 1:obj.rows
    obj.value[i, j] = obj.requires["x_beta"][i, j] +
                      obj.requires["can_diff"] * obj.requires["x"][i, j]
  end
end

function getcanbetadiff(obj::MetropolisVector, p::Int64)
  obj.can[p] - obj.cur[p]
end
  # value::Matrix
  # updater::Function
  # requires


# for the metropolis update function
# beta example
#   needs to update canxbeta
#   needs to update z
#   needs to update zstar
#   needs to update lly

