# when looping over entries of a matrix, rows of a column are contiguous
# columns of a row are not
for j = 1:50, i = 1:40
  a[i, j] = i + j
end
# will be faster than
for i = 1:40, j = 1:50
  a[i, j] = i + j
end

# if you will be filling an array with calculated values, it is
# quicker to pre-allocate the array and loop over it using
a = Array(Float64, 40, 50)
for j = 1:50, i = 1:40
  a[i, j] = i::Float64 + j::Float64
end

# create Array{Int64, 40, 50} or Array{Int32, 40, 50} filled with 1
y = fill(1, 40, 50)

# creates Array{Float64, 50, 3} or Array{Float32, 50, 3} filled with 1.0
x = fill(1.0, 40, 50, 3)
for j = 1:50, i = 1:40
    x[i, j, 2:3] = rand(Normal(0, 1), 2)
end

# julia does not drop a dimension from an array when it's size is 1
# if you need to turn a 3D array to 2D use reshape
# NOTE: reshape will copy the array slice not reference the original
reshape(x[:, t, :], ns, 3)
for t = 1:nt
  can_x_beta[:, t] = reshape(x[:, t, :], ns, 3) * beta
end

# as long as you've already initialized the distribution
rand(Normal(0, 1), 1)
# takes as long as
temp = Normal(0, 1)
rand(temp, 1)

# julia uses a hash table in place of lists in R
results = {"x" => x, "y" => y, "can_x_beta" => can_x_beta}
results["can_x_beta"]
# is the same as
results <- list(x=x, y=y, can_x_beta=can_x_beta)
results$can_x_beta

# to get rowsums use
sum(x, 2)
# to get colsums use
sum(x, 1)

# julia generates a sequence using
x = linspace(0.01, 0.99, 10)  # from 0.01 to 0.99 with a length of 10

# you can expand.grid in julia using
x = hcat(linspace(0.01, 0.99, 10), inner=[1], outer=[10],
         linspace(0.01, 0.99, 10), inner=[10], outer=[1])

# duplicating arrays
x = fill(1.0, 100, 100)

# by default julia copies