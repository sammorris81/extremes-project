include("derivative.jl")
function quadratic(f)
  # compute first derivative
  f1 = derivative(f)

  # Get the y intercept
  c = f(0.00)

  # get the y intercept of the first derivative
  b = f1(0.0)

  a = f(1.0) - b - c

  return (-b + sqrt(b^2 - 4a * c + 0im)) / 2a, (-b - sqrt(b^2 - 4a * c + 0im)) / 2a
end

