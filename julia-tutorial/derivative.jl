function derivative(f)
  return function(x)
    h = x == 0 ? sqrt(eps(Float64)) : sqrt(eps(Float64)) * x
    xph = x + h
    dx = xph - x

    f1 = f(xph)
    f0 = f(x)

    return(f1 - f0) / dx
  end
end
