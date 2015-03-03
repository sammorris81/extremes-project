println("Hello, World!")

function is_prime(n::Int64)
  if n <= 3
    return true
  end

  if n % 2 == 0
    return false
  end

  # initialize a counter variable
  i = 3

  while i <= sqrt(n)
    if n % i == 0
      return false
    end

    i += 2
  end

  return true
end

is_prime(199)