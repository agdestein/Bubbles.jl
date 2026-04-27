a = 2; b = 3

function sq(a)
    return a, a .^ 2
end

function ad(a, b)
    return a + b 
end

println(sq(a))
println(ad(sq(a)...))