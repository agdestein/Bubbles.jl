include("src/time_stepping.jl")

(; time, itime, coef_names, coefs, com, vcom) = read_bubble_history_log("output/bubble/bubble_history.txt")

println(time[end-3:end])
println(size(com))
println(com[end-3:end, :])
println(coefs[end-3:end, :])

plot(time, coefs[:, 7])
