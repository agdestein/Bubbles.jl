using Makie, CairoMakie
# using Makie, GLMakie
include("../src/SH.jl")    # import source code

a1, a2, a3 = 3., 2., 1.
# ℓₘ = 22
n_fit = 8066

r_test, ϕ_test, θ_test = ellipsoid(a1, a2, a3, 16382)
# Y_test = get_SH(ℓₘ, ϕ_test, θ_test)
# x_test, y_test, z_test = spc2cart(r_test, ϕ_test, θ_test)

# Ns = [50, 201, 513, 1059, 2049, 4051, 8066]
ℓs = [3, 5, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
# err_max = zeros(Float64, length(Ns))
err_max = zeros(Float64, length(ℓs))
err_mean = similar(err_max)

let i = 1;
    # for n_fit = Ns
    for ℓₘ = ℓs
        r_fit, ϕ_fit, θ_fit = ellipsoid(a1, a2, a3, n_fit)
        Y = get_SH(ℓₘ, ϕ_fit, θ_fit)
        c = fit_coefs_LS(Y, r_fit)

        Y_test = get_SH(ℓₘ, ϕ_test, θ_test)

        r = Y_test * c 
        errors_rel = abs.(r - r_test) ./ r_test 
        println("ℓ=$(ℓₘ), max abs rel error: $(maximum(errors_rel)), mean: $(sum(errors_rel) / length(errors_rel))")
        err_max[i] = maximum(errors_rel)
        err_mean[i] = sum(errors_rel) / length(errors_rel)

        i = i + 1
    end
end

set_theme!(Theme(fontsize = 20))

# fig = Figure()
# ax1 = Axis3(fig[1, 1])
# ax2 = Axis3(fig[1, 2])
# scatter!(ax1, x_test, y_test, z_test)
# scatter!(ax2, x, y, z)
# display(fig)

fig = Figure(backgroundcolor = :transparent)
# ax = Axis(fig[1, 1], xlabel = L"N$_{\text{fit}}$ [-]", ylabel = L"$\frac{|r_{\text{fit}} - r|}{r}$ [-]", xscale=log10, yscale=log10) #xticks=LogTicks(2:4) to have only integer power ticks
ax = Axis(fig[1, 1], xlabel = "#basis functions [-]", ylabel = L"$\frac{|r_{\text{fit}} - r|}{r}$ [-]", 
            xscale=log10, yscale=log10, xticks=LogTicks(1:3), yticks=LogTicks(-8:1), backgroundcolor = :transparent) 
# scatterlines!(ax, Ns, err_max, marker = :circle, linestyle = :dash, label = "Max")
# scatterlines!(ax, Ns, err_mean, marker = :circle, linestyle = :dash, label = "Mean")
scatterlines!(ax, (ℓs .+ 1) .^ 2, err_max, marker = :circle, linestyle = :dash, label = "Max")
scatterlines!(ax, (ℓs .+ 1) .^ 2, err_mean, marker = :circle, linestyle = :dash, label = "Mean")
axislegend(ax, position = :rt)
# display(fig)
save("Fit_ellipsoid_8066_3_2_1.png", fig)