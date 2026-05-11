using Makie, CairoMakie
# using Makie, GLMakie
include("../src/SH.jl")    # import source code

function fit_and_p_drop_ellipsoid(a1, a2, a3, ℓₘ, npoints, σ)
    r_fit, ϕ_fit, θ_fit, κ, p_drop_ellipsoid, S_ellipsoid = ellipsoid(a1, a2, a3, npoints, σ)
    Y, dY_dϕ, dY_dθ, d²Y_dϕ², d²Y_dθdϕ, d²Y_dθ², ℓs, ms, one, mone, zero = get_SH_der2(ℓₘ, ϕ_fit, θ_fit)
    c = fit_coefs_LS(Y, r_fit)
    r = Y * c 

    println("Volume: $(volume(c, Y)), actual: $(4. / 3. * π * a1 * a2 * a3)")

    r_test, ϕ_test, θ_test, _, p_drop_accurate, S_accurate = ellipsoid(a1, a2, a3, 16382, σ)
    Y_test = get_SH(ℓₘ, ϕ_test, θ_test)
    errors_rel = abs.(Y_test * c - r_test) ./ r_test 
    println("ℓ=$(ℓₘ), max abs rel error: $(maximum(errors_rel)), mean: $(sum(errors_rel) / length(errors_rel))")

    dr_dϕ = dY_dϕ * c 
    dr_dθ = dY_dθ * c 
    d²r_dϕ² = d²Y_dϕ² * c 
    d²r_dϕdθ = d²Y_dθdϕ * c 
    d²r_dθ² = d²Y_dθ² * c 
    dr_dθ_div_sinϕ, EN_lim = northpole(c, ℓs, one, mone, zero, θ_fit)
    # n_r, n_ϕ, n_θ = unit_normal(r, dr_dϕ, dr_dθ, dr_dθ_div_sinϕ, ϕ_fit)
    p_drop, κ_SH, S = compute_p_drop(r, dr_dϕ, dr_dθ, dr_dθ_div_sinϕ, d²r_dϕ², d²r_dϕdθ, d²r_dθ², EN_lim, ϕ_fit, σ)

    # println(p_drop)
    # println(p_drop_ellipsoid)
    # println(p_drop_accurate)

    V_SH = volume(c, Y)
    V = 4. / 3. * π * a1 * a2 * a3
    V_error = abs(V_SH - V) / V 

    if a1 == a2 && a1 == a3 # sphere
        println("Sphere detected")
        p_drop_accurate = - 2. * σ / a1
        p_error = abs(p_drop - p_drop_accurate) / p_drop_accurate
    else                    # ellipsoid
        p_error = abs(p_drop - p_drop_accurate) / p_drop_accurate
    end
    return V_error, p_error, V, p_drop_accurate
end


################## sphere ##################################################################################################
# σ = 73e-3 # [N/m]
# a1 = 0.00099983445971464261     # to make volume equal to FT
# a2, a3 = a1, a1     # sphere
# ℓₘ = 0
# n_fit = 8066

# V_err, p_err, V, p = fit_and_p_drop_ellipsoid(a1, a2, a3, ℓₘ, n_fit, σ)

# V_FT = 4.1867103085536E-09
# p_FT = -1.4600945786532E+02
# V_err_FT = abs(V_FT - V) / V
# p_err_FT = abs(p_FT - p) / p

# println(V)
# println(V_err)
# println(V_err_FT)
# println(p_err)
# println(p_err_FT)

# tbl = (cat = [1, 1, 2, 2],
#         height = [V_err_FT, -p_err_FT, V_err, -p_err],
#         grp = [1, 2, 1, 2])

# set_theme!(Theme(fontsize = 20))
# colors = Makie.wong_colors()

# println(tbl.height)

# fig = Figure(backgroundcolor = :transparent)
# # ax = Axis(fig[1,1], xticks = (1:2, ["TRI", "SH"]), ylabel = "Relative absolute error", yscale=log10, yticks=LogTicks(-14:-3), 
# #         backgroundcolor = :transparent)

# ax = Axis(fig[1,1], xticks = (1:2, ["TRI", "SH"]), ylabel = L"\frac{\left|[p]_{\text{sim}} - [p]\right|}{\left|[p]\right|}", yscale=log10, yticks=LogTicks(-14:-3), 
#         backgroundcolor = :transparent)
# barplot!(ax, [1, 2], [-p_err_FT, -p_err])
# # barplot!(ax, tbl.cat, tbl.height, dodge = tbl.grp, color = colors[tbl.grp])   

# # labels = [L"V", L"[p]"]
# # elements = [PolyElement(polycolor = colors[i]) for i in 1:length(labels)]
# # Legend(fig[1,1], elements, labels, halign = :right, valign = :top, margin = (10, 10, 10, 10), backgroundcolor = :transparent)

# # resize_to_layout!(fig)
# colsize!(fig.layout, 1, Aspect(1, 1.))
# # display(fig)
# save("Sphere_p_drop_FT_volume.png", fig)
###################################################################################################################################

######### ellipsoid ###############################################################################################################
σ = 73e-3 # [N/m]
a1, a2, a3 = 3e-3 * 0.99994761398402279035, 2e-3 * 0.99994761398402279035, 1e-3 * 0.99994761398402279035
# a1, a2, a3 = 3e-3, 2e-3, 1e-3
ℓₘ = 10
n_fit = 8066

# ℓ_maxs = [3, 5, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
ℓ_maxs = [3, 5, 8, 20]

V_errs = zeros(Float64, length(ℓ_maxs))
p_errs = similar(V_errs)

ref = [0., 0.]

let i = 1;
    # for n_fit = Ns
    for ℓₘ = ℓ_maxs
        V_errs[i], p_errs[i], V, p = fit_and_p_drop_ellipsoid(a1, a2, a3, ℓₘ, n_fit, σ)
        println(ℓₘ)

        if i == length(ℓ_maxs)
            ref[:] .= [V, p]
        end

        i = i + 1
    end
end

V_ref, p_ref = [ref[1], ref[2]]

V_FT = 2.5128791623079E-08
p_FT = -7.8860407278865E+01
V_err_FT = abs(V_FT - V_ref) / V_ref
p_err_FT = abs(p_FT - p_ref) / p_ref

# println(p_ref)
# println(p_FT)
# println(p_err_FT)

set_theme!(Theme(fontsize = 20))

fig = Figure(backgroundcolor = :transparent)

ax = Axis(fig[1, 1], xlabel = "#basis functions", ylabel = L"\frac{\left|[p]_{\text{sim}} - [p]\right|}{\left|[p]\right|}", 
            xscale=log10, yscale=log10, xticks=LogTicks(0:4), yticks=LogTicks(-15:0), backgroundcolor = :transparent,
            xminorticksvisible = true, xminorticks = IntervalsBetween(9)) 

scatterlines!(ax, (ℓ_maxs .+ 1) .^ 2, abs.(p_errs), marker = :circle, linestyle = :dash, label = L"[p]_{\text{SH}}")
hlines!(ax, abs(p_err_FT), linestyle = :dot, label = L"[p]_{\text{TRI}}", color = Cycled(1))
axislegend(ax, position = :lb, backgroundcolor = :transparent)

# ax = Axis(fig[1, 1], xlabel = "#basis functions", ylabel = "Relative absolute error", 
#             xscale=log10, yscale=log10, xticks=LogTicks(0:4), yticks=LogTicks(-15:0), backgroundcolor = :transparent,
#             xminorticksvisible = true, xminorticks = IntervalsBetween(9)) 

# scatterlines!(ax, (ℓ_maxs .+ 1) .^ 2, V_errs, marker = :circle, linestyle = :dash, label = L"V_{\text{SH}}")
# scatterlines!(ax, (ℓ_maxs .+ 1) .^ 2, - p_errs, marker = :circle, linestyle = :dash, label = L"[p]_{\text{SH}}")
# hlines!(ax, V_err_FT, linestyle = :dot, label = L"V_{\text{TRI}}", color = Cycled(1))
# hlines!(ax, - p_err_FT, linestyle = :dot, label = L"[p]_{\text{TRI}}", color = Cycled(2))
# axislegend(ax, position = :lb, backgroundcolor = :transparent)
display(fig)

# save("Ellipsoid_p_drop_8066_V_FT.png", fig)
# save("Ellipsoid_V_p_drop_8066.png", fig)
###################################################################################################################################
