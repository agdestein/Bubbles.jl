using Makie, CairoMakie
# using Makie, GLMakie
include("../src/SH.jl")    # import source code

N = 50
ℓₘ = 3

_, ϕ, θ = get_points_spc(N)

Y, dY_dϕ, dY_dθ, d²Y_dϕ², d²Y_dθdϕ, d²Y_dθ², ℓs, ms, one, mone, zero = get_SH_der2(ℓₘ, ϕ, θ)

hs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]   # roundoff error wins from truncation error at low h
err_t = zeros(length(hs))
err_p = similar(err_t)
err_tt = similar(err_t)
err_pp = similar(err_t)
err_pt = similar(err_t)

let i = 1;
    for h = hs
        Y_ϕ_left = get_SH(ℓₘ, ϕ[2:end-1] .- h, θ[2:end-1])
        Y_ϕ_right = get_SH(ℓₘ, ϕ[2:end-1] .+ h, θ[2:end-1])
        Y_θ_left = get_SH(ℓₘ, ϕ[2:end-1], θ[2:end-1] .- h) 
        Y_θ_right = get_SH(ℓₘ, ϕ[2:end-1], θ[2:end-1] .+ h)
        Y_rr = get_SH(ℓₘ, ϕ[2:end-1] .+ h, θ[2:end-1] .+ h)
        Y_ll = get_SH(ℓₘ, ϕ[2:end-1] .- h, θ[2:end-1] .- h)
        Y_rl = get_SH(ℓₘ, ϕ[2:end-1] .+ h, θ[2:end-1] .- h)
        Y_lr = get_SH(ℓₘ, ϕ[2:end-1] .- h, θ[2:end-1] .+ h)

        Y_ϕ = (Y_ϕ_right - Y_ϕ_left) / (2. * h)
        Y_θ = (Y_θ_right - Y_θ_left) / (2. * h)
        Y_ϕϕ = (Y_ϕ_right - 2. * Y[2:end-1, :] + Y_ϕ_left) / (h ^ 2)
        Y_θθ = (Y_θ_right - 2. * Y[2:end-1, :] + Y_θ_left) / (h ^ 2)
        Y_ϕθ = (Y_rr - Y_rl - Y_lr + Y_ll) / (4. * h ^ 2)

        err_t[i] = sum(abs.((dY_dθ[2:end-1, :] - Y_θ)))
        err_p[i] = sum(abs.((dY_dϕ[2:end-1, :] - Y_ϕ)))
        err_tt[i] = sum(abs.((d²Y_dθ²[2:end-1, :] - Y_θθ)))
        err_pp[i] = sum(abs.((d²Y_dϕ²[2:end-1, :] - Y_ϕϕ)))
        err_pt[i] = sum(abs.((d²Y_dθdϕ[2:end-1, :] - Y_ϕθ)))

        println(Y_ϕϕ[2, 1:10])

        i = i + 1
        println(i)
    end
end

println(d²Y_dϕ²[3, 1:10])

# println(err_p)
# println(err_t)
# println(err_pp)
# println(err_tt)
# println(err_pt)

set_theme!(Theme(fontsize = 20))

fig = Figure()
ax = Axis(fig[1,1], xlabel = "h", ylabel = "Error", xscale=log10, yscale=log10)
scatterlines!(ax, hs, err_p, marker = :circle, linestyle = :dash, label = "ϕ")
scatterlines!(ax, hs, err_t, marker = :circle, linestyle = :dash, label = "θ")
scatterlines!(ax, hs, err_pp, marker = :circle, linestyle = :dash, label = "ϕϕ")
scatterlines!(ax, hs, err_tt, marker = :circle, linestyle = :dash, label = "θθ")
scatterlines!(ax, hs, err_pt, marker = :circle, linestyle = :dash, label = "ϕθ")
axislegend(ax, position = :rt)
display(fig)
