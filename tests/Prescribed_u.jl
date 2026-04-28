using Makie, CairoMakie
# using Makie, GLMakie
include("../src/SH.jl")    # import source code

function get_uz_spc(c, Y, ϕ, θ)
    r = Y * c 
    u_z = 1. .- (r .* sin.(ϕ) .* cos.(θ) / 2.).^2    # parabolic velocity profile in the z direction
    return u_z
end

function get_uz_cart(x)
    u_z = 1. .- (x / 2.).^2    # parabolic velocity profile in the z direction
    return u_z
end

function get_u_rotz_spc(c, Y, ϕ, θ)
    r = Y * c 
    u_x = - π * r .* sin.(ϕ) .* sin.(θ)
    u_y = π * r .* sin.(ϕ) .* cos.(θ)
    return u_x, u_y 
end

function test_z_rotation(c, centr, ℓₘ, npoints, V, dt, nt)
    # Spherical design cubature points:
    r, ϕ, θ = get_points_spc(npoints)
    r_test, ϕ_test, θ_test = get_points_spc(16382)  # for testing only

    Y, dY_dϕ, dY_dθ, ℓs, ms, one, mone = get_SH_der(ℓₘ, ϕ, θ)   # spherical harmonics (and partial derivatives) at cubature points, their identifiers (ℓ, m) and some specific indices (where m==1 and m==-1)
    Y_test = get_SH(ℓₘ, ϕ_test, θ_test)                         # spherical harmonics at test points

    x0, y0, z0 = spc2cart(r_test, ϕ_test, θ_test)
    x0, y0, z0 = x0 .+ centr[1], y0 .+ centr[2], z0 .+ centr[3]

    Vs = zeros(Float64, nt)

    println("#cubature points: $(length(ϕ)), #testing points: $(length(ϕ_test)), #basis functions: $(nbf), dt: $(dt)")

    for i in 1:nt   # RK4 time stepping
        k1, k1_centr = time_step(c, Y, dY_dϕ, dY_dθ, ϕ, θ, 
                            hcat(get_u_rotz_spc(c, Y, ϕ, θ)..., zeros(Float64, length(ϕ))), npoints, V, ℓs, ms, one, mone)
        k2, k2_centr = time_step(c + k1 * dt / 2., Y, dY_dϕ, dY_dθ, ϕ, θ, 
                            hcat(get_u_rotz_spc(c + k1 * dt / 2., Y, ϕ, θ)..., zeros(Float64, length(ϕ))), npoints, V, ℓs, ms, one, mone)
        k3, k3_centr = time_step(c + k2 * dt / 2., Y, dY_dϕ, dY_dθ, ϕ, θ, 
                            hcat(get_u_rotz_spc(c + k2 * dt / 2., Y, ϕ, θ)..., zeros(Float64, length(ϕ))), npoints, V, ℓs, ms, one, mone)
        k4, k4_centr = time_step(c + k3 * dt, Y, dY_dϕ, dY_dθ, ϕ, θ, 
            hcat(get_u_rotz_spc(c + k3 * dt, Y, ϕ, θ)..., zeros(Float64, length(ϕ))), npoints, V, ℓs, ms, one, mone)

        c = c + (k1 + 2. * k2 + 2. * k3 + k4) * dt / 6.
        # println((k1_centr + 2. * k2_centr + 2. * k3_centr + k4_centr) * dt / 6.)
        centr = centr .+ (k1_centr + 2. * k2_centr + 2. * k3_centr + k4_centr) * dt / 6.

        Vs[i] = volume(c, Y_test)
    end
    
    r = Y_test * c 
    x, y, z = centr[1] .+ r .* sin.(ϕ_test) .* cos.(θ_test), centr[2] .+ r .* sin.(ϕ_test) .* sin.(θ_test), centr[3] .+ r .* cos.(ϕ_test)
    println("Centroid end: $(centr)")
    # ux, uy = get_u_rotz_cart(x0, y0, z0)
    # x_ref = x0 + ux * dt * nt 
    # y_ref = y0 + uy * dt * nt
    # r_ref, ϕ_ref, θ_ref = cart2spc(x_ref, y_ref, z_0)
    # Y_ref = get_SH(maximum(ℓs), ϕ_ref, θ_ref)
    # r_sim = Y_ref * c 
    # error_rel = (r_sim - r_ref) ./ r_ref
    error_rel = (r - r_test) ./ r_test

    ## For visualization:
    n = 100
    ϕs = range(0, π, length = n)
    θs = range(0, 2π, length = n)
    Y_vis = get_SH(maximum(ℓs), ϕs, θs)
    xs, ys, zs = spc2cart(Y_vis * c, ϕs, θs)

    return x0, y0, z0, x, y, z, Vs, error_rel, xs, ys, zs
end

function test_z_translation(c, centr, ℓₘ, npoints, V, dt, nt)
    # Spherical design cubature points:
    r, ϕ, θ = get_points_spc(npoints)
    r_test, ϕ_test, θ_test = get_points_spc(16382)  # for testing only

    Y, dY_dϕ, dY_dθ, ℓs, ms, one, mone = get_SH_der(ℓₘ, ϕ, θ)   # spherical harmonics (and partial derivatives) at cubature points, their identifiers (ℓ, m) and some specific indices (where m==1 and m==-1)
    Y_test = get_SH(ℓₘ, ϕ_test, θ_test)                         # spherical harmonics at test points

    x0, y0, z0 = spc2cart(r_test, ϕ_test, θ_test)
    x0, y0, z0 = x0 .+ centr[1], y0 .+ centr[2], z0 .+ centr[3]

    Vs = zeros(Float64, nt)

    println("#cubature points: $(length(ϕ)), #testing points: $(length(ϕ_test)), #basis functions: $(nbf), dt: $(dt), #t: $(nt)")

    for i in 1:nt   # RK4 time stepping
        k1, k1_centr = time_step(c, Y, dY_dϕ, dY_dθ, ϕ, θ, 
                            hcat(zeros(Float64, length(ϕ)), zeros(Float64, length(ϕ)), get_uz_spc(c, Y, ϕ, θ)), npoints, V, ℓs, ms, one, mone)
        k2, k2_centr = time_step(c + k1 * dt / 2., Y, dY_dϕ, dY_dθ, ϕ, θ, 
                            hcat(zeros(Float64, length(ϕ)), zeros(Float64, length(ϕ)), get_uz_spc(c + k1 * dt / 2., Y, ϕ, θ)), npoints, V, ℓs, ms, one, mone)
        k3, k3_centr = time_step(c + k2 * dt / 2., Y, dY_dϕ, dY_dθ, ϕ, θ, 
                            hcat(zeros(Float64, length(ϕ)), zeros(Float64, length(ϕ)), get_uz_spc(c + k2 * dt / 2., Y, ϕ, θ)), npoints, V, ℓs, ms, one, mone)
        k4, k4_centr = time_step(c + k3 * dt, Y, dY_dϕ, dY_dθ, ϕ, θ, 
            hcat(zeros(Float64, length(ϕ)), zeros(Float64, length(ϕ)), get_uz_spc(c + k3 * dt, Y, ϕ, θ)), npoints, V, ℓs, ms, one, mone)

        c = c + (k1 + 2. * k2 + 2. * k3 + k4) * dt / 6.
        # println((k1_centr + 2. * k2_centr + 2. * k3_centr + k4_centr) * dt / 6.)
        centr[3] = centr[3] .+ (k1_centr[3] + 2. * k2_centr[3] + 2. * k3_centr[3] + k4_centr[3]) * dt / 6.

        Vs[i] = volume(c, Y_test)
    end
    
    r = Y_test * c 
    x, y, z = centr[1] .+ r .* sin.(ϕ_test) .* cos.(θ_test), centr[2] .+ r .* sin.(ϕ_test) .* sin.(θ_test), centr[3] .+ r .* cos.(ϕ_test)
    println("Centroid end: $(centr)")
    z_ref = z0 + get_uz_cart(x0) * dt * nt    # for computing errors
    r_ref, ϕ_ref, θ_ref = cart2spc(x0, y0, z_ref .- centr[3])
    Y_ref = get_SH(maximum(ℓs), ϕ_ref, θ_ref)
    r_sim = Y_ref * c 
    error_rel = (r_sim - r_ref) ./ r_ref

    ## For visualization:
    n = 100
    ϕs = range(0, π, length = n)
    θs = range(0, 2π, length = n)
    Y_vis = get_SH(maximum(ℓs), ϕs, θs)
    xs, ys, zs = spc2cart(Y_vis * c, ϕs, θs)

    return x0, y0, z0, x, y, z, Vs, error_rel, xs, ys, zs
end

# add text_x_profile, and test_rotations (3 of them)

npoints = 4051

V = 4. / 3. * π             # bubble volume, assumed constant

# Initial spherical harmonics coefficients
ℓₘ = 22; nbf = (ℓₘ + 1)^2  # number of basis functions (spherical harmonics)
c0 = zeros(Float64, nbf)
c0[1] = 1. * sqrt(4. * π)   # we start with a unit sphere: only the first spherical harmonic is active; normalized to radius 1.

centr0 = zeros(Float64, 3)  # bubble centroid

dt = 1e-3; nt = Int64(round(1/dt)) # time step, nr of time steps

### Comment/uncomment x vs y test case #############################################################################################################
# Parabolic velocity profile in z direction:
# x0, y0, z0, x, y, z, Vs = parabolic_z(c0, centr0, Y, dY_dϕ, ϕ, θ, uz, uzmax, npoints, V, dt, nt, Y_test, ϕ_test, θ_test)
x0, y0, z0, x, y, z, Vs, errors_rel, xvis, yvis, zvis = test_z_translation(c0, centr0, ℓₘ, npoints, V, dt, nt)
# x0, y0, z0, x, y, z, Vs, errors_rel, xvis, yvis, zvis = test_z_rotation(c0, centr0, ℓₘ, npoints, V, dt, nt)


# Linear velocity profile in x direction:
# x0, y0, z0, x, y, z = linear_x(c0, centr0, Y, dY_dϕ, dY_dθ, ϕ, θ, ux, uxmax, npoints, V, dt, nt, Y_test, ϕ_test, θ_test, ℓs, ms, one, mone)
####################################################################################################################################################

# scatter(x, y, z)

err_max, err_mean, err_min = maximum(abs.(errors_rel)), sum(abs.(errors_rel))/length(errors_rel), minimum(errors_rel)
println("Max error: $(err_max), min: $(err_min), relative volume: $(Vs[end]/V)")
println("Max abs error: $(maximum(abs.(errors_rel))), mean: $(sum(abs.(errors_rel))/length(errors_rel))")

set_theme!(Theme(fontsize = 20))

fig = Figure(backgroundcolor = :transparent)
# ax1 = Axis(fig[1, 1], xlabel = L"$\frac{r_{\text{sim}}-r}{r}$ [-]", ylabel = "Count [-]")
# ax1 = Axis3(fig[1, 1], title = "Max relative abs error: $(round(err_max, sigdigits=3)), mean: $(round(err_mean, sigdigits=3))")
# ax2 = Axis(fig[2, 1], xlabel = "Time [s]", ylabel = L"$\frac{V_{\text{sim}}-V}{V}$ [-]")
# scatter!(ax1, x, y, z, color=errors_rel)
# lines!(ax2, range(dt, 1, nt), Vs / V .- 1)
# axislegend(ax, position = :rb)
# fig

ax = Axis(fig[1,1], xlabel = "Time [s]", ylabel = L"$\frac{V_{\text{sim}}-V}{V}$ [-]", backgroundcolor = :transparent)
scatterlines!(ax, range(dt, 1, nt), Vs / V .- 1, marker = :circle, linestyle = :dash)
# display(fig)
save("Volume_test_dt=$(dt)_ell=$(ℓₘ)_N=$(npoints).png", fig)