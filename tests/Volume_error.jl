using Makie, CairoMakie
include("../src/SH.jl")    # import source code

function volume_error(npoints, ℓₘ, dt, z_x)
    # Spherical design cubature points:
    r, ϕ, θ = get_points_spc(npoints)
    r_test, ϕ_test, θ_test = get_points_spc(16382)  # for testing only

    # Initial spherical harmonics coefficients
    nbf = (ℓₘ + 1)^2  # number of basis functions (spherical harmonics)
    println("# cubature points: $(length(ϕ)), #basis functions: $(nbf), time step: $(dt)")
    c0 = zeros(Float64, nbf)
    c0[1] = 1. * sqrt(4. * π)   # we start with a perfect sphere: only the first spherical harmonic is active; normalized to radius 1.
    centr0 = zeros(Float64, 3)  # bubble centroid

    nt = Int64(round(1/dt)) # time step, nr of time steps

    Y, dY_dϕ, dY_dθ, ℓs, ms, one, mone = get_SH_der(ℓₘ, ϕ, θ)   # spherical harmonics (and partial derivatives) at cubature points, their identifiers (ℓ, m) and some specific indices (where m==1 and m==-1)
    Y_test = get_SH(ℓₘ, ϕ_test, θ_test)
    # Parabolic velocity profile in z direction:
    if z_x
        x0, y0, z0, x, y, z, Vs = parabolic_z(c0, centr0, Y, dY_dϕ, ϕ, θ, uz, uzmax, npoints, V, dt, nt, Y_test, ϕ_test, θ_test)

    # Linear velocity profile in x direction:
    else
        x0, y0, z0, x, y, z, Vs = linear_x(c0, centr0, Y, dY_dϕ, dY_dθ, ϕ, θ, ux, uxmax, npoints, V, dt, nt, Y_test, ϕ_test, θ_test, ℓs, ms, one, mone)
    end

    return Vs
end

# Prescribed velocity field:
uzmax = [0., 0., 1.]        # max velocity in z direction 
uxmax = [1. / 3., 0., 0.]   # max velocity in x direction
C = 2.
uz = x -> 1. .- (x/C).^2    # parabolic velocity profile in the z direction 
ux = y -> C .+ y            # linear velocity profile in the x direction
V = 4. / 3. * π             # bubble volume, assumed constant

## Varying Δt ############################
npoints = 1059; ℓₘ = 10
dts = [1e-1, 1e-2, 1e-3, 1e-4]

fig = Figure(backgroundcolor = :transparent)
ax = Axis(fig[1, 1], xlabel = "Time [s]", ylabel = "Relative error in bubble volume [-]", backgroundcolor = :transparent)

for (i, dt) in enumerate(dts)
    nt_i = Int64(round(1/dt))
    Vs_i = volume_error(npoints, ℓₘ, dt, true)
    lines!(ax, range(dt, 1, nt_i), Vs_i / V, label = "Δt=$(dt)")
end

axislegend(ax, position = :rb)
save("Volume_test_dt.png", fig)
###########################################

# ## Varying npoints ############################
# npointss = [201, 513, 1059, 2049, 4051]; ℓₘ = 10
# dt = 1e-1; nt = Int64(round(1/dt))

# fig = Figure(backgroundcolor = :transparent)
# ax = Axis(fig[1, 1], xlabel = "Time [s]", ylabel = "Relative error in bubble volume [-]", backgroundcolor = :transparent)

# for (i, np) in enumerate(npointss)
#     Vs_i = volume_error(np, ℓₘ, dt, true)
#     lines!(ax, range(dt, 1, nt), Vs_i / V, label = "#points=$(np)")
# end

# axislegend(ax, position = :rb)
# save("Volume_test_ncub.png", fig)
# ###########################################

# ## Varying ℓ_max ############################
# npoints = 4051; ℓₘs = [4, 7, 10, 14, 20]
# dt = 1e-3; nt = Int64(round(1/dt))

# fig = Figure(backgroundcolor = :transparent)
# ax = Axis(fig[1, 1], xlabel = "Time [s]", ylabel = "Relative error in bubble volume [-]", backgroundcolor = :transparent)

# for (i, ℓₘ) in enumerate(ℓₘs)
#     Vs_i = volume_error(npoints, ℓₘ, dt, true)
#     lines!(ax, range(dt, 1, nt), Vs_i / V, label = "#basis functions=$((ℓₘ + 1)^2)")
# end

# axislegend(ax, position = :rb)
# save("Volume_test_nbf.png", fig)
# ###########################################