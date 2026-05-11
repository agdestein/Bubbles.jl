using Makie, CairoMakie
import IncompressibleNavierStokes as N
include("src/SH.jl")    # import source code

n = 16
ax = N.tanh_grid(0.0, 1.0, n)
setup = N.Setup(;
    x = (ax, ax, ax),
    boundary_conditions = (;
        u = (
            (N.DirichletBC(), N.DirichletBC()),
            (N.DirichletBC(), N.DirichletBC()),
            (N.DirichletBC(), N.DirichletBC()),
        ),
    ),
)

ℓₘ = 10
ncub = 8066
# a1, a2, a3 = 1e-3, 1e-3, 1e-3   # [m] Initial bubble half-axes (ellipsoid)
R = 1e-3    # [m] Initial bubble radius (sphere)

# Spherical design cubature points:
_, ϕ, θ = get_points_spc(ncub)

# Bubble initialization:
c = zeros(Float64, (ℓₘ + 1) ^ 2)    # spherical harmonics coefficients
c[0] = R * sqrt(4. * π)
centr = zeros(Float64, 3)           # centroid position 
V = 4. / 3. * π * R^3               # total bubble volume
Bub = (; c, centr, V)

# Precomputed spherical harmonics (derivatives) at spherical design cubature points:
Y, dY_dϕ, dY_dθ, d²Y_dϕ², d²Y_dθdϕ, d²Y_dθ², ℓs, ms, one, mone, zero = get_SH_der2(ℓₘ, ϕ_fit, θ_fit)
Precomp_SH = (; ϕ, θ, Y, dY_dϕ, dY_dθ, d²Y_dϕ², d²Y_dθdϕ, d²Y_dθ², ℓs, ms, one, mone, zero)

# While time stepping we use:
# Dynamic_SH = (; r, dr_dϕ, dr_dθ, d²r_dϕ², d²r_dϕdθ, d²r_dθ², dr_dθ_div_sinϕ, EN_lim)    # computed by Y2r 
Dynamic_SH = Y2r(Bub, Precomp_SH)