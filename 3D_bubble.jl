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

Bub, Precomp_SH = bubble_setup(ncub, ℓₘ, R)

# While time stepping we use:
Dynamic_SH = Y2r(Bub, Precomp_SH)