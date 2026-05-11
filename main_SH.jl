import IncompressibleNavierStokes as N
include("src/time_stepping.jl")

n = 16
L = 1e-1
ax = N.tanh_grid(0.0, L, n)
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

psolver = N.default_psolver(setup)

ncub, ℓₘ, R, σ = 8066, 10, 1e-3, 73e-3
Bub, Precomp_SH = bubble_setup(ncub, ℓₘ, R, σ, L)

# Velocity field
u = zeros(n + 2, n + 2, n + 2, 3);

solveandplot(u, Bub, setup, psolver, Precomp_SH)
