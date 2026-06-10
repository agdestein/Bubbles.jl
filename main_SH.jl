import IncompressibleNavierStokes as N
include("src/time_stepping.jl")

n = 40
# L = 1e-2
L = 4.0
# ax = N.tanh_grid(0.0, L, n)
x = LinRange(0.0, L, n+1), LinRange(0.0, L, n+1), LinRange(0.0, L, n+1)
setup = N.Setup(;
    x = x,
    boundary_conditions = (;
        u = (
            (N.SymmetricBC(), N.SymmetricBC()),   # free-slip
            (N.SymmetricBC(), N.SymmetricBC()),
            (N.SymmetricBC(), N.SymmetricBC()),
            # (N.DirichletBC(), N.DirichletBC()),   # no-slip
            # (N.DirichletBC(), N.DirichletBC()),
            # (N.DirichletBC(), N.DirichletBC()),
            # (N.PeriodicBC(), N.PeriodicBC()),
            # (N.PeriodicBC(), N.PeriodicBC()),
            # (N.PeriodicBC(), N.PeriodicBC()),
        ),
    ),
)

# psolver = N.default_psolver(setup)

# ncub, ℓₘ, R, σ = 8066, 15, 1e-3, 73e-4
ncub, ℓₘ, R, σ = 8066, 2, 1.0, 10.0
Bub, Precomp_SH = bubble_setup(ncub, ℓₘ, R, σ, L)

# Velocity field
u = zeros(n + 2, n + 2, n + 2, 3);

solveandplot(u, Bub, setup, Precomp_SH, L);
