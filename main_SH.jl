import IncompressibleNavierStokes as N
include("src/time_stepping.jl")

L = 4

n = 40 # 5*5, 5*6, 5*8, 5*10
# L = 1e-2
# L = 4.0
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

# @show minimum(setup.Δu[1]), any(iszero, setup.Δu[1])
# @show any(iszero, setup.Δ[1]), any(iszero, setup.Δ[2]), any(iszero, setup.Δ[3])
# @show minimum(setup.Δ[1]), setup.Δ[1]    # print the whole spacing vector

# psolver = N.default_psolver(setup)
ncub, ℓₘ, R, σ = 8066, 2, 1.0, 10.0
# ncub, ℓₘ, R, σ = 8066, 15, 1e-3, 73e-3

# ncub, ℓₘ, R, σ = 8066, 2, 1.0, 10.0
Bub, Precomp_SH = bubble_setup(ncub, ℓₘ, R, σ, L)

# Velocity field
u = zeros(n + 2, n + 2, n + 2, 3);

# solveandplot(u, Bub, setup, Precomp_SH, L);
solveandplot_mat(u, Bub, setup, Precomp_SH, L);
