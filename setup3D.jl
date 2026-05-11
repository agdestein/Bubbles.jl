import IncompressibleNavierStokes as N

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

# Pressure points
setup.xp[1] # Vector with x1 coordinates of p
setup.xp[2] # Vector with x2 coordinates of p
setup.xp[3] # Vector with x3 coordinates of p

# Velocity points
setup.xu[1] # 3 vectors with coordinates of u1
setup.xu[2] # 3 vectors with coordinates of u2
setup.xu[3] # 3 vectors with coordinates of u3

# Coordinates of u1
setup.xu[1][1] # Staggered in x1 direction
setup.xu[1][2] # Centered in x2 direction
setup.xu[1][3] # Centered in x3 direction

# Velocity field
u = zeros(n + 2, n + 2, n + 2, 3)
u |> size

# Cartesina
I = CartesianIndex(5, 3, 8)
dim = 1
u[I, dim] = 10.0 # With CartesianIndex and an individual index
u[5, 3, 8, dim] # With individual indices
i_linear = reshape(eachindex(u), size(u))[I, dim] # Linear index corresponding to (I, dim)
eachindex(u) # All linear indices of u
CartesianIndices(u) # All Cartesian indices of u
I_total = CartesianIndices(u)[i_linear] # Total Cartesian index (5, 3, 8, dim)
u[i_linear] # With linear index
u[I_total] # With total Cartesian index

# Look up velocity at an index and find corresponding coordinates
setup.xu[dim][1][I[1]]
setup.xu[dim][2][I[2]]
setup.xu[dim][3][I[3]]
