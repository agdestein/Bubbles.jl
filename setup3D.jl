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

# Pressure points (cell centered)
setup.xp[1] # Vector with x1 coordinates of p
setup.xp[2] # Vector with x2 coordinates of p
setup.xp[3] # Vector with x3 coordinates of p

# Velocity points (staggered in the respective dimension)
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

# Quadrature point
xquad = 0.8, 0.7, 0.35

# Find indices of staggered points LEFT of quadrature point
neighbors = map(1:3) do dim
    xdim = setup.xu[dim][dim] # Vector of staggered points in direction dim
    i = 1
    while xdim[i + 1] < xquad[dim] && i < length(xdim)
        i += 1
    end
    return i
end

function toto(dim, xquad)
    xdim = setup.xu[dim][dim] # Vector of staggered points in direction dim
    i = 1
    while xdim[i + 1] < xquad[dim] && i < length(xdim)
        i += 1
    end
    return i
end

map(toto, 1:3)

# Check that bounding box is correct
bounds = map(1:3) do dim
    i = neighbors[dim] # Left
    xdim = setup.xu[dim][dim]
    return xdim[i], xdim[i + 1] # Left and right
end
xquad

bounds
bounds[1]
bounds[1][2]
neighbors

function map_surface_tension!(Fu, setup, surf_tension, r, ϕ, θ)
    xcub, ycub, zcub = spc2cart(r, ϕ, θ)

    for i = eachindex(xcub)     # 1:length(xcub)
        # Find indices of staggered points LEFT of quadrature point
        xquad = xcub[i], ycub[i], zcub[i]
        neighbors = map(1:3) do dim
            xdim = setup.xu[dim][dim] # Vector of staggered points in direction dim
            i = 1
            while xdim[i + 1] < xquad[dim] && i < length(xdim)
                i += 1
            end
            return i
        end

        # Find computational cell where 'xquad' resides
        bounds = map(1:3) do dim
            i = neighbors[dim] # Left
            xdim = setup.xu[dim][dim]
            return xdim[i], xdim[i + 1] # Left and right
        end

        # Volumetric surface tension force [N/m³]
        Fσ = surf_tension[i, :] / (bounds[1][2] - bounds[1][1]) / (bounds[2][2] - bounds[2][1]) / (bounds[3][2] - bounds[3][1])
        
        # Add to existing force (convection-diffusion etc.)
        for dim = 1:3
            Fu[neighbors[dim][1], dim] += Fσ[dim] * (xquad[dim] - bounds[dim][1]) / (bounds[dim][2] - bounds[dim][1])
            Fu[neighbors[dim][2], dim] += Fσ[dim] * (bounds[dim][2] - xquad[dim]) / (bounds[dim][2] - bounds[dim][1])
        end

    end

end