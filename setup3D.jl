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

    return nothing
end

function map_velocity(u, setup, r, ϕ, θ)
    xcub, ycub, zcub = spc2cart(r, ϕ, θ)

    (; xp) = setup
    xu = setup.x[1][2:end], setup.x[2][2:end], setup.x[3][2:end]

    ucub = zeros(Float64, (length(xcub), 3))

    for i = eachindex(xcub)
        x1, x2, x3 = xcub[i], ycub[i], zcub[i]

        ip = 1; while ip < length(xp[1]) && xp[1][ip] < x1; ip += 1; end
        iu = 1; while iu < length(xu[1]) && xu[1][iu] < x1; iu += 1; end
        jp = 1; while jp < length(xp[2]) && xp[2][jp] < x2; jp += 1; end
        ju = 1; while ju < length(xu[2]) && xu[2][ju] < x2; ju += 1; end
        kp = 1; while kp < length(xp[3]) && xp[3][kp] < x3; kp += 1; end 
        ku = 1; while ku < length(xu[3]) && xu[3][ku] < x3; ku += 1; end

        # Linear interpolation weights for each dimension and CENTER/EDGE type
        w1u = (x1 - xu[1][iu - 1]) / (xu[1][iu] - xu[1][iu - 1])
        w1p = (x1 - xp[1][ip - 1]) / (xp[1][ip] - xp[1][ip - 1])
        w2u = (x2 - xu[2][ju - 1]) / (xu[2][ju] - xu[2][ju - 1])
        w2p = (x2 - xp[2][jp - 1]) / (xp[2][jp] - xp[2][jp - 1])
        w3u = (x3 - xu[3][ku - 1]) / (xu[3][ku] - xu[3][ku - 1])
        w3p = (x3 - xp[3][kp - 1]) / (xp[3][kp] - xp[3][kp - 1])

        # Compute velocity at marker control point by bilinear interpolation
        ucub[i, 1] =
            (1 - w1u) * (1 - w2p) * (1 - w3p) * u[iu - 1, jp - 1, kp - 1, 1] +
            w1u * (1 - w2p) * (1 - w3p) * u[iu, jp - 1, kp - 1, 1] +
            (1 - w1u) * w2p * (1 - w3p) * u[iu - 1, jp, kp - 1, 1] +
            w1u * w2p * (1 - w3p) * u[iu, jp, kp - 1, 1] + 
            (1 - w1u) * (1 - w2p) * w3p * u[iu - 1, jp - 1, kp, 1] +
            w1u * (1 - w2p) * w3p * u[iu, jp - 1, kp, 1] +
            (1 - w1u) * w2p * w3p * [iu - 1, jp, kp, 1] +
            w1u * w2p * w3p * [iu, jp, kp, 1]

        ucub[i, 2] =
            (1 - w1p) * (1 - w2u) * (1 - w3p) * u[ip - 1, ju - 1, kp - 1, 2] +
            w1p * (1 - w2u) * (1 - w3p) * u[ip, ju - 1, kp - 1, 2] +
            (1 - w1p) * w2u * (1 - w3p) * u[ip - 1, ju, kp - 1, 2] +
            w1p * w2u * (1 - w3p) * u[ip, ju, kp - 1, 2] + 
            (1 - w1p) * (1 - w2u) * w3p * u[ip - 1, ju - 1, kp, 1] +
            w1p * (1 - w2u) * w3p * u[ip, ju - 1, kp, 1] +
            (1 - w1p) * w2u * w3p * [ip - 1, ju, kp, 1] +
            w1p * w2u * w3p * [ip, ju, kp, 1]

        ucub[i, 3] = 
            (1 - w1p) * (1 - w2p) * (1 - w3u) * u[ip - 1, jp - 1, ku - 1, 2] +
            w1p * (1 - w2p) * (1 - w3u) * u[ip, jp - 1, ku - 1, 2] +
            (1 - w1p) * w2p * (1 - w3u) * u[ip - 1, jp, ku - 1, 2] +
            w1p * w2p * (1 - w3u) * u[ip, jp, ku - 1, 2] + 
            (1 - w1p) * (1 - w2p) * w3u * u[ip - 1, jp - 1, ku, 1] +
            w1p * (1 - w2p) * w3u * u[ip, jp - 1, ku, 1] +
            (1 - w1p) * w2p * w3u * [ip - 1, jp, ku, 1] +
            w1p * w2p * w3u * [ip, jp, ku, 1]

    end
    return ucub
end