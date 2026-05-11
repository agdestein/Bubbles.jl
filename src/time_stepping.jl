using Makie, CairoMakie
import IncompressibleNavierStokes as NS
include("SH.jl")    # import source code

using Adapt
using CUDA
using CUDSS
using LinearAlgebra
using StaticArrays
import AcceleratedKernels as AK


getparams() = (;
    gravity = (0.0, 0.0, -9.81),
    # densities = (; liquid = 1.0e3, gas = 1.25),
    # densities = (; liquid = 1.0, gas = 1e-3),
    # viscosities = (; liquid = 1.0e-3, gas = 1.8e-5),
    densities = (; liquid = 1.0, gas = 1.0),
    viscosities = (; liquid = 5.0e-4, gas = 5.0e-4),

    # Time integration
    dt = 2.0e-3,
    nsubstep = 25, # Steps between plot updates
    nstep = 400,
)

"Left index `n` times away in direction `i`."
@inline left(I::CartesianIndex{D}, i, n = 1) where {D} =
    CartesianIndex(ntuple(j -> j == i ? I[j] - n : I[j], D))

"Right index `n` times away in direction `i`."
@inline right(I::CartesianIndex{D}, i, n = 1) where {D} =
    CartesianIndex(ntuple(j -> j == i ? I[j] + n : I[j], D))

function map_surface_tension!(Fu, setup, surf_tension, r, ϕ, θ, Bub)
    xcub, ycub, zcub = spc2cart(r, ϕ, θ)


    for i = eachindex(xcub)     # 1:length(xcub)
        # Find indices of pressure volume containing quadrature point
        xquad =
         xcub[i] + Bub.centr[1],
         ycub[i] + Bub.centr[2],
         zcub[i] + Bub.centr[3]


        neighbors = ntuple(3) do dim
            xdim = setup.xu[dim][dim] # Vector of staggered points in direction dim

            ii = 1
            while xdim[ii] < xquad[dim] && ii < length(xdim)
                ii += 1
            end
            return ii
        end

        # Find computational cell where 'xquad' resides
        bounds = ntuple(3) do dim
            ii = neighbors[dim] # Left
            xdim = setup.xu[dim][dim]
            return xdim[ii-1], xdim[ii] # Left and right
        end

        # Volumetric surface tension force [N/m³]
        Fσ = surf_tension[i, :] / (bounds[1][2] - bounds[1][1]) / (bounds[2][2] - bounds[2][1]) / (bounds[3][2] - bounds[3][1])
        
        # Add to existing force (convection-diffusion etc.)
        I = CartesianIndex(neighbors)
        for dim = 1:3
            @assert I[dim] > 1
            Fu[left(I, dim, 1), dim] += Fσ[dim] * (xquad[dim] - bounds[dim][1]) / (bounds[dim][2] - bounds[dim][1])
            Fu[I, dim] += Fσ[dim] * (bounds[dim][2] - xquad[dim]) / (bounds[dim][2] - bounds[dim][1])
        end

    end

    return nothing
end

function map_velocity(u, setup, r, ϕ, θ, Bub)
    xcub, ycub, zcub = spc2cart(r, ϕ, θ)

    (; xp) = setup
    xu = setup.x[1][2:end], setup.x[2][2:end], setup.x[3][2:end]

    ucub = zeros(Float64, (length(xcub), 3))

    for i = eachindex(xcub)
        x1, x2, x3 = xcub[i] + Bub.centr[1], ycub[i] + Bub.centr[2], zcub[i] + Bub.centr[3]

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
            (1 - w1u) * w2p * w3p * u[iu - 1, jp, kp, 1] +
            w1u * w2p * w3p * u[iu, jp, kp, 1]

        ucub[i, 2] =
            (1 - w1p) * (1 - w2u) * (1 - w3p) * u[ip - 1, ju - 1, kp - 1, 2] +
            w1p * (1 - w2u) * (1 - w3p) * u[ip, ju - 1, kp - 1, 2] +
            (1 - w1p) * w2u * (1 - w3p) * u[ip - 1, ju, kp - 1, 2] +
            w1p * w2u * (1 - w3p) * u[ip, ju, kp - 1, 2] + 
            (1 - w1p) * (1 - w2u) * w3p * u[ip - 1, ju - 1, kp, 2] +
            w1p * (1 - w2u) * w3p * u[ip, ju - 1, kp, 2] +
            (1 - w1p) * w2u * w3p * u[ip - 1, ju, kp, 2] +
            w1p * w2u * w3p * u[ip, ju, kp, 2]

        ucub[i, 3] = 
            (1 - w1p) * (1 - w2p) * (1 - w3u) * u[ip - 1, jp - 1, ku - 1, 3] +
            w1p * (1 - w2p) * (1 - w3u) * u[ip, jp - 1, ku - 1, 3] +
            (1 - w1p) * w2p * (1 - w3u) * u[ip - 1, jp, ku - 1, 3] +
            w1p * w2p * (1 - w3u) * u[ip, jp, ku - 1, 3] + 
            (1 - w1p) * (1 - w2p) * w3u * u[ip - 1, jp - 1, ku, 3] +
            w1p * (1 - w2p) * w3u * u[ip, jp - 1, ku, 3] +
            (1 - w1p) * w2p * w3u * u[ip - 1, jp, ku, 3] +
            w1p * w2p * w3u * u[ip, jp, ku, 3]

    end
    return ucub
end

function applygravity!(f)
    g = getparams().gravity
    AK.foreachindex(f) do ilin
        II = CartesianIndices(f)[ilin]
        dim = II.I[4]   # 4 in 3D
        f[II] += g[dim]
    end
end

function convectiondiffusion_nonconstant_SH!(f, u, fractions, setup)
    (; Iu) = setup
    (; viscosities, densities) = getparams()
    AK.foreachindex(f) do ilin
        II = CartesianIndices(f)[ilin]
        i, j, k, dim = II.I
        I = CartesianIndex(i, j, k)
        I in Iu[dim] || return nothing
        conv =
            NS.tensordivergence(setup, NS.convstress, (setup, u), dim, 1, I) +
            NS.tensordivergence(setup, NS.convstress, (setup, u), dim, 2, I) +
            NS.tensordivergence(setup, NS.convstress, (setup, u), dim, 3, I)
        diff =
            NS.tensordivergence(setup, NS.diffstress, (setup, u, 1.0), dim, 1, I) +
            NS.tensordivergence(setup, NS.diffstress, (setup, u, 1.0), dim, 2, I) + 
            NS.tensordivergence(setup, NS.diffstress, (setup, u, 1.0), dim, 3, I)
        a = (fractions[I] + fractions[right(I, dim, 1)]) / 2 # Interpolate to velocity point
        dens = (1 - a) * densities.liquid + a * densities.gas
        visc = dens / ((1 - a) * densities.liquid / viscosities.liquid + a * densities.gas / viscosities.gas)
        f[II] += conv + visc * diff
        return nothing
    end
    return nothing
end

"""
Return `true` if `point` is inside the bubble.
This is determined by whether the number of intersections of the segment `point`-`xcenter` is odd or even with the marker segments.
`point` and `xcenter` are `MyPoint`s, while `x` is a vector of `MyPoint`s.
"""
function check_if_inside_SH(point, Bub, Precomp_SH)
    _, ϕp, θp = cart2spc(point[1], point[2], point[3])
    dist_point_centr = sqrt((point[1] - Bub.centr[1])^2 + (point[2] - Bub.centr[2])^2 + (point[3] - Bub.centr[3])^2)
    dist_bub_centr = dot(Bub.c, get_SH(maximum(Precomp_SH.ℓs), ϕp, θp))
    signed_dist = abs(dist_point_centr) - dist_bub_centr

    return signed_dist < 0
end

function compute_fractions_SH!(fractions, Bub, Precomp_SH, setup)
    (; xp) = setup
    xu = setup.x # Includes leftmost point
    AK.foreachindex(fractions) do index
        I = CartesianIndices(fractions)[index]
        i, j, k = I.I
        ninside = 0
        for dk in (0,1), dj in (0, 1), di in (0, 1)
            point = (xu[1][i + di], xu[2][j + dj], xu[3][k + dk])
            ninside += check_if_inside_SH(point, Bub, Precomp_SH)
        end
        fractions[I] = ninside / 8 # 1.0 if all corners inside, 0.0 if all corners outside
    end
    return nothing
end

"""
Perform one time step for the total state `U = (; u, x)`, where
`u` is the velocity field and `x` are the control points defining the bubble.
Wray's low-storage RK3 method is used, which only relies on two 
temporary registers `F` and `U0` (same size as `U`).
In addition, we need a pressure register `p` and a surface tension register `tension`.
"""
function rk3step_SH!(F, U0, U, t, dt, fractions, p, psolver, setup, Bub, Precomp_SH)
    # RK coefficients
    a = 8 / 15, 5 / 12, 3 / 4
    b = 1 / 4, 0.0
    rk_c = 0.0, 8 / 15, 2 / 3
    nstage = length(a)

    # Update current solution
    t0 = t
    foreach(copyto!, U0, U)

    # RK3 substeps
    for i in 1:nstage
        Dynamic_SH = Y2r(Bub, Precomp_SH)
        # Apply right-hand side function to current state U, put in F
        compute_fractions_SH!(fractions, Bub, Precomp_SH, setup) # Current phase fractions
        fill!(F.u, 0) # Initialize with 0
        # NS.convectiondiffusion!(F.u, U.u, setup, viscosity) # This adds to existing force
        convectiondiffusion_nonconstant_SH!(F.u, U.u, fractions, setup) # This adds to existing force
        tension = surface_tension(Precomp_SH, Dynamic_SH, Bub.σ)
        map_surface_tension!(F.u, setup, tension, Dynamic_SH.r, Precomp_SH.ϕ, Precomp_SH.θ, Bub)
        applygravity!(F.u)
        u_cub = map_velocity(U.u, setup, Dynamic_SH.r, Precomp_SH.ϕ, Precomp_SH.θ, Bub)
        dc_dt, dcentr_dt = time_step(Bub, Precomp_SH, Dynamic_SH, u_cub)

        # Evolve U: flow field
        t = t0 + rk_c[i] * dt
        @. U.u = U0.u + a[i] * dt * F.u
        NS.apply_bc_u!(U.u, t, setup)
        NS.project!(U.u, setup; psolver, p)
        # Evolve U: bubble
        @. U.c = U0.c + a[i] * dt * dc_dt
        @. U.centr = U0.centr + a[i] * dt * dcentr_dt

        # Evolve U0
        # Skip for last iter
        if i < nstage
            @. U0.u += b[i] * dt * F.u
            @. U0.c += b[i] * dt * dc_dt
            @. U0.centr += b[i] * dt * dcentr_dt
        end

        # Fill boundary values at new time
        NS.apply_bc_u!(U.u, t, setup)
    end

    # Full time step
    t = t0 + dt

    return nothing
end

"""
""" 
function solveandplot(u, Bub, setup, psolver, Precomp_SH)
    params = getparams()
    (; dt, nsubstep, nstep) = params

    # Allocate registers
    U = (; u, Bub.c, Bub.centr) # Current state
    U0 = deepcopy(U) # RK3 accumulator for previous stages
    F = deepcopy(U) # RK3 right hand side
    # tension = similar(x) # Surface tension at markers
    p = NS.scalarfield(setup) # Pressure
    fractions = NS.scalarfield(setup) # Phase fractions (1.0 if inside bubble, 0.0 outside)

    t = 0.0
    for itime in 1:nstep
        for isub in 1:nsubstep
            # Perform one RK3 step of step size `dt`
            rk3step_SH!(F, U0, U, t, dt, fractions, p, psolver, setup, Bub, Precomp_SH)
            t += dt

            # @info "itime = $itime / $nstep, isub = $isub / $nsubstep, t = $(round(t, digits = 4))" # maximum(abs, U.u)
        end

        @info "itime = $itime / $nstep, t = $(round(t, digits = 4))" # maximum(abs, U.u)
    end

    return U
end
