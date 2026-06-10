using Makie, WGLMakie
# using Makie, CairoMakie
import IncompressibleNavierStokes as NS
include("SH.jl")    # import source code
const IS = NS.IterativeSolvers

using Adapt
using CUDA
using CUDSS
using LinearAlgebra
using StaticArrays
using SparseArrays
import AcceleratedKernels as AK


getparams() = (;
    gravity = (0.0, 0.0, -9.81),
    # gravity = (0.0, 0.0, 0.0),
    # densities = (; liquid = 1.0e3, gas = 1.25),
    # densities = (; liquid = 1.0, gas = 1e-3),
    # viscosities = (; liquid = 1.0e-3, gas = 1.8e-5),

    # densities = (; liquid = 1e3, gas = 1.25),
    # viscosities = (; liquid = 1e-6, gas = 1.44e-5),  # kinematic!
    densities = (; liquid = 0.1, gas = 100.0),
    viscosities = (; liquid = 1e-5, gas = 0.35e-2),  # kinematic!

    # Time integration
    # dt = 0.5e-3,
    dt = 0.05,
    nsubstep = 1, # Steps between plot updates
    nstep = 200,
)

"Left index `n` times away in direction `i`."
@inline left(I::CartesianIndex{D}, i, n = 1) where {D} =
    CartesianIndex(ntuple(j -> j == i ? I[j] - n : I[j], D))

"Right index `n` times away in direction `i`."
@inline right(I::CartesianIndex{D}, i, n = 1) where {D} =
    CartesianIndex(ntuple(j -> j == i ? I[j] + n : I[j], D))

function map_surface_tension!(Fu, setup, surf_tension, r, ϕ, θ, Bub, fractions)
    xcub, ycub, zcub = spc2cart(r, ϕ, θ)

    (; densities) = getparams()

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
            ii = neighbors[dim] # Right
            xdim = setup.xu[dim][dim]
            return xdim[ii-1], xdim[ii] # Left and right
        end

        # Volumetric surface tension force [N/m³]
        Fσ = surf_tension[i, :] / (bounds[1][2] - bounds[1][1]) / (bounds[2][2] - bounds[2][1]) / (bounds[3][2] - bounds[3][1])
        
        # Add to existing force (convection-diffusion etc.)
        I = CartesianIndex(neighbors)
        for dim = 1:3
            @assert I[dim] > 1
            # Find phase fraction for each velocity cell and divide Fσ by effective density:
            a = (fractions[I] + fractions[right(I, dim, 1)]) / 2        # interpolate to velocity point
            dens = (1 - a) * densities.liquid + a * densities.gas       # effective density
            a_left = (fractions[left(I, dim, 1)] + fractions[I]) / 2
            dens_left = (1 - a_left) * densities.liquid + a_left * densities.gas
            Fu[left(I, dim, 1), dim] += Fσ[dim] * (xquad[dim] - bounds[dim][1]) / (bounds[dim][2] - bounds[dim][1]) / dens_left
            Fu[I, dim] += Fσ[dim] * (bounds[dim][2] - xquad[dim]) / (bounds[dim][2] - bounds[dim][1]) / dens
        end

    end

    return nothing
end

function map_velocity(u, setup, xcub, ycub, zcub)
    (; xp) = setup
    xu = setup.x[1][2:end], setup.x[2][2:end], setup.x[3][2:end]

    ucub = zeros(Float64, (length(xcub), 3))

    for i = eachindex(xcub)
        # x1, x2, x3 = xcub[i] + Bub.centr[1], ycub[i] + Bub.centr[2], zcub[i] + Bub.centr[3]
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

function apply_effective_gravity!(f, fractions, setup)
    g = getparams().gravity
    (; densities) = getparams()
    (; Iu) = setup
    ρ_ref = densities.liquid

    AK.foreachindex(f) do ilin
        II = CartesianIndices(f)[ilin]
        i, j, k, dim = II.I
        I = CartesianIndex(i, j, k)
        I in Iu[dim] || return nothing

        a_face = (fractions[I] + fractions[right(I, dim)]) / 2
        ρ_face = (1 - a_face) * densities.liquid + a_face * densities.gas
        f[II] += g[dim] * (1 - ρ_ref / ρ_face)
        return nothing
    end
end

function convectiondiffusion_nonconstant_SH!(f, u, sdf, setup)
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

        frac_left = abs(sdf[I]) / (abs(sdf[I]) + abs(sdf[right(I,dim,1)]))
        left_inside = sdf[I] < 0
        right_inside = sdf[right(I,dim,1)] < 0
        ν_left = [viscosities.liquid, viscosities.gas][left_inside + 1]
        ν_right = [viscosities.liquid, viscosities.gas][right_inside + 1]
        ν_eff = frac_left * ν_left + (1 - frac_left) * ν_right

        # ρ_eff = left_inside * (frac_left * densities.gas + (1 - frac_left) * densities.liquid) + 
        #         (1 - left_inside) * (frac_left * densities.liquid + (1 - frac_left) * densities.gas)
        # ν_eff = left_inside * (frac_left * viscosities.gas + (1 - frac_left) * viscosities.liquid) + 
        #         (1 - left_inside) * (frac_left * viscosities.liquid + (1 - frac_left) * viscosities.gas)
        f[II] += conv + ν_eff * diff

        # a = (fractions[I] + fractions[right(I, dim, 1)]) / 2 # Interpolate to velocity point
        # dens = (1 - a) * densities.liquid + a * densities.gas
        # visc = dens / ((1 - a) * densities.liquid / viscosities.liquid + a * densities.gas / viscosities.gas)
        # f[II] += conv + visc * diff
    end
    return nothing
end

"""
Return `true` if `point` is inside the bubble.
This is determined by whether the number of intersections of the segment `point`-`xcenter` is odd or even with the marker segments.
`point` and `xcenter` are `MyPoint`s, while `x` is a vector of `MyPoint`s.
"""
function check_if_inside_SH(point, Bub, Precomp_SH)
    rp, ϕp, θp = cart2spc(point[1] - Bub.centr[1], point[2] - Bub.centr[2], point[3] - Bub.centr[3])
    # dist_point_centr = sqrt((point[1] - Bub.centr[1])^2 + (point[2] - Bub.centr[2])^2 + (point[3] - Bub.centr[3])^2)
    dist_bub_centr = dot(Bub.c, get_SH(maximum(Precomp_SH.ℓs), ϕp, θp))
    signed_dist = rp - dist_bub_centr

    return signed_dist < 0, signed_dist # signed distance is negative inside gas bubble
end

function compute_fractions_SH!(fractions, sdf, Bub, Precomp_SH, setup)
    (; xp) = setup
    xu = setup.x # Includes leftmost point
    AK.foreachindex(fractions) do index
        I = CartesianIndices(fractions)[index]
        i, j, k = I.I
        ninside = 0
        for dk in (0, 1), dj in (0, 1), di in (0, 1)
            # point = (xu[1][i + di], xu[2][j + dj], xu[3][k + dk])
            point = (xu[1][i + di], xu[2][j + dj], xu[3][k + dk])
            ninside += check_if_inside_SH(point, Bub, Precomp_SH)[1]
        end
        center = (xp[1][i], xp[2][j], xp[3][k])
        sdf[I] = check_if_inside_SH(center, Bub, Precomp_SH)[2]
        fractions[I] = ninside / 8 # 1.0 if all corners inside, 0.0 if all corners outside
    end
    return nothing
end

"""
Apply the variable-coefficient pressure operator on `p` and write the result to `Ap`.

This computes a face-centered flux approximation of `(1/ρ)∇p`, where `ρ` is
selected from `sdf` (inside bubble: gas, outside: liquid), then takes its
divergence to form `∇·((1/ρ)∇p)`.
"""
function apply_varcoef_poisson!(Ap, p, sdf, setup, flux)
    T = eltype(p)
    (; Iu, Δu) = setup
    (; densities) = getparams()

    fill!(flux, 0)
    fill!(Ap, 0)
    NS.apply_bc_p!(p, T(0), setup)

    AK.foreachindex(flux) do ilin
        II = CartesianIndices(flux)[ilin]
        i, j, k, dim = II.I
        I = CartesianIndex(i, j, k)
        I in Iu[dim] || return nothing

        I_right = right(I, dim)
        # Linear interface weighting to define an effective face density.
        frac_left = abs(sdf[I]) / (abs(sdf[I]) + abs(sdf[I_right]) + eps(T))
        left_inside = sdf[I] < 0
        right_inside = sdf[I_right] < 0
        ρ_left = left_inside ? densities.gas : densities.liquid
        ρ_right = right_inside ? densities.gas : densities.liquid
        ρ_face = frac_left * ρ_left + (1 - frac_left) * ρ_right
        flux[II] = (p[I_right] - p[I]) / Δu[dim][II.I[dim]] / ρ_face
        return nothing
    end

    NS.divergence!(Ap, flux, setup)
    NS.scalewithvolume!(Ap, setup)
    NS.apply_bc_p!(Ap, T(0), setup)

    return nothing
end

"""
    zero_mean_ip!(q, Ip)

Enforce zero mean on pressure unknowns indexed by `Ip`.

The pressure Poisson problem is defined up to an additive constant; projecting
onto the mean-free subspace removes this nullspace in both operator application
and right-hand side preparation.
"""
function zero_mean_ip!(q, Ip)
    q_ip = view(q, Ip)
    q_ip .-= sum(q_ip) / length(q_ip)
    return nothing
end

"""
    VarCoefPoissonOp

Matrix-free linear operator for the variable-density pressure Poisson solve over
interior pressure indices `Ip`.
"""
struct VarCoefPoissonOp{TS,TA,TF,TQ,TI}
    setup::TS
    sdf::TA
    flux::TF
    q::TQ
    Aq::TQ
    Ip::TI
end

Base.size(op::VarCoefPoissonOp) = (length(op.Ip), length(op.Ip))
Base.size(op::VarCoefPoissonOp, dim::Integer) = size(op)[dim]

"""
    LinearAlgebra.mul!(y, op::VarCoefPoissonOp, x)

Apply the matrix-free variable-coefficient Poisson operator to vector `x` and
store the result in `y`.

`x` and `y` live on the packed interior index set `op.Ip`; this method expands
to grid fields, applies boundary conditions, evaluates the operator, and packs
back to vector form while preserving a zero-mean gauge.
"""
function LinearAlgebra.mul!(y, op::VarCoefPoissonOp, x)
    fill!(op.q, 0)
    # Unpack Krylov vector into grid storage on pressure points.
    @inbounds for (k, I) in enumerate(op.Ip)
        op.q[I] = x[k]
    end
    zero_mean_ip!(op.q, op.Ip)
    NS.apply_bc_p!(op.q, zero(eltype(op.q)), op.setup)

    apply_varcoef_poisson!(op.Aq, op.q, op.sdf, op.setup, op.flux)
    zero_mean_ip!(op.Aq, op.Ip)

    # Pack matrix-free result back into the Krylov vector layout.
    @inbounds for (k, I) in enumerate(op.Ip)
        y[k] = op.Aq[I]
    end
    y .-= sum(y) / length(y)
    return y
end

"""
Project velocity onto the divergence-free space using a spatially varying density
by solving the variable-coefficient Poisson equation
∇·((1/ρ)∇p) = ∇·u with homogeneous pressure boundary conditions.
"""
function project_vd!(u, sdf, jumps, dt, setup; psolver, p)
    T = eltype(u)
    (; Ip, Iu, Δu) = setup
    (; densities) = getparams()

    rhs = similar(p)
    rhs_jump = similar(p)
    flux = similar(u)
    # invρ_face = similar(u)
    q = similar(p)
    Aq = similar(p)

    # face_inv_density!(invρ_face, fractions, setup)

    # Use a fresh initial guess to make projection diagnostics easier to interpret.
    # fill!(p, 0)
    # NS.apply_bc_p!(p, T(0), setup)

    Ip_vec = collect(Ip)
    b = zeros(T, length(Ip_vec))
    x = zeros(T, length(Ip_vec))
    op = VarCoefPoissonOp(setup, sdf, flux, q, Aq, Ip_vec)
    tol_rel = T(1e-8)
    tol_abs = T(1e-12)
    restart = 200
    maxiter = 500

    # Defect-correction with GMRES: solve once, apply correction, recompute
    # divergence residual, and optionally solve one more correction.
    npass = 2
    iters_total = 0
    rr0 = zero(T)
    rr = zero(T)
    converged = false

    for pass in 1:npass
        NS.divergence!(rhs, u, setup)
        NS.scalewithvolume!(rhs, setup)
        NS.divergence!(rhs_jump, jumps, setup)
        NS.scalewithvolume!(rhs_jump, setup)
        # Capillary jump is an acceleration term; projection correction uses dt*jumps.
        rhs .-= dt .* rhs_jump
        NS.apply_bc_p!(rhs, T(0), setup)
        zero_mean_ip!(rhs, Ip)

        @inbounds for (k, I) in enumerate(Ip_vec)
            b[k] = rhs[I]
        end
        rr_pass = sum(abs2, b)
        if pass == 1
            rr0 = rr_pass
        end

        # Use each pass's own rr as reference so the early-exit threshold is correct.
        tol2 = max(tol_abs^2, (tol_rel^2) * max(rr_pass, eps(T)))
        if rr_pass <= tol2
            rr = rr_pass
            converged = true
            break
        end

        fill!(x, 0)
        _, history = IS.gmres!(
            x,
            op,
            b;
            reltol = tol_rel,
            abstol = tol_abs,
            restart = restart,
            maxiter = maxiter,
            initially_zero = true,
            log = true,
        )
        iters_total += history.iters

        fill!(p, 0)
        @inbounds for (k, I) in enumerate(Ip_vec)
            p[I] = x[k]
        end
        zero_mean_ip!(p, Ip)
        NS.apply_bc_p!(p, T(0), setup)

        AK.foreachindex(u) do ilin
            II = CartesianIndices(u)[ilin]
            i, j, k, dim = II.I
            I = CartesianIndex(i, j, k)
            I in Iu[dim] || return nothing

            I_right = right(I, dim, 1)
            frac_left = abs(sdf[I]) / (abs(sdf[I]) + abs(sdf[I_right]) + eps(T))
            left_inside  = sdf[I]      < 0
            right_inside = sdf[I_right] < 0
            ρ_left  = left_inside  ? densities.gas : densities.liquid
            ρ_right = right_inside ? densities.gas : densities.liquid
            ρ_eff = frac_left * ρ_left + (1 - frac_left) * ρ_right
            u[II] -= (1 / ρ_eff) * (p[I_right] - p[I]) / Δu[dim][II.I[dim]] + dt * jumps[II]
            return nothing
        end

        NS.divergence!(rhs, u, setup)
        NS.scalewithvolume!(rhs, setup)
        NS.divergence!(rhs_jump, jumps, setup)
        NS.scalewithvolume!(rhs_jump, setup)
        rhs .-= dt .* rhs_jump
        NS.apply_bc_p!(rhs, T(0), setup)
        zero_mean_ip!(rhs, Ip)
        rr = sum(abs2, view(rhs, Ip))
        if rr <= tol2
            converged = true
            break
        end
    end

    return (; iter = iters_total, rr0 = rr0, rr = rr, method = :varcoef_gmres_refined, converged = converged)
end

function divergence_stats!(divbuf, u, setup)
    NS.divergence!(divbuf, u, setup)
    NS.scalewithvolume!(divbuf, setup)
    div_inside = view(divbuf, setup.Ip)
    l2 = sqrt(sum(abs2, div_inside))
    linf = maximum(abs, div_inside)
    return l2, linf
end

function p_jump!(f, sdf, Bub, Precomp_SH, setup)
    (; Iu, Δu) = setup
    (; densities) = getparams()

    jumps = similar(f)
    fill!(jumps, 0.0)
    ρ_eff = similar(f)
    fill!(ρ_eff, densities.liquid)  # for boundary values (I **not** in Iu[dim])

    AK.foreachindex(f) do ilin
        II = CartesianIndices(f)[ilin]
        i, j, k, dim = II.I
        I = CartesianIndex(i, j, k)
        I in Iu[dim] || return nothing

        u_point = setup.xu[dim][dim][II.I[dim]]
        
        if u_point > 0 && u_point < maximum(setup.x[dim]) && sdf[i,j,k] * sdf[right(I,dim,1)] < 0
            x_surf = [setup.xu[dim][1][i], setup.xu[dim][2][j], setup.xu[dim][3][k]]
            # x_surf[dim] = x_surf[dim] - 
            #         (setup.xp[dim][II.I[dim]+1] - setup.xp[dim][II.I[dim]]) / (sdf[right(I,dim,1)] - sdf[I]) * sdf[I]
            frac_left = abs(sdf[I]) / (abs(sdf[I]) + abs(sdf[right(I,dim,1)]))
            x_surf[dim] = setup.xp[dim][II.I[dim]] + frac_left * (setup.xp[dim][II.I[dim]+1] - setup.xp[dim][II.I[dim]]) 

            _, ϕ_surf, θ_surf = cart2spc(x_surf[1] - Bub.centr[1], x_surf[2] - Bub.centr[2], x_surf[3] - Bub.centr[3])
            (; Y, dY_dϕ, dY_dθ, d²Y_dϕ², d²Y_dθdϕ, d²Y_dθ², ℓs, ms, one, mone, zero) = get_SH_der2(maximum(Precomp_SH.ℓs), ϕ_surf, θ_surf)
            Precomp_SH_temp = (; ϕ = [ϕ_surf], θ = [θ_surf], Y, dY_dϕ, dY_dθ, d²Y_dϕ², d²Y_dθdϕ, d²Y_dθ², ℓs, ms, one, mone, zero)
            Dynamic_SH_temp = Y2r(Bub, Precomp_SH_temp)

            dS = surface_element(Precomp_SH_temp, Dynamic_SH_temp)
            κ = surface_curvature(Precomp_SH_temp, Dynamic_SH_temp, dS)[1]  # vector to scalar

            # add pressure jump correction to pressure gradient in momentum equation:
            # Use local face density consistent with project_vd!
            # a_face = (fractions[I] + fractions[right(I, dim, 1)]) / 2
            # ρ_face = (1 - a_face) * densities.liquid + a_face * densities.gas
            left_inside = sdf[I] < 0
            ρ_eff[II] = left_inside * (frac_left * densities.gas + (1 - frac_left) * densities.liquid) + 
                    (1 - left_inside) * (frac_left * densities.liquid + (1 - frac_left) * densities.gas)
            # p_jump = - Bub.σ * κ / Δu[dim][II.I[dim]] / ρ_eff[II]   # κ is negative (outward-facing normal)

            # `p_jump` is positive and is added to ∇p (subtracted from -∇p in momentum eqn) if left cell is liquid
            # f[II] += (1 - left_inside) * p_jump - left_inside * p_jump

            # jumps[II] = (1 - left_inside) * p_jump - left_inside * p_jump
            p_jump = - Bub.σ * κ / Δu[dim][II.I[dim]]
            jumps[II] = left_inside ? - p_jump : p_jump # signs consistent with ∇p, **not** -∇p

        else
            frac_left = abs(sdf[I]) / (abs(sdf[I]) + abs(sdf[right(I,dim,1)]))
            ρ_left = (sdf[I] < 0) ? densities.gas : densities.liquid
            ρ_right = (sdf[right(I,dim,1)] < 0) ? densities.gas : densities.liquid
            @assert ρ_left == ρ_right "interface should not cross, left, right = $((ρ_left, ρ_right))"
            ρ_eff[II] = ρ_left
        end
    end

    # println("ρ_eff min, max: $(minimum(ρ_eff)), $(maximum(ρ_eff))")
    return jumps, ρ_eff
end

function laplacian_mat_variable_ρ(setup, ρ_eff)
    P = NS.pad_scalarfield_mat(setup)
    Bp = NS.bc_p_mat(setup)
    Bu = NS.bc_u_mat(setup)
    G = NS.pressuregradient_mat(setup)
    M = NS.divergence_mat(setup)
    Ω = NS.volume_mat(setup)
    # Rinv = inv_ρ_eff_mat(setup, ρ_eff)  # accounts for variable density
    Rinv = Diagonal(vec(adapt(Array, 1 ./ ρ_eff)))  # accounts for variable density
    P' * Ω * M * Rinv * Bu * G * Bp * P
end

function psolver_cg_matrix_variable_ρ(setup, ρ_eff; kwargs...)
    (; x, Np, Ip, boundary_conditions, backend) = setup 
    T = eltype(x[1])
    L = laplacian_mat_variable_ρ(setup, ρ_eff)

    asymm = norm(L - L', 1)/norm(L, 1)  # relative asymmetry, check before using conjugate gradient
    @info "Laplacian relative asymmetry" asymm
    L0 = NS.laplacian_mat(setup) # original, non-variable ρ, for comparison
    @assert sign(sum(diag(L))) == sign(sum(diag(L0))) "Laplacian has flipped definiteness"

    isdefinite = any(bc -> bc[1] isa NS.PressureBC || bc[2] isa NS.PressureBC, boundary_conditions.u)
    if isdefinite
        ftemp = fill!(similar(x[1], prod(Np)), 0)
        ptemp = fill!(similar(x[1], prod(Np)), 0)
        viewrange = (:)
    else
        ftemp = fill!(similar(x[1], prod(Np) + 1), 0)
        ptemp = fill!(similar(x[1], prod(Np) + 1), 0)
        e = fill(T(1), prod(Np))
        L = [L e; e' 0]
        L = NS.sparseadapt(backend, L)
        viewrange = 1:prod(Np)
    end
    function psolve!(p)
        copyto!(view(ftemp, viewrange), view(view(p, Ip), :))
        IS.cg!(ptemp, L, ftemp; kwargs...)
        copyto!(view(view(p, Ip), :), view(ptemp, viewrange))
        p
    end
end

function poisson_var_ρ_mat!(p, jumps, ρ_eff, setup, u, a, dt, stage)
    (; Ip) = setup

    Rinv = Diagonal(vec(adapt(Array, 1 ./ ρ_eff)))
    jump_vec = vec(adapt(Array, jumps))

    P = NS.pad_scalarfield_mat(setup); Bp = NS.bc_p_mat(setup)
    Bu = NS.bc_u_mat(setup); G = NS.pressuregradient_mat(setup)
    M = NS.divergence_mat(setup); Ω = NS.volume_mat(setup)

    # RHS:
    u_vec = vec(adapt(Array, u))
    div_jump = P' * Ω * M * Rinv * jump_vec
    rhs = (P' * Ω * M * Bu * u_vec) ./ (a[stage] * dt) .+ div_jump

    println("min, max ρ_eff: $(minimum(ρ_eff)), $(maximum(ρ_eff))")
    println("min, max jump: $(minimum(jump_vec)), $(maximum(jump_vec))")

    # check: jump divergence should be zero-mean:
    @info "jump divergence mean:" abs(sum(div_jump))
    @info "1e-10 * norm:" 1e-10 * norm(div_jump)
    @assert abs(sum(div_jump)) < 1e-10 * norm(div_jump) "jump not zero-mean, sign error?"

    # load rhs to p over Ip:
    copyto!(view(p, Ip), rhs)
    psolver = psolver_cg_matrix_variable_ρ(setup, ρ_eff; abstol = 1e-10, reltol = 1e-8, maxiter = 1000)
    psolver(p)

    return Bu, G, Bp, P, Rinv, jump_vec
end

function project_var_ρ_mat!(u, p, Bu, G, Bp, P, Rinv, jump_vec, setup, a, dt, stage)
    (; N, dimension, Ip) = setup 
    D = dimension()

    u_vec = vec(adapt(Array, u))
    p_vec = vec(view(p, Ip))
    gp = Bu * G * Bp * P * p_vec # gradient on velocity grid, without jump 
    corr = Rinv * (gp .- jump_vec)
    u .= reshape(u_vec .- (a[stage] * dt) .* corr, N..., D)
    return nothing
end

"""
Perform one time step for the total state `U = (; u, x)`, where
`u` is the velocity field and `x` are the control points defining the bubble.
Wray's low-storage RK3 method is used, which only relies on two 
temporary registers `F` and `U0` (same size as `U`).
In addition, we need a pressure register `p` and a surface tension register `tension`.
"""
function rk3step_SH!(F, U0, U, t, dt, fractions, sdf, p, setup, Bub, Precomp_SH)
    # RK coefficients
    a = 8 / 15, 5 / 12, 3 / 4
    b = 1 / 4, 0.0
    rk_c = 0.0, 8 / 15, 2 / 3
    nstage = length(a)

    (; Ip, dimension, N) = setup
    D = dimension()

    # Update current solution
    t0 = t
    foreach(copyto!, U0, U)
    # ρ_eff = similar(F.u)

    @info "#Julia threads:" Threads.nthreads()

    # RK3 substeps
    for i in 1:nstage
        Dynamic_SH = Y2r(Bub, Precomp_SH)
        println("coefs: $(Bub.c)")
        println("volume: $(volume(Dynamic_SH))")
        # Apply right-hand side function to current state U, put in F
        compute_fractions_SH!(fractions, sdf, Bub, Precomp_SH, setup) # Current phase fractions
        fill!(F.u, 0) # Initialize with 0
        # NS.convectiondiffusion!(F.u, U.u, setup, viscosity) # This adds to existing force
        convectiondiffusion_nonconstant_SH!(F.u, U.u, sdf, setup) # This adds to existing force
        # NS.applypressure!(F.u, p, setup) # already inside `project!`
        # tension = surface_tension(Precomp_SH, Dynamic_SH, Bub.σ)
        # println(tension[1, 1:end])

        jumps, ρ_eff = p_jump!(F.u, sdf, Bub, Precomp_SH, setup)
        # map_surface_tension!(F.u, setup, tension, Dynamic_SH.r, Precomp_SH.ϕ, Precomp_SH.θ, Bub, fractions)
        # applygravity!(F.u)
        # apply_effective_gravity!(F.u, fractions, setup)
        xcub, ycub, zcub = spc2cart(Dynamic_SH.r, Precomp_SH.ϕ, Precomp_SH.θ)
        xcub .= xcub .+ Bub.centr[1]
        ycub .= ycub .+ Bub.centr[2]
        zcub .= zcub .+ Bub.centr[3] 
        u_cub = map_velocity(U.u, setup, xcub, ycub, zcub)
        dc_dt, dcentr_dt = time_step(Bub, Precomp_SH, Dynamic_SH, u_cub)

        # Evolve U: flow field
        t = t0 + rk_c[i] * dt
        @. U.u = U0.u + a[i] * dt * F.u
        NS.apply_bc_u!(U.u, t, setup)

        ################## pressure solve:
        Bu, G, Bp, P, Rinv, jump_vec = poisson_var_ρ_mat!(p, jumps, ρ_eff, setup, U.u, a, dt, i)

        # velocity projection to ensure divergence free-ness:
        project_var_ρ_mat!(U.u, p, Bu, G, Bp, P, Rinv, jump_vec, setup, a, dt, i)
        ##################################################################################

        # div_pre_l2, div_pre_linf = divergence_stats!(divbuf, U.u, setup)
        # u_before_proj = copy(U.u)
        # proj_stats = project_vd!(U.u, sdf, jumps, dt, setup; psolver, p)
        # div_post_l2, div_post_linf = divergence_stats!(divbuf, U.u, setup)
        # du_proj = U.u .- u_before_proj
        # du_proj_l2 = sqrt(sum(abs2, du_proj))
        # @info "divergence monitor" stage = i pre_l2 = div_pre_l2 pre_linf = div_pre_linf post_l2 = div_post_l2 post_linf = div_post_linf solver_iter = proj_stats.iter solver_rr0 = proj_stats.rr0 solver_rr = proj_stats.rr proj_method = proj_stats.method solver_converged = proj_stats.converged du_proj_l2 = du_proj_l2
        
        
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
function get_bubble(Bub, Precomp_SH)
    rbub = Precomp_SH.Y * Bub.c
    centr = Bub.centr
    ϕ, θ = Precomp_SH.ϕ, Precomp_SH.θ
    xbub, ybub, zbub = centr[1] .+ rbub .* sin.(ϕ) .* cos.(θ),
                        centr[2] .+ rbub .* sin.(ϕ) .* sin.(θ), 
                        centr[3] .+ rbub .* cos.(ϕ)

    return xbub, ybub, zbub
end

"""
""" 
function solveandplot(u, Bub, setup, Precomp_SH, L)
    nvis = 50
    preϕ = range(0.001, 0.999π, nvis)
    preθ = range(0, 2π, nvis)
    # _, ϕ, θ = get_points_spc(npoints)
    ϕvis = zeros(Float64, (length(preϕ), length(preθ)))
    θvis = similar(ϕvis)

    for i = eachindex(preϕ)
        for j = eachindex(preθ)
            ϕvis[i,j] = preϕ[i]
            θvis[i,j] = preθ[j]
        end
    end

    Yvis = get_SH(ℓₘ, ϕvis, θvis)

    params = getparams()
    (; dt, nsubstep, nstep) = params

    # Allocate registers
    U = (; u, Bub.c, Bub.centr) # Current state
    U0 = deepcopy(U) # RK3 accumulator for previous stages
    F = deepcopy(U) # RK3 right hand side
    # tension = similar(x) # Surface tension at markers
    p = NS.scalarfield(setup) # Pressure
    # divbuf = NS.scalarfield(setup) # Divergence monitor buffer
    fractions = NS.scalarfield(setup) # Phase fractions (1.0 if inside bubble, 0.0 outside)
    sdf = NS.scalarfield(setup) # 1.0 if cell center (pressure point) inside bubble, 0.0 if outside 

    xbub, ybub, zbub = get_bubble(Bub, Precomp_SH)

    t = 0.0

    println("min x: $(minimum(xbub)), max x: $(maximum(xbub))")
    println("min y: $(minimum(ybub)), max y: $(maximum(ybub))")
    println("min z: $(minimum(zbub)), max z: $(maximum(zbub))")
    # println("coefs: $(Bub.c)")
    # error()

    fig = Figure(size = (800, 800)) 
    ax = Axis3(fig[1,1], aspect = :equal)
    xlims!(ax, (0, L))
    ylims!(ax, (0, L))
    zlims!(ax, (0, L))

    # Makie.record(fig, "test_bubble.mp4", 1:nstep) do itime
    for itime in 1:nstep
        for isub in 1:nsubstep
            # Perform one RK3 step of step size `dt`
            rk3step_SH!(F, U0, U, t, dt, fractions, sdf, p, setup, Bub, Precomp_SH)
            t += dt
            @info "bubble centroid = $(Bub.centr), mean z-velocity = $(sum(U.u[:, :, :, 3])/(3*size(U.u)[1]))"

            xbub, ybub, zbub = get_bubble(Bub, Precomp_SH)
            println("min x: $(minimum(xbub)), max x: $(maximum(xbub))")
            println("min y: $(minimum(ybub)), max y: $(maximum(ybub))")
            println("min z: $(minimum(zbub)), max z: $(maximum(zbub))")
            # println("coefs: $(Bub.c)")

            # @info "itime = $itime / $nstep, isub = $isub / $nsubstep, t = $(round(t, digits = 4))" # maximum(abs, U.u)
        end

        xv, yv, zv = setup.xp[1][2:end-1], setup.xp[2][2:end-1], setup.xp[3][2:end-1]
        uv = map_velocity(U.u, setup, xv, yv, zv)
        ps = [Point3f(x, y, z) for x in xv for y in yv for z in zv]
        ns = [Vec3f(u1, u2, u3) for u1 in uv[1:end, 1] for u2 in uv[1:end, 2] for u3 in uv[1:end, 3]]
        mag = norm.(ns)
        ns .= ns / maximum(mag) * .002
        mag[mag .< .9 * maximum(mag)] .= NaN

        # xb, yb, zb = get_bubble(Bub, Precomp_SH)
        rbub = Yvis * Bub.c
        centr = Bub.centr
        xb, yb, zb = centr[1] .+ rbub .* sin.(vec(ϕvis)) .* cos.(vec(θvis)),
                            centr[2] .+ rbub .* sin.(vec(ϕvis)) .* sin.(vec(θvis)), 
                            centr[3] .+ rbub .* cos.(vec(ϕvis))

        empty!(ax)

        arrows3d!(ax, ps, ns, color = mag, colormap = :viridis, alpha = .2, lengthscale=.5)
        surface!(ax, reshape(xb, nvis, nvis), reshape(yb, nvis, nvis), reshape(zb, nvis, nvis), 
                color = ones(Float64, (nvis, nvis)), colormap = :magma)

        save("plots/snapshot_t=$(round(t, digits = 4)).png", fig)

        @info "itime = $itime / $nstep, t = $(round(t, digits = 4))" # maximum(abs, U.u)
    end

    return U
end
