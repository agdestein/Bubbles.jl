"""
Functionality for dynamically representing bubbles using a linear combination of spherical harmonics as basis functions, and a centroid position.
The basis function coefficients and the bubble centroid are updated by a given fluid velocity field. Two prescribed velocity fields are included as test cases.

Current functionality inludes:
    - Evaluating spherical harmonics and their partial derivatives at given points 
    - Using spherical design cubature rules to efficiently numerically integrate integrals over the unit sphere
    - RK4 time stepping to update the bubble centroid position and shape coefficients

To be implemented:
    - Computation of surface tension forces at given positions on the bubble surface
    - Finding the intersection of a bubble with a given grid line
    - Numerical integration over a given part of the bubble surface
    - Bubble visualization as a closed surface, also over time
"""

using SphericalHarmonics, SphericalHarmonicModes
using DelimitedFiles
using Makie, WGLMakie


function cart2spc(x, y, z)
    r = sqrt.(x.^2 + y.^2 + z.^2)
    ϕ = acos.(z ./ r)
    θ = atan.(y, x)
    return r, ϕ, θ
end

function spc2cart(r, ϕ, θ)
    x, y, z = r .* sin.(ϕ) .* cos.(θ), r .* sin.(ϕ) .* sin.(θ), r .* cos.(ϕ)
    return x, y, z
end

function get_points_spc(npoints)
    points = readdlm("src/cub/sd$(npoints).txt")
    x, y, z = points[:,1], points[:,2], points[:,3]
    return cart2spc(x, y, z)
end

function get_SH(ℓₘ, ϕ, θ)
    nbf = (ℓₘ + 1)^2  # number of basis functions (spherical harmonics)

    Ytemp = SphericalHarmonics.computeYlm.(ϕ, θ, lmax = ℓₘ, 
                                        SHType = SphericalHarmonics.RealHarmonics())
    Y = zeros(Float64, length(ϕ), nbf) # convert to matrix

    for i = 1:length(ϕ)
        Y[i, :] .= Ytemp[i][:]
    end
    Ytemp = 0   # free

    return Y
end

function get_SH_der(ℓₘ, ϕ, θ)
    nbf = (ℓₘ + 1)^2  # number of basis functions (spherical harmonics)

    modes = ML(0:ℓₘ)                # (ℓ, m) tuples
    ℓs = first.(modes)              # ℓ values, in storage order
    ms = last.(modes)               # m values, in storage order
    ms_nonneg = last.(ML(0:ℓₘ, 0:ℓₘ))   # only nonnegative m values, in storage order
    pos = findall(m -> m > 0, ms); neg = findall(m -> m < 0, ms); zero = findall(m -> m == 0, ms)
    one = findall(m -> m == 1, ms); mone = findall(m -> m == -1, ms)
    nonneg = findall(m -> m >= 0, ms); ℓs_nonneg = ℓs[nonneg]
    nonneg_pos = findall(m -> m > 0, ms_nonneg); nonneg_zero = findall(m -> m == 0, ms_nonneg)

    nonneg_neg = zeros(Int, (ℓₘ^2 + ℓₘ) ÷ 2)    # correct order for negative m (most negative first)
    m_flip = zeros(Int, length(ms))
    m_flip[1] = 1
    let i = 1; j = 2
        for ℓ in 1:ℓₘ 
            nonneg_neg[i:i + ℓ - 1] .= i + ℓ + ℓ : -1 : i + ℓ + 1
            m_flip[j:j + 2 * ℓ] .= j + 2 * ℓ : -1 : j
            i = i + ℓ
            j = j + 2 * ℓ + 1
        end
    end

    Ytemp = SphericalHarmonics.computeYlm.(ϕ, θ, lmax = ℓₘ, 
                                        SHType = SphericalHarmonics.RealHarmonics())
    Ptemp = SphericalHarmonics.computePlmcostheta.(ϕ, ℓₘ)   # only nonnegative m by default

    Y = zeros(Float64, length(ϕ), nbf) # convert to matrix
    P = zeros(Float64, length(ϕ), length(ms_nonneg))

    for i = 1:length(ϕ)
        Y[i, :] .= Ytemp[i][:]
        P[i, :] .= Ptemp[i][:]
    end

    Ytemp = 0; Ptemp = 0    # free

    dY_dϕ = zeros(Float64, length(ϕ), nbf)
    dY_dϕ[:, zero] .= sqrt.(ℓs[zero] .* (ℓs[zero] .+ 1) / (2. * π))' .* P[:, nonneg_zero .+ 1]

    dY_dϕ[:, neg] .= sin.(abs.(ms[neg])' .* θ[:, :]) .* (
            sqrt.((ℓs[neg] .- abs.(ms[neg])) .* (ℓs[neg] .+ abs.(ms[neg]) .+ 1))' .* P[:, mod.(nonneg_neg .+ 1, ℓₘ * (ℓₘ + 1) ÷ 2 .+ ℓₘ .+ 1) .+ 1] .* (ℓs[neg] .!= abs.(ms[neg]))'
            - sqrt.((ℓs[neg] .+ abs.(ms[neg])) .* (ℓs[neg] .- abs.(ms[neg]) .+ 1))' .* P[:, nonneg_neg]
        ) / (2. * sqrt(π))
    dY_dϕ[:, pos] .= cos.(ms[pos]' .* θ[:, :]) .* (
            sqrt.((ℓs[pos] - ms[pos]) .* (ℓs[pos] + ms[pos] .+ 1))' .* P[:, mod.(nonneg_pos .+ 1, ℓₘ * (ℓₘ + 1) ÷ 2 .+ ℓₘ .+ 1) .+ 1] .* (ℓs[pos] .!= ms[pos])'
            - sqrt.((ℓs[pos] + ms[pos]) .* (ℓs[pos] - ms[pos] .+ 1))' .* P[:, nonneg_pos]
        ) / (2. * sqrt(π))

    dY_dθ = - ms' .* Y[:, m_flip]

    return Y, dY_dϕ, dY_dθ, ℓs, ms, one, mone
end

function K_lm(ℓ, m)
    return sqrt.((2. * ℓ .+ 1)/(4. * π) .* factorial.(ℓ - m) ./ factorial(ℓ + m))
end

function get_uz_spc(c, Y, umax, uz, ϕ, θ)
    r = Y * c 
    u_z = umax[3] * uz(r .* sin.(ϕ) .* cos.(θ))
    return u_z
end

function get_uz_cart(umax, uz, x)
    u_z = umax[3] * uz(x)
    return u_z
end

function get_ux_spc(c, Y, umax, ux, ϕ, θ)
    r = Y * c 
    u_x = umax[1] * ux(r .* sin.(ϕ) .* sin.(θ))
    return u_x
end

function get_ux_cart(umax, ux, y)
    u_x = umax[1] * ux(y)
    return u_x
end

function time_step_z(c, Y, dY_dϕ, ϕ, u_z, npoints, V)
    r = Y * c
    dr_dϕ = dY_dϕ * c

    # Bubble centroid velocity:
    u_centr = 4. * π / npoints / V * sum(r .* (r .* u_z .* cos.(ϕ) + dr_dϕ .* u_z .* sin.(ϕ)) .* r .* cos.(ϕ))

    # Spherical harmonics coefficients 'velocity'
    dϕ_dt = (- sin.(ϕ) .* (u_z .- u_centr)) ./ r
    dc_dt = 4. * π / npoints * ((cos.(ϕ) .* (u_z .- u_centr) - dϕ_dt .* dr_dϕ)' * Y)[1, :]
    return dc_dt, u_centr
end

function time_step_x(c, Y, dY_dϕ, dY_dθ, ϕ, θ, u_x, npoints, V, ℓs, ms, one, mone)
    r = Y * c 
    dr_dϕ = dY_dϕ * c 
    dr_dθ = dY_dθ * c 

    # Bubble centroid velocity:
    u_centr = 4. * π / npoints / V * sum(r .* u_x .* (r .* sin.(ϕ).^2 .* cos.(θ) - dr_dϕ .* cos.(ϕ) .* cos.(θ) .* sin.(ϕ) + sin.(θ) .* dr_dθ) .* r .* cos.(θ))

    # Spherical harmonics coefficients 'velocity'
    dϕ_dt = cos.(ϕ) .* cos.(θ) .* (u_x .- u_centr) ./ r 
    θ_term = similar(dϕ_dt)
    θ_term[2:end] .= - (u_x[2:end] .- u_centr) .* sin.(θ[2:end]) ./ (r[2:end] .* sin.(ϕ[2:end])) .* dr_dθ[2:end]
    θ_term[1] = - (u_x[1] - u_centr) * sin(θ[1]) / r[1] * sum(K_lm.(ℓs[one], ms[one]) .* ℓs[one] .* (ℓs[one] .+ 1) / 2. .* c[one] * sin(θ[1]) - 
                                                                K_lm.(ℓs[one], -ms[mone]) .* ℓs[mone] .* (ℓs[mone] .+ 1) / 2. .* c[mone] * cos(θ[1]))
    dc_dt = 4. * π / npoints * ((sin.(ϕ) .* cos.(θ) .* (u_x .- u_centr))' * Y - (dϕ_dt .* dr_dϕ)' * Y - θ_term' * Y)[1, :]
    return dc_dt, u_centr
end

function parabolic_z(c, centr, Y, dY_dϕ, ϕ, θ, u, umax, npoints, V, dt, nt, Y_test, ϕ_test, θ_test)
    r0 = Y_test * c 
    x0, y0, z0 = spc2cart(r0, ϕ_test, θ_test)
    x0, y0, z0 = x0 .+ centr[1], y0 .+ centr[2], z0 .+ centr[3]

    for _ in 1:nt   # RK4 time stepping
        k1, k1_centr = time_step_z(c, Y, dY_dϕ, ϕ, get_uz_spc(c, Y, umax, u, ϕ, θ), npoints, V)
        k2, k2_centr = time_step_z(c + k1 * dt / 2., Y, dY_dϕ, ϕ, get_uz_spc(c + k1 * dt / 2., Y, umax, u, ϕ, θ), npoints, V)
        k3, k3_centr = time_step_z(c + k2 * dt / 2., Y, dY_dϕ, ϕ, get_uz_spc(c + k2 * dt / 2., Y, umax, u, ϕ, θ), npoints, V)
        k4, k4_centr = time_step_z(c + k3 * dt, Y, dY_dϕ, ϕ, get_uz_spc(c + k3 * dt, Y, umax, u, ϕ, θ), npoints, V)

        c = c + (k1 + 2. * k2 + 2. * k3 + k4) * dt / 6.
        centr[3] = centr[3] .+ (k1_centr + 2. * k2_centr + 2. * k3_centr + k4_centr) * dt / 6.
    end
    
    r = Y_test * c 
    x, y, z = r .* sin.(ϕ_test) .* cos.(θ_test), r .* sin.(ϕ_test) .* sin.(θ_test), centr[3] .+ r .* cos.(ϕ_test)
    z_ref = z0 .+ get_uz_cart(umax, u, x0) * dt * nt    # for computing errors later

    return x0, y0, z0, x, y, z
end

function linear_x(c, centr, Y, dY_dϕ, dY_dθ, ϕ, θ, u, umax, npoints, V, dt, nt, Y_test, ϕ_test, θ_test, ℓs, ms, one, mone)
    r0 = Y_test * c 
    x0, y0, z0 = spc2cart(r0, ϕ_test, θ_test)
    x0, y0, z0 = x0 .+ centr[1], y0 .+ centr[2], z0 .+ centr[3]

    for _ in 1:nt   # RK4 time stepping
        k1, k1_centr = time_step_x(c, Y, dY_dϕ, dY_dθ, ϕ, θ, get_ux_spc(c, Y, umax, u, ϕ, θ), npoints, V, ℓs, ms, one, mone)
        k2, k2_centr = time_step_x(c + k1 * dt / 2., Y, dY_dϕ, dY_dθ, ϕ, θ, get_ux_spc(c + k1 * dt / 2., Y, umax, u, ϕ, θ), npoints, V, ℓs, ms, one, mone)
        k3, k3_centr = time_step_x(c + k2 * dt / 2., Y, dY_dϕ, dY_dθ, ϕ, θ, get_ux_spc(c + k2 * dt / 2., Y, umax, u, ϕ, θ), npoints, V, ℓs, ms, one, mone)
        k4, k4_centr = time_step_x(c + k3 * dt, Y, dY_dϕ, dY_dθ, ϕ, θ, get_ux_spc(c + k3 * dt, Y, umax, u, ϕ, θ), npoints, V, ℓs, ms, one, mone)

        c = c + (k1 + 2. * k2 + 2. * k3 + k4) * dt / 6.
        centr[1] = centr[1] .+ (k1_centr + 2. * k2_centr + 2. * k3_centr + k4_centr) * dt / 6.
    end
    
    r = Y_test * c 
    x, y, z = centr[1] .+ r .* sin.(ϕ_test) .* cos.(θ_test), r .* sin.(ϕ_test) .* sin.(θ_test), r .* cos.(ϕ_test)
    x_ref = x0 .+ get_ux_cart(umax, u, y0) * dt * nt    # for computing errors later

    return x0, y0, z0, x, y, z
end