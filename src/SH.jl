"""
Functionality for dynamically representing bubbles using a linear combination of spherical harmonics as basis functions, and a centroid position.
The basis function coefficients and the bubble centroid are updated by a given fluid velocity field. Two prescribed velocity fields are included as test cases.

Current functionality inludes:
    - Evaluating spherical harmonics and their partial derivatives at given points 
    - Using spherical design cubature rules to efficiently numerically integrate integrals over the unit sphere
    - RK4 time stepping to update the bubble centroid position and shape coefficients
    - Computation of the total bubble volume and surface area from spherical harmonics 
    - Evaluation of bubble surface curvature at spherical design cubature points 
    - Evaluation of total pressure drop over the bubble surface

To be implemented:
    - Computation of surface tension forces at given positions on the bubble surface
    - Finding the intersection of a bubble with a given grid line
    - Numerical integration over a given part of the bubble surface
    - Bubble visualization as a closed surface, also over time
"""

using SphericalHarmonics, SphericalHarmonicModes
using DelimitedFiles

"""
Convert Cartesian coordinates (x, y, z) to spherical (r, ϕ, θ).
"""
function cart2spc(x, y, z)
    r = sqrt.(x.^2 + y.^2 + z.^2)
    ϕ = acos.(z ./ r)
    θ = atan.(y, x)
    return r, ϕ, θ
end

"""
Convert spherical coordinates (r, ϕ, θ) to Cartesian (x, y, z).
"""
function spc2cart(r, ϕ, θ)
    x, y, z = r .* sin.(ϕ) .* cos.(θ), r .* sin.(ϕ) .* sin.(θ), r .* cos.(ϕ)
    return x, y, z
end

"""
Get spherical design cubature point set of cardinality 'npoints', where 'npoints' ∈ {50, 201, 513, 1059, 2049, 4051, 8066, 16382}.
Return the points in spherical coordinates (r, ϕ, θ).
"""
function get_points_spc(npoints)
    points = readdlm("src/cub/sd$(npoints).txt")
    x, y, z = points[:,1], points[:,2], points[:,3]
    return cart2spc(x, y, z)
end

"""
Evaluate all spherical harmonics up to and including order ℓₘ at spherical design cubature points (ϕ, θ). 
"""
function get_SH(ℓₘ, ϕ, θ)
    nbf = (ℓₘ + 1)^2  # number of basis functions (spherical harmonics)

    Ytemp = SphericalHarmonics.computeYlm.(ϕ, θ, lmax = ℓₘ, 
                                        SHType = SphericalHarmonics.RealHarmonics())
    Y = zeros(Float64, length(ϕ), nbf) # convert to matrix

    if length(size(ϕ)) == 0
        return Ytemp
    else
        for i = 1:length(ϕ)
            Y[i, :] .= Ytemp[i][:]
        end
        Ytemp = 0   # free

        return Y
    end
end

"""
Evaluate all spherical harmonics up to and including order ℓₘ at spherical design cubature points (ϕ, θ). 
Also evaluates first order partial derivatives of the spherical harmonics.
"""
function get_SH_der(ℓₘ, ϕ, θ)
    nbf = (ℓₘ + 1)^2  # number of basis functions (spherical harmonics)

    modes = ML(0:ℓₘ)                # (ℓ, m) tuples
    ℓs = first.(modes)              # ℓ values, in storage order
    ms = last.(modes)               # m values, in storage order
    pos = findall(m -> m > 0, ms); neg = findall(m -> m < 0, ms); zero = findall(m -> m == 0, ms)
    one = findall(m -> m == 1, ms); mone = findall(m -> m == -1, ms)
    nonneg = findall(m -> m >= 0, ms); ms_nonneg = ms[nonneg]
    nonneg_pos = findall(m -> m > 0, ms_nonneg); nonneg_zero = findall(m -> m == 0, ms_nonneg)

    nonneg_neg = zeros(Int, (ℓₘ^2 + ℓₘ) ÷ 2)    # correct order for negative m (most negative first)
    m_flip = zeros(Int, length(ms))             # points to -m at position of m, for dY_dθ
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
    dY_dϕ[:, zero] .= sqrt.(ℓs[zero] .* (ℓs[zero] .+ 1) / 2.)' .* P[:, nonneg_zero .+ 1]

    dY_dϕ[:, neg] .= sin.(abs.(ms[neg])' .* θ[:, :]) .* (
            sqrt.((ℓs[neg] .- abs.(ms[neg])) .* (ℓs[neg] .+ abs.(ms[neg]) .+ 1))' .* P[:, clamp.(nonneg_neg .+ 1, 1, (ℓₘ * (ℓₘ + 1)) ÷ 2 .+ ℓₘ .+ 1)] .* (ℓs[neg] .!= abs.(ms[neg]))'
            - sqrt.((ℓs[neg] .+ abs.(ms[neg])) .* (ℓs[neg] .- abs.(ms[neg]) .+ 1))' .* P[:, nonneg_neg .- 1]
        ) / 2.
    dY_dϕ[:, pos] .= cos.(ms[pos]' .* θ[:, :]) .* (
            sqrt.((ℓs[pos] - ms[pos]) .* (ℓs[pos] + ms[pos] .+ 1))' .* P[:, clamp.(nonneg_pos .+ 1, 1, (ℓₘ * (ℓₘ + 1)) ÷ 2 .+ ℓₘ .+ 1)] .* (ℓs[pos] .!= ms[pos])'
            - sqrt.((ℓs[pos] + ms[pos]) .* (ℓs[pos] - ms[pos] .+ 1))' .* P[:, nonneg_pos .- 1]
        ) / 2.

    dY_dθ = - ms' .* Y[:, m_flip]

    return Y, dY_dϕ, dY_dθ, ℓs, ms, one, mone, zero
end

"""
Evaluate all spherical harmonics up to and including order ℓₘ at spherical design cubature points (ϕ, θ). 
Also evaluates first and second order partial derivatives of the spherical harmonics.
"""
function get_SH_der2(ℓₘ, ϕ, θ)
    nbf = (ℓₘ + 1)^2  # number of basis functions (spherical harmonics)
    ϕv = ndims(ϕ) == 0 ? [ϕ] : vec(ϕ)
    θv = ndims(θ) == 0 ? [θ] : vec(θ)
    length(ϕv) == length(θv) || throw(ArgumentError("ϕ and θ must have the same length"))
    θmat = reshape(θv, :, 1)

    modes = ML(0:ℓₘ)                # (ℓ, m) tuples
    ℓs = first.(modes)              # ℓ values, in storage order
    ms = last.(modes)               # m values, in storage order
    pos = findall(m -> m > 0, ms); neg = findall(m -> m < 0, ms); zero = findall(m -> m == 0, ms)
    one = findall(m -> m == 1, ms); mone = findall(m -> m == -1, ms)
    two = findall(m -> m == 2, ms); mtwo = findall(m -> m == -2, ms)
    nonneg = findall(m -> m >= 0, ms); ms_nonneg = ms[nonneg]
    nonneg_pos = findall(m -> m > 0, ms_nonneg); nonneg_zero = findall(m -> m == 0, ms_nonneg)

    nonneg_neg = zeros(Int, (ℓₘ^2 + ℓₘ) ÷ 2)    # correct order for negative m (most negative first)
    m_flip = zeros(Int, length(ms))             # points to -m at position of m, for dY_dθ
    m_flip[1] = 1
    let i = 1; j = 2
        for ℓ in 1:ℓₘ 
            nonneg_neg[i:i + ℓ - 1] .= i + ℓ + ℓ : -1 : i + ℓ + 1
            m_flip[j:j + 2 * ℓ] .= j + 2 * ℓ : -1 : j
            i = i + ℓ
            j = j + 2 * ℓ + 1
        end
    end

    Ytemp = SphericalHarmonics.computeYlm.(ϕv, θv, lmax = ℓₘ, 
                                        SHType = SphericalHarmonics.RealHarmonics())
    Ptemp = SphericalHarmonics.computePlmcostheta.(ϕv, ℓₘ)   # only nonnegative m by default

    Y = zeros(Float64, length(ϕv), nbf) # convert to matrix
    P = zeros(Float64, length(ϕv), length(ms_nonneg))

    for i = 1:length(ϕv)
        Y[i, :] .= Ytemp[i][:]
        P[i, :] .= Ptemp[i][:]
    end

    Ytemp = 0; Ptemp = 0    # free

    dY_dϕ = zeros(Float64, length(ϕv), nbf)
    dY_dϕ[:, zero] .= sqrt.(ℓs[zero] .* (ℓs[zero] .+ 1) / 2.)' .* P[:, clamp.(nonneg_zero .+ 1, 1, (ℓₘ * (ℓₘ + 1)) ÷ 2 .+ ℓₘ .+ 1)]
    Y_ϕ_neg = (
            sqrt.((ℓs[neg] .- abs.(ms[neg])) .* (ℓs[neg] .+ abs.(ms[neg]) .+ 1))' .* P[:, clamp.(nonneg_neg .+ 1, 1, (ℓₘ * (ℓₘ + 1)) ÷ 2 .+ ℓₘ .+ 1)] .* (ℓs[neg] .!= abs.(ms[neg]))'
            - sqrt.((ℓs[neg] .+ abs.(ms[neg])) .* (ℓs[neg] .- abs.(ms[neg]) .+ 1))' .* P[:, nonneg_neg .- 1]
        ) / 2.
    Y_ϕ_pos = (
            sqrt.((ℓs[pos] - ms[pos]) .* (ℓs[pos] + ms[pos] .+ 1))' .* P[:, clamp.(nonneg_pos .+ 1, 1, (ℓₘ * (ℓₘ + 1)) ÷ 2 .+ ℓₘ .+ 1)] .* (ℓs[pos] .!= ms[pos])'
            - sqrt.((ℓs[pos] + ms[pos]) .* (ℓs[pos] - ms[pos] .+ 1))' .* P[:, nonneg_pos .- 1]
        ) / 2.
    dY_dϕ[:, neg] .= sin.(abs.(ms[neg])' .* θmat) .* Y_ϕ_neg
    dY_dϕ[:, pos] .= cos.(ms[pos]' .* θmat) .* Y_ϕ_pos

    dY_dθ = - ms' .* Y[:, m_flip]

    d²Y_dθ² = - ((ms).^2)' .* Y

    d²Y_dθdϕ = zeros(Float64, length(ϕv), nbf)
    d²Y_dθdϕ[:, neg] .= abs.(ms[neg])' .* cos.(abs.(ms[neg])' .* θmat) .* Y_ϕ_neg
    d²Y_dθdϕ[:, pos] .= - ms[pos]' .* sin.(ms[pos]' .* θmat) .* Y_ϕ_pos

    d²Y_dϕ² = zeros(Float64, length(ϕv), nbf)
    d²Y_dϕ²[:, neg] .= sin.(abs.(ms[neg])' .* θmat) .* (sqrt.((ℓs[neg] - abs.(ms[neg]) .- 1) .* (ℓs[neg] - abs.(ms[neg])) .* (ℓs[neg] + abs.(ms[neg]) .+ 1) .* (ℓs[neg] + abs.(ms[neg]) .+ 2))' .* 
                        P[:, clamp.(nonneg_neg .+ 2, 1, (ℓₘ * (ℓₘ + 1)) ÷ 2 .+ ℓₘ .+ 1)] .* (abs.(ms[neg]) .+ 2 .≤ ℓs[neg])'
                        - 2. * ((ℓs[neg] + abs.(ms[neg])) .* (ℓs[neg] - abs.(ms[neg])) + ℓs[neg])' .* P[:, nonneg_neg]
                        - ((ℓs[neg] + abs.(ms[neg])) .* (ℓs[neg] - abs.(ms[neg]) .+ 1))' .* P[:, nonneg_neg] .* (abs.(ms[neg]) .== 1)'
                        + sqrt.((ℓs[neg] + abs.(ms[neg]) .- 1) .* (ℓs[neg] + abs.(ms[neg])) .* (ℓs[neg] - abs.(ms[neg]) .+ 1) .* (ℓs[neg] - abs.(ms[neg]) .+ 2))' .*
                        P[:, clamp.(nonneg_neg .- 2, 1, (ℓₘ * (ℓₘ + 1)) ÷ 2 .+ ℓₘ .+ 1)] .* (abs.(ms[neg]) .!= 1)'
        ) / 4.
    d²Y_dϕ²[:, zero] .= (sqrt.((ℓs[zero] .+ 2) .* (ℓs[zero] .+ 1) .* ℓs[zero] .* (ℓs[zero] .- 1))' .* P[:, clamp.(nonneg_zero .+ 2, 1, (ℓₘ * (ℓₘ + 1)) ÷ 2 .+ ℓₘ .+ 1)]
                        - (ℓs[zero] .* (ℓs[zero] .+ 1))' .* P[:, nonneg_zero]
        ) / (2. * sqrt(2.))
    d²Y_dϕ²[:, pos] .= cos.(ms[pos]' .* θmat) .* (sqrt.((ℓs[pos] - ms[pos] .- 1) .* (ℓs[pos] - ms[pos]) .* (ℓs[pos] + ms[pos] .+ 1) .* (ℓs[pos] + ms[pos] .+ 2))' .* 
                        P[:, clamp.(nonneg_pos .+ 2, 1, (ℓₘ * (ℓₘ + 1)) ÷ 2 .+ ℓₘ .+ 1)] .* (ms[pos] .+ 2 .≤ ℓs[pos])'
                        - 2. * ((ℓs[pos] + ms[pos]) .* (ℓs[pos] - ms[pos]) + ℓs[pos])' .* P[:, nonneg_pos] 
                        - ((ℓs[pos] + ms[pos]) .* (ℓs[pos] - ms[pos] .+ 1))' .* P[:, nonneg_pos] .* (ms[pos] .== 1)'
                        + sqrt.((ℓs[pos] + ms[pos] .- 1) .* (ℓs[pos] + ms[pos]) .* (ℓs[pos] - ms[pos] .+ 1) .* (ℓs[pos] - ms[pos] .+ 2))' .*
                        P[:, clamp.(nonneg_pos .- 2, 1, (ℓₘ * (ℓₘ + 1)) ÷ 2 .+ ℓₘ .+ 1)] .* (ms[pos] .!= 1)'
        ) / 4.

    return (; Y, dY_dϕ, dY_dθ, d²Y_dϕ², d²Y_dθdϕ, d²Y_dθ², ℓs, ms, one, mone, zero, two, mtwo)
end

function bubble_setup(ncub, ℓₘ, R, σ, L)
    # Spherical design cubature points:
    _, ϕ, θ = get_points_spc(ncub)

    # Bubble initialization:
    c = zeros(Float64, (ℓₘ + 1) ^ 2)    # spherical harmonics coefficients
    c[1] = R * sqrt(4. * π)

    # Oscillating drop test case (Lamb, 1932):
    c[7] = sin(5e-3)    # ℓ=2, m=0

    # Precomputed spherical harmonics (derivatives) at spherical design cubature points:
    (; Y, dY_dϕ, dY_dθ, d²Y_dϕ², d²Y_dθdϕ, d²Y_dθ², ℓs, ms, one, mone, zero, two, mtwo) = get_SH_der2(ℓₘ, ϕ, θ)
    Precomp_SH = (; ϕ, θ, Y, dY_dϕ, dY_dθ, d²Y_dϕ², d²Y_dθdϕ, d²Y_dθ², ℓs, ms, one, mone, zero, two, mtwo)

    centr = [L/2, L/2, L/2]           # centroid position
    V = volume(Y2r((; c), Precomp_SH)) # total bubble volume from current SH shape
    Bub = (; c, centr, V, σ)

    return Bub, Precomp_SH
end

function fit_coefs_LS(Y, r)
    c = Y \ r
    return c
end

function fit_coefs_orth(Y, r)
    c = 4π/length(r) * (Y' * r) # Y has size (N, #basis functions), r has size (N)
    return c
end

"""
Evaluate spherical harmonic normalization constant for all degrees ℓ at order m=0.
"""
function K_lzero(ℓ)
    return sqrt.((2. * ℓ .+ 1)/(4. * π))
end

"""
Evaluate spherical harmonic normalization constant for all degrees ℓ at order m=1.
"""
function K_lone(ℓ)
    return sqrt.((2. * ℓ .+ 1)/(4. * π) ./ (ℓ .* (ℓ .+ 1)))
end

function K_ltwo(ℓ)
    return sqrt.((2. * ℓ .+ 1)/(4. * π) ./ (ℓ .* (ℓ .+ 1) .* (ℓ .- 1) .* (ℓ .+ 2)))
end

# function K_lmone(ℓ)
#     return sqrt.((2. * ℓ .+ 1)/(4. * π) .* (ℓ .* (ℓ .+ 1)))
# end

"""
Compute total bubble volume.
"""
function volume(Dynamic_SH) 
    return 4. * π / length(Dynamic_SH.r) * sum(Dynamic_SH.r .^ 3) / 3.
end

"""
Evaluate relevant limits at singularities at the north pole (first spherical design cubature point: ϕ = θ = 0).
"""
function northpole(Bub, Precomp_SH)
    c = Bub.c 
    ℓs, one, mone, zero, two, mtwo, θ = Precomp_SH.ℓs, Precomp_SH.one, Precomp_SH.mone, Precomp_SH.zero, Precomp_SH.two, Precomp_SH.mtwo, Precomp_SH.θ

    dr_dθ_div_sinϕ = sqrt(2) * sum(K_lone.(ℓs[one]) .* ℓs[one] .* (ℓs[one] .+ 1) / 2. .* c[one] * sin(θ[1]) - 
                        K_lone.(ℓs[mone]) .* ℓs[mone] .* (ℓs[mone] .+ 1) / 2. .* c[mone] * cos(θ[1]))
    # d²r_dθ²_div_sinϕ = sum(K_lone.(ℓs[one]) .* ℓs[one] .* (ℓs[one] .+ 1) / 2. .* c[one] * cos(θ[1]) +
    #                     K_lone.(ℓs[mone]) .* ℓs[mone] .* (ℓs[mone] .+ 1) / 2. .* c[mone] * sin(θ[1]))
    EN_lim = - sum(c[zero] .* K_lzero(ℓs[zero]) .* ℓs[zero] .* (ℓs[zero] .+ 1) / 2.) - sqrt(2.0) * sum(
        c[two] .* K_ltwo(ℓs[two]) .* (ℓs[two] .+ 2) .* (ℓs[two] .+ 1) .* ℓs[two] .* (ℓs[two] .- 1) / 4. .* cos(2. * θ[1])
        + c[mtwo] .* K_ltwo(ℓs[mtwo]) .* (ℓs[mtwo] .+ 2) .* (ℓs[mtwo] .+ 1) .* ℓs[mtwo] .* (ℓs[mtwo] .- 1) / 4. .* sin(2. * θ[1])
    )
    FM_lim = sqrt(2.0) * sum(
        - c[two] .* K_ltwo(ℓs[two]) .* (ℓs[two] .+ 2) .* (ℓs[two] .+ 1) .* ℓs[two] .* (ℓs[two] .- 1) / 4. .* sin(2. * θ[1])
        + c[mtwo] .* K_ltwo(ℓs[mtwo]) .* (ℓs[mtwo] .+ 2) .* (ℓs[mtwo] .+ 1) .* ℓs[mtwo] .* (ℓs[mtwo] .- 1) / 4. .* cos(2. * θ[1])
    )
    
    return dr_dθ_div_sinϕ, EN_lim, FM_lim
end

function Y2r(Bub, Precomp_SH)
    r = Precomp_SH.Y * Bub.c
    dr_dϕ = Precomp_SH.dY_dϕ * Bub.c 
    dr_dθ = Precomp_SH.dY_dθ * Bub.c 
    d²r_dϕ² = Precomp_SH.d²Y_dϕ² * Bub.c 
    d²r_dϕdθ = Precomp_SH.d²Y_dθdϕ * Bub.c 
    d²r_dθ² = Precomp_SH.d²Y_dθ² * Bub.c 
    dr_dθ_div_sinϕ, EN_lim, FM_lim = northpole(Bub, Precomp_SH)

    Dynamic_SH = (; r, dr_dϕ, dr_dθ, d²r_dϕ², d²r_dϕdθ, d²r_dθ², dr_dθ_div_sinϕ, EN_lim, FM_lim)

    return Dynamic_SH
end

"""
Evaluate outwards facing unit normal at spherical design cubature points, in both spherical and Cartesian coordinates.
"""
function unit_normal(Precomp_SH, Dynamic_SH)
    ϕ, θ = Precomp_SH.ϕ, Precomp_SH.θ
    r, dr_dϕ, dr_dθ, dr_dθ_div_sinϕ = Dynamic_SH.r, Dynamic_SH.dr_dϕ, Dynamic_SH.dr_dθ, Dynamic_SH.dr_dθ_div_sinϕ

    n_length = zeros(Float64, length(ϕ))
    n_length[2:end] .= sqrt.(r[2:end] .^ 2 + dr_dϕ[2:end] .^ 2 + (dr_dθ[2:end] ./ sin.(ϕ[2:end])) .^2)
    n_length[1] = sqrt(r[1] ^ 2 + dr_dϕ[1] ^ 2 + dr_dθ_div_sinϕ ^ 2) 
    n_r = r ./ n_length
    n_ϕ = dr_dϕ ./ n_length 
    n_θ = zeros(Float64, length(ϕ))
    n_θ[2:end] .= dr_dθ[2:end] ./ sin.(ϕ[2:end]) ./ n_length[2:end]
    n_θ[1] = dr_dθ_div_sinϕ / n_length[1]
    # println("length: $(n_length[1]), div: $(dr_dθ_div_sinϕ), n_θ: $(n_θ[1])")

    # n_x, n_y, n_z = similar(n_r), similar(n_r), similar(n_r)
    n_x = r .* sin.(ϕ) .* cos.(θ) - dr_dϕ .* cos.(ϕ) .* cos.(θ)
    n_x[2:end] .= n_x[2:end] + dr_dθ[2:end] ./ sin.(ϕ[2:end]) .* sin.(θ[2:end])
    n_x[1] = n_x[1] + dr_dθ_div_sinϕ * sin(θ[1])

    n_y = r .* sin.(ϕ) .* sin.(θ) - dr_dϕ .* cos.(ϕ) .* sin.(θ)
    n_y[2:end] .= n_y[2:end] - dr_dθ[2:end] ./ sin.(ϕ[2:end]) .* cos.(θ[2:end])
    n_y[1] = n_y[1] - dr_dθ_div_sinϕ * cos(θ[1])

    n_z = r .* cos.(ϕ) + dr_dϕ .* sin.(ϕ)

    n_x .= n_x ./ n_length
    n_y .= n_y ./ n_length
    n_z .= n_z ./ n_length

    return n_r, n_ϕ, n_θ, n_x, n_y, n_z
end

"""
Evaluate differential surface element divided by sin(ϕ) at spherical design cubature points.
"""
function surface_element(Precomp_SH, Dynamic_SH)
    ϕ = Precomp_SH.ϕ
    r, dr_dϕ, dr_dθ, dr_dθ_div_sinϕ = Dynamic_SH.r, Dynamic_SH.dr_dϕ, Dynamic_SH.dr_dθ, Dynamic_SH.dr_dθ_div_sinϕ

    dS = zeros(Float64, length(ϕ))
    dS[2:end] .= r[2:end] .* sqrt.(r[2:end] .^ 2 .+ dr_dϕ[2:end] .^ 2 .+ (dr_dθ[2:end] ./ sin.(ϕ[2:end])) .^ 2)
    if abs(ϕ[1]) < 1e-10  # north pole: use limit
        dS[1] = r[1] * sqrt(r[1] ^ 2 + dr_dϕ[1] ^ 2 + dr_dθ_div_sinϕ ^ 2)
    else    # scalar call (single element vector)
        dS[1] = r[1] * sqrt(r[1] ^ 2 + dr_dϕ[1] ^ 2 + (dr_dθ[1] / sin(ϕ[1])) ^ 2)
    end

    return dS
end

"""
Compute total bubble surface area.
"""
function surface_area(dS, ϕ)
    S = 4. * π / length(ϕ) * sum(dS)
    return S 
end

"""
Evaluate twice the local mean curvature (2H = κ) at spherical design cubature points.
"""
function surface_curvature(Precomp_SH, Dynamic_SH, dS)
    ϕ = Precomp_SH.ϕ
    (; r, dr_dϕ, dr_dθ, dr_dθ_div_sinϕ, d²r_dϕ², d²r_dϕdθ, d²r_dθ², EN_lim, FM_lim) = Dynamic_SH

    E = dr_dϕ.^2 .+ r.^2
    F = dr_dϕ .* dr_dθ
    G = dr_dθ.^2 .+ r.^2 .* sin.(ϕ).^2
    denom = dS ./ r

    M = similar(E)
    L = (r .* d²r_dϕ² .- 2 .* dr_dϕ.^2 .- r.^2) ./ denom
    N = (r .* d²r_dθ² .+ r.*dr_dϕ.*sin.(ϕ).*cos.(ϕ) .- 2 .* dr_dθ.^2 .- r.^2 .* sin.(ϕ).^2) ./ denom

    EG_min_F²_div_sinϕ² = r.^2 .* denom.^2

    EN_div_sinϕ² = similar(EG_min_F²_div_sinϕ²); GL_div_sinϕ² = similar(EN_div_sinϕ²); FM_div_sinϕ² = similar(EN_div_sinϕ²)
    

    # All divided by sin²ϕ (both denominator and numerator):
    # EN = zeros(Float64, length(ϕ)); GL = similar(EN); FM = similar(EN)
    # EN[2:end] .= (dr_dϕ[2:end] .^ 2 .+ r[2:end] .^ 2) .* (r[2:end] .* d²r_dθ²[2:end] ./ (sin.(ϕ[2:end]) .^ 2) .+
    #                                                        r[2:end] .* dr_dϕ[2:end] .* cos.(ϕ[2:end]) ./ sin.(ϕ[2:end]) .-
    #                                                        2. .* (dr_dθ[2:end] ./ sin.(ϕ[2:end])) .^ 2 .-
    #                                                        r[2:end] .^ 2)
    # GL[2:end] .= ((dr_dθ[2:end] ./ sin.(ϕ[2:end])) .^ 2 .+ r[2:end] .^ 2) .* (r[2:end] .* d²r_dϕ²[2:end] .-
    #                                                                              2. .* dr_dϕ[2:end] .^ 2 .-
    #                                                                              r[2:end] .^ 2)
    # FM[2:end] .= dr_dϕ[2:end] .* dr_dθ[2:end] ./ sin.(ϕ[2:end]) .* (-2. .* dr_dϕ[2:end] .* dr_dθ[2:end] ./ sin.(ϕ[2:end]) .+
    #                                                                   r[2:end] .* d²r_dϕdθ[2:end] ./ sin.(ϕ[2:end]) .-
    #                                                                   r[2:end] .* dr_dθ[2:end] .* cos.(ϕ[2:end]) ./ (sin.(ϕ[2:end]) .^ 2))
    if abs(ϕ[1]) < 1e-10  # north pole: use limits
        M[2:end] .= (r[2:end] .* d²r_dϕdθ[2:end] .- 2 .* dr_dϕ[2:end] .* dr_dθ[2:end] 
                - r[2:end] .* cos.(ϕ[2:end]) .* dr_dθ[2:end] ./ sin.(ϕ[2:end])) ./ denom[2:end]
        M[1] = (r[1] * d²r_dϕdθ[1] - 2. * dr_dϕ[1] * dr_dθ[1] - r[1] * cos(ϕ[1]) * dr_dθ_div_sinϕ) / denom[1]
        EN_div_sinϕ²[2:end] .= E[2:end] .* N[2:end] ./ sin.(ϕ[2:end]).^2
        EN_div_sinϕ²[1] = E[1] * (r[1] * EN_lim - 2 * dr_dθ_div_sinϕ^2 - r[1]^2) / denom[1]
        
        GL_div_sinϕ²[2:end] .= G[2:end] .* L[2:end] ./ sin.(ϕ[2:end]).^2
        GL_div_sinϕ²[1] = (dr_dθ_div_sinϕ^2 + r[1]^2) .* L[1]
        
        FM_div_sinϕ²[2:end] .= F[2:end] .* M[2:end] ./ sin.(ϕ[2:end]).^2
        FM_div_sinϕ²[1] = dr_dϕ[1] * dr_dθ_div_sinϕ * (r[1] * FM_lim - 2 * dr_dϕ[1] * dr_dθ_div_sinϕ) / denom[1]

        
        # EN[1] = (dr_dϕ[1] ^ 2 + r[1] ^ 2) * (r[1] * EN_lim - 2. * dr_dθ_div_sinϕ ^ 2 - r[1] ^ 2)
        # GL[1] = (dr_dθ_div_sinϕ ^ 2 + r[1] ^ 2) * (r[1] * d²r_dϕ²[1] - 2. * dr_dϕ[1] ^ 2 - r[1] ^ 2)
        # FM[1] = dr_dϕ[1] * dr_dθ_div_sinϕ * (-2. * dr_dϕ[1] * dr_dθ_div_sinϕ + r[1] * FM_lim)
    else    # scalar call away from north pole (single element vector)
        M[1] = (r[1] * d²r_dϕdθ[1] - 2. * dr_dϕ[1] * dr_dθ[1] - r[1] * cos(ϕ[1]) * dr_dθ[1] / sin(ϕ[1])) / denom[1]
        EN_div_sinϕ²[1] = E[1] * N[1] / sin(ϕ[1])^2
        GL_div_sinϕ²[1] = G[1] * L[1] / sin(ϕ[1])^2
        FM_div_sinϕ²[1] = F[1] * M[1] / sin(ϕ[1])^2
        
        # sinϕ1 = sin(ϕ[1])
        # drdθ_sinϕ1 = dr_dθ[1] / sinϕ1
        # EN[1] = (dr_dϕ[1] ^ 2 + r[1] ^ 2) * (r[1] * d²r_dθ²[1] / sinϕ1 ^ 2 +
        #                                        r[1] * dr_dϕ[1] * cos(ϕ[1]) / sinϕ1 -
        #                                        2. * drdθ_sinϕ1 ^ 2 - r[1] ^ 2)
        # GL[1] = (drdθ_sinϕ1 ^ 2 + r[1] ^ 2) * (r[1] * d²r_dϕ²[1] - 2. * dr_dϕ[1] ^ 2 - r[1] ^ 2)
        # FM[1] = dr_dϕ[1] * drdθ_sinϕ1 * (-2. * dr_dϕ[1] * drdθ_sinϕ1 +
        #                                    r[1] * d²r_dϕdθ[1] / sinϕ1 -
        #                                    r[1] * dr_dθ[1] * cos(ϕ[1]) / sinϕ1 ^ 2)
    end

    κ = (EN_div_sinϕ² .+ GL_div_sinϕ² - 2 .* FM_div_sinϕ²) ./ EG_min_F²_div_sinϕ²

    # κ = ((EN .+ GL .- 2. .* FM) ./ dS .* r) ./ (dS .^ 2)

    return κ
end

"""
Compute local surface tension force at spherical design cubature points, in Cartesian coordinates.
"""
function surface_tension(Precomp_SH, Dynamic_SH, σ)
    _, _, _, n_x, n_y, n_z = unit_normal(Precomp_SH, Dynamic_SH)

    dS = surface_element.(Precomp_SH, Dynamic_SH)

    κ = surface_curvature.(Precomp_SH, Dynamic_SH, dS)

    # Note: kappa is negative and unit normal points out of bubble, so surface tension points into the bubble
    pre_surf_tension = σ * κ .* (dS * 4. * π / length(Precomp_SH.ϕ))  

    surf_tension = zeros(Float64, (length(Precomp_SH.ϕ), 3))

    # Surface tension force in each Cartesian coordinate [N]
    surf_tension[:, 1] .= pre_surf_tension .* n_x
    surf_tension[:, 2] .= pre_surf_tension .* n_y
    surf_tension[:, 3] .= pre_surf_tension .* n_z

    return surf_tension
end

"""
Compute the total pressure drop over a bubble, parametrized using spherical harmonics.
"""
function compute_p_drop(Precomp_SH, Dynamic_SH, σ)
    # Divided by sinϕ:
    dS = surface_element(Precomp_SH, Dynamic_SH)

    S = surface_area(dS, Precomp_SH.ϕ)

    κ = surface_curvature(Precomp_SH, Dynamic_SH, dS)

    # println("Mean |FM|: $(sum(abs.(FM))/length(FM)), max: $(maximum(abs.(FM)))")

    p_drop = 4. * π / length(Precomp_SH.ϕ) * σ / S * sum(κ .* dS)

    return p_drop, κ, S
end

"""
Compute the total surface area, the local surface curvature, and the total pressure drop over the bubble surface 
for a perfectly ellipsoidal bubble, with half-axes 'axis1', 'axis2', 'axis3'.
"""
function ellipsoid(axis1, axis2, axis3, npoints, σ)
    # max_ax = max(axis1, max(axis2, axis3))
    # a1, a2, a3 = axis1 / max_ax, axis2 / max_ax, axis3 / max_ax     # normalize
    a1, a2, a3 = axis1, axis2, axis3
    _, ϕ, θ = get_points_spc(npoints)
    r = sqrt.(1. ./ (sin.(ϕ) .^ 2 .* cos.(θ) .^ 2 / (a1 ^ 2) + sin.(ϕ) .^ 2 .* sin.(θ) .^ 2 / (a2 ^ 2) + cos.(ϕ) .^ 2 / (a3 ^ 2)))

    E = a1 ^ 2 * cos.(ϕ) .^ 2 .* cos.(θ) .^ 2 + a2 ^ 2 * cos.(ϕ) .^ 2 .* sin.(θ) .^ 2 + a3 ^ 2 * sin.(ϕ) .^ 2
    F = (a2 ^ 2 - a1 ^ 2) * sin.(ϕ) .* cos.(ϕ) .* sin.(θ) .* cos.(θ) 
    G = sin.(ϕ) .^ 2 .* (a1 ^ 2 * sin.(θ) .^ 2 + a2 ^ 2 * cos.(θ) .^ 2)
    n_length = sqrt.(a1 ^ 2 * a2 ^ 2 * cos.(ϕ) .^ 2 + a3 ^ 2 * sin.(ϕ) .^ 2 .* (a1 ^ 2 * sin.(θ) .^ 2 + a2 ^ 2 * cos.(θ) .^ 2))
    L = - a1 * a2 * a3 ./ n_length
    N = - a1 * a2 * a3 * sin.(ϕ) .^ 2 ./ n_length    # M ≡ 0
    
    G_div_sin²ϕ = a1 ^ 2 * sin.(θ) .^ 2 + a2 ^ 2 * cos.(θ) .^ 2
    N_div_sin²ϕ = - a1 * a2 * a3 ./ n_length
    denom = a1 ^ 2 * a2 ^ 2 * cos.(ϕ) .^ 2 + a3 ^ 2 * sin.(ϕ) .^ 2 .* (a1 ^ 2 * sin.(θ) .^ 2 + a2 ^ 2 * cos.(θ) .^ 2)   # (EG - F^2) / sin²ϕ
    num = E .* N_div_sin²ϕ + G_div_sin²ϕ .* L
    κ = num ./ denom    # twice the local mean curvature

    # # Divided by sin²ϕ:
    # EN = - a1 * a2 * a3 ./ n_length .* ((a1^2 * cos.(θ) .^ 2 + a2^2 * sin.(θ) .^ 2) .* cos.(ϕ) .^ 2 + a3^2 * sin.(ϕ) .^ 2)
    # GL = - a1*a2*a3 ./ n_length .* (a2^2 * cos.(θ) .^ 2 + a1^2 * sin.(θ) .^ 2)     

    # κ2 = (EN + GL) ./ denom

    # κ_ref = a1 * a2 * a3 * ((3. * (a1 ^ 2 + a2 ^ 2) + 2. * a3 ^ 2 .+ (a1 ^ 2 + a2 ^ 2 - 2. * a3 ^ 2) * cos.(2. * ϕ) -
    #                         2. * (a1 ^ 2 - a2 ^ 2) .* cos.(2. * θ) .* sin.(ϕ) .^ 2) ./ 
    #                         (4. * (a1 ^ 2 * a2 ^ 2 * cos.(ϕ) .^ 2 + 
    #                         a3 ^ 2 * sin.(ϕ) .^ 2 .* (a2 ^ 2 * cos.(θ) .^ 2 + a1 ^ 2 * sin.(θ) .^ 2)) .^ (1.5)))

    S = 4. * π / npoints * sum(n_length)

    p_drop = 4. * π * σ / npoints / S * sum(κ .* n_length)

    return r, ϕ, θ, κ, p_drop, S
end

############## to move (to Prescribed_u.jl)
function get_ux_spc(c, Y, umax, ux, ϕ, θ)
    r = Y * c 
    u_x = umax[1] * ux(r .* sin.(ϕ) .* sin.(θ))
    return u_x
end

function get_ux_cart(umax, ux, y)
    u_x = umax[1] * ux(y)
    return u_x
end
###########################################

"""
Perform one explicit time (sub-)step of a translating and deforming bubble described as a linear combination of spherical harmonics.
- u: shape (npoints, 3); velocity field interpolated to spherical design cubature points (ϕ, θ) relative to centroid.
Returns update of coefficients 'dc_dt' and of centroid '[u_centr_x, u_centr_y, u_centr_z]'.
"""
function time_step(Bub, Precomp_SH, Dynamic_SH, u, V)
    r, dr_dϕ, dr_dθ, dr_dθ_div_sinϕ = Dynamic_SH.r, Dynamic_SH.dr_dϕ, Dynamic_SH.dr_dθ, Dynamic_SH.dr_dθ_div_sinϕ
    ϕ, θ, ℓs, one, mone, Y = Precomp_SH.ϕ, Precomp_SH.θ, Precomp_SH.ℓs, Precomp_SH.one, Precomp_SH.mone, Precomp_SH.Y
    c = Bub.c
    npoints = length(ϕ)

    # Bubble centroid velocity:
    u_centr = r.^2 .* u[:, 1] .* (r .* sin.(ϕ) .* cos.(θ) - dr_dϕ .* cos.(ϕ) .* cos.(θ)) + 
                    r.^2 .* u[:, 2] .* (r .* sin.(ϕ) .* sin.(θ) - dr_dϕ .* cos.(ϕ) .* sin.(θ)) + 
                    r.^2 .* u[:, 3] .* (r .* cos.(ϕ) + dr_dϕ .* sin.(ϕ))
    u_centr[2:end] .= u_centr[2:end] + r[2:end].^2 .* (u[2:end, 1] .* dr_dθ[2:end] .* sin.(θ[2:end]) ./ sin.(ϕ[2:end]) - 
                                                u[2:end, 2] .* dr_dθ[2:end] .* cos.(θ[2:end]) ./ sin.(ϕ[2:end]))
    # dr_dθ_div_sinϕ = sqrt(2.) * sum(K_lone.(ℓs[one]) .* ℓs[one] .* (ℓs[one] .+ 1) / 2. .* c[one] * sin(θ[1]) - 
    #                     K_lone.(ℓs[mone]) .* ℓs[mone] .* (ℓs[mone] .+ 1) / 2. .* c[mone] * cos(θ[1]))
    # println(dr_dθ_div_sinϕ)
    u_centr[1] = u_centr[1] + r[1]^2 * (u[1, 1] * dr_dθ_div_sinϕ * sin(θ[1]) - 
                                        u[1, 2] * dr_dθ_div_sinϕ * cos(θ[1]))
    u_centr_x = 4. * π / npoints / V * sum(u_centr .* sin.(ϕ) .* cos.(θ))
    u_centr_y = 4. * π / npoints / V * sum(u_centr .* sin.(ϕ) .* sin.(θ))
    u_centr_z = 4. * π / npoints / V * sum(u_centr .* cos.(ϕ))

    # Spherical harmonics coefficients 'velocity'
    dϕ_dt = (cos.(ϕ) .* cos.(θ) .* (u[:, 1] .- u_centr_x) + 
            cos.(ϕ) .* sin.(θ) .* (u[:, 2] .- u_centr_y) - 
            sin.(ϕ) .* (u[:, 3] .- u_centr_z)) ./ r 
    θ_term = similar(dϕ_dt)
    θ_term[2:end] .= (- (u[2:end, 1] .- u_centr_x) .* sin.(θ[2:end]) + 
                        (u[2:end, 2] .- u_centr_y) .* cos.(θ[2:end])) ./ (r[2:end] .* sin.(ϕ[2:end])) .* dr_dθ[2:end]
    θ_term[1] = (- (u[1, 1] - u_centr_x) * sin(θ[1]) + 
                    (u[1, 2] - u_centr_y) * cos(θ[1])) / r[1] * dr_dθ_div_sinϕ
    r_term = (sin.(ϕ) .* cos.(θ) .* (u[:, 1] .- u_centr_x) + 
            sin.(ϕ) .* sin.(θ) .* (u[:, 2] .- u_centr_y) + 
            cos.(ϕ) .* (u[:, 3] .- u_centr_z))
    dc_dt = 4. * π / npoints * (r_term' * Y - (dϕ_dt .* dr_dϕ)' * Y - θ_term' * Y)[1, :]
    return dc_dt, [u_centr_x, u_centr_y, u_centr_z]
end

# function time_step_z(c, Y, dY_dϕ, ϕ, u_z, npoints, V)
#     r = Y * c
#     dr_dϕ = dY_dϕ * c

#     # Bubble centroid velocity:
#     u_centr = 4. * π / npoints / V * sum(r .* (r .* u_z .* cos.(ϕ) + dr_dϕ .* u_z .* sin.(ϕ)) .* r .* cos.(ϕ))

#     # Spherical harmonics coefficients 'velocity'
#     dϕ_dt = (- sin.(ϕ) .* (u_z .- u_centr)) ./ r
#     dc_dt = 4. * π / npoints * ((cos.(ϕ) .* (u_z .- u_centr) - dϕ_dt .* dr_dϕ)' * Y)[1, :]
#     return dc_dt, u_centr
# end

# function time_step_x(c, Y, dY_dϕ, dY_dθ, ϕ, θ, u_x, npoints, V, ℓs, ms, one, mone)
#     r = Y * c 
#     dr_dϕ = dY_dϕ * c 
#     dr_dθ = dY_dθ * c 

#     # Bubble centroid velocity:
#     u_centr = 4. * π / npoints / V * sum(r .* u_x .* (r .* sin.(ϕ).^2 .* cos.(θ) - dr_dϕ .* cos.(ϕ) .* cos.(θ) .* sin.(ϕ) + sin.(θ) .* dr_dθ) .* r .* cos.(θ))

#     # Spherical harmonics coefficients 'velocity'
#     dϕ_dt = cos.(ϕ) .* cos.(θ) .* (u_x .- u_centr) ./ r 
#     θ_term = similar(dϕ_dt)
#     θ_term[2:end] .= - (u_x[2:end] .- u_centr) .* sin.(θ[2:end]) ./ (r[2:end] .* sin.(ϕ[2:end])) .* dr_dθ[2:end]
#     θ_term[1] = - (u_x[1] - u_centr) * sin(θ[1]) / r[1] * sum(K_lm.(ℓs[one], ms[one]) .* ℓs[one] .* (ℓs[one] .+ 1) / 2. .* c[one] * sin(θ[1]) - 
#                                                                 K_lm.(ℓs[one], -ms[mone]) .* ℓs[mone] .* (ℓs[mone] .+ 1) / 2. .* c[mone] * cos(θ[1]))
#     dc_dt = 4. * π / npoints * ((sin.(ϕ) .* cos.(θ) .* (u_x .- u_centr))' * Y - (dϕ_dt .* dr_dϕ)' * Y - θ_term' * Y)[1, :]
#     return dc_dt, u_centr
# end

# function parabolic_z(c, centr, Y, dY_dϕ, ϕ, θ, u, umax, npoints, V, dt, nt, Y_test, ϕ_test, θ_test)
#     r0 = Y_test * c 
#     x0, y0, z0 = spc2cart(r0, ϕ_test, θ_test)
#     x0, y0, z0 = x0 .+ centr[1], y0 .+ centr[2], z0 .+ centr[3]

#     Vs = zeros(Float64, nt)

#     for i in 1:nt   # RK4 time stepping
#         k1, k1_centr = time_step_z(c, Y, dY_dϕ, ϕ, get_uz_spc(c, Y, umax, u, ϕ, θ), npoints, V)
#         k2, k2_centr = time_step_z(c + k1 * dt / 2., Y, dY_dϕ, ϕ, get_uz_spc(c + k1 * dt / 2., Y, umax, u, ϕ, θ), npoints, V)
#         k3, k3_centr = time_step_z(c + k2 * dt / 2., Y, dY_dϕ, ϕ, get_uz_spc(c + k2 * dt / 2., Y, umax, u, ϕ, θ), npoints, V)
#         k4, k4_centr = time_step_z(c + k3 * dt, Y, dY_dϕ, ϕ, get_uz_spc(c + k3 * dt, Y, umax, u, ϕ, θ), npoints, V)

#         c = c + (k1 + 2. * k2 + 2. * k3 + k4) * dt / 6.
#         centr[3] = centr[3] .+ (k1_centr + 2. * k2_centr + 2. * k3_centr + k4_centr) * dt / 6.

#         Vs[i] = volume(c, Y_test)
#     end
    
#     r = Y_test * c 
#     x, y, z = r .* sin.(ϕ_test) .* cos.(θ_test), r .* sin.(ϕ_test) .* sin.(θ_test), centr[3] .+ r .* cos.(ϕ_test)
#     z_ref = z0 .+ get_uz_cart(umax, u, x0) * dt * nt    # for computing errors later

#     return x0, y0, z0, x, y, z, Vs
# end

############## to move (to Prescribed_u.jl)
function linear_x(c, centr, Y, dY_dϕ, dY_dθ, ϕ, θ, u, umax, npoints, V, dt, nt, Y_test, ϕ_test, θ_test, ℓs, ms, one, mone)
    r0 = Y_test * c 
    x0, y0, z0 = spc2cart(r0, ϕ_test, θ_test)
    x0, y0, z0 = x0 .+ centr[1], y0 .+ centr[2], z0 .+ centr[3]

    Vs = zeros(Float64, nt)

    for _ in 1:nt   # RK4 time stepping
        k1, k1_centr = time_step_x(c, Y, dY_dϕ, dY_dθ, ϕ, θ, get_ux_spc(c, Y, umax, u, ϕ, θ), npoints, V, ℓs, ms, one, mone)
        k2, k2_centr = time_step_x(c + k1 * dt / 2., Y, dY_dϕ, dY_dθ, ϕ, θ, get_ux_spc(c + k1 * dt / 2., Y, umax, u, ϕ, θ), npoints, V, ℓs, ms, one, mone)
        k3, k3_centr = time_step_x(c + k2 * dt / 2., Y, dY_dϕ, dY_dθ, ϕ, θ, get_ux_spc(c + k2 * dt / 2., Y, umax, u, ϕ, θ), npoints, V, ℓs, ms, one, mone)
        k4, k4_centr = time_step_x(c + k3 * dt, Y, dY_dϕ, dY_dθ, ϕ, θ, get_ux_spc(c + k3 * dt, Y, umax, u, ϕ, θ), npoints, V, ℓs, ms, one, mone)

        c = c + (k1 + 2. * k2 + 2. * k3 + k4) * dt / 6.
        centr[1] = centr[1] .+ (k1_centr + 2. * k2_centr + 2. * k3_centr + k4_centr) * dt / 6.

        Vs[i] = volume(c, Y_test)
    end
    
    r = Y_test * c 
    x, y, z = centr[1] .+ r .* sin.(ϕ_test) .* cos.(θ_test), r .* sin.(ϕ_test) .* sin.(θ_test), r .* cos.(ϕ_test)
    x_ref = x0 .+ get_ux_cart(umax, u, y0) * dt * nt    # for computing errors later

    return x0, y0, z0, x, y, z, Vs
end
###########################################