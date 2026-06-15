function p_jump_mat(f, sdf, Bub, Precomp_SH, setup)
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
        return nothing
    end

    # println("ρ_eff min, max: $(minimum(ρ_eff)), $(maximum(ρ_eff))")
    return jumps, ρ_eff
end

####### Matrix-based Poisson solver and velocity projection ################################################################
function laplacian_mat_variable_ρ(setup, ρ_eff)
    P = NS.pad_scalarfield_mat(setup)
    Bp = NS.bc_p_mat(setup)
    Bu = NS.bc_u_mat(setup)
    G = NS.pressuregradient_mat(setup)
    M = NS.divergence_mat(setup)
    Ω = NS.volume_mat(setup)
    # Rinv = inv_ρ_eff_mat(setup, ρ_eff)  # accounts for variable density
    Rinv = Diagonal(vec(adapt(Array, 1 ./ ρ_eff)))  # accounts for variable density
    return P' * Ω * M * Rinv * Bu * G * Bp * P
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
    psolver = psolver_cg_matrix_variable_ρ(setup, ρ_eff; abstol = 1e-12, reltol = 1e-12, maxiter = 1000)
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
function rk3step_SH_mat!(F, U0, U, t, dt, fractions, sdf, p, setup, Bub, Precomp_SH)
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

        jumps, ρ_eff = p_jump_mat(F.u, sdf, Bub, Precomp_SH, setup)
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