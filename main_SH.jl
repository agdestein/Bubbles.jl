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

function restart_from_last!(u, Bub;
    vtk_dir = joinpath("output", "vtk_L_0_1"),
    bubble_log_path = joinpath("output", "bubble_L_0_1", "bubble_history.txt"),
)
    hist = read_bubble_history_log(bubble_log_path)
    nrow = length(hist.time)
    nrow > 0 || error("bubble history contains no data rows: $(bubble_log_path)")

    hist_by_itime = Dict{Int,Tuple{Float64,Int}}()
    for i in eachindex(hist.itime)
        hist_by_itime[hist.itime[i]] = (hist.time[i], i) # keep latest row for repeated itime
    end

    vels = read_all_staggered_vtk_velocities(vtk_dir)
    vtk_time_by_itime = Dict{Int,Union{Nothing,Float64}}()
    for v in vels
        vtk_time_by_itime[parse(Int, v.suffix)] = v.time
    end

    common_itimes = sort!(collect(intersect(keys(hist_by_itime), keys(vtk_time_by_itime))))
    isempty(common_itimes) && error("no common saved itime in bubble history and staggered VTK files")

    tol = 1e-8
    chosen_itime = nothing
    chosen_idx = nothing
    chosen_time = nothing
    for it in reverse(common_itimes)
        htime, hidx = hist_by_itime[it]
        vtime = vtk_time_by_itime[it]
        if vtime === nothing || isapprox(htime, vtime; atol = tol, rtol = tol)
            chosen_itime = it
            chosen_idx = hidx
            chosen_time = htime
            break
        end
    end

    if chosen_itime === nothing
        it = common_itimes[end]
        htime, hidx = hist_by_itime[it]
        @warn "No time-aligned common itime found; restarting from latest common itime only" itime = it hist_time = htime vtk_time = vtk_time_by_itime[it]
        chosen_itime = it
        chosen_idx = hidx
        chosen_time = htime
    end

    Bub.c .= vec(hist.coefs[chosen_idx, :])
    Bub.centr .= vec(hist.com[chosen_idx, :])

    suffix = lpad(string(chosen_itime), 6, '0')
    face = read_staggered_vtk_velocity_components(vtk_dir, suffix)

    fill!(u, 0.0)
    u[2:end-1, 2:end-1, 2:end-1, 1] .= face.ux
    u[2:end-1, 2:end-1, 2:end-1, 2] .= face.uy
    u[2:end-1, 2:end-1, 2:end-1, 3] .= face.uz

    @info "Restart state selected" itime = chosen_itime time = chosen_time
    return (; t = chosen_time, itime = chosen_itime)
end

# solveandplot(u, Bub, setup, Precomp_SH, L);

restart = true
if restart
    state = restart_from_last!(u, Bub)
    solveandplot_mat(u, Bub, setup, Precomp_SH, L; restart = true, t0 = state.t, itime0 = state.itime)
else
    solveandplot_mat(u, Bub, setup, Precomp_SH, L)
end
