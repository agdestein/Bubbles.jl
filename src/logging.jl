using WriteVTK

function write_pvd(path, entries)
	open(path, "w") do io
		println(io, "<?xml version=\"1.0\"?>")
		println(io, "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">")
		println(io, "  <Collection>")
		for (t, relfile) in entries
			println(io, "    <DataSet timestep=\"$(t)\" group=\"\" part=\"0\" file=\"$(replace(relfile, '\\' => '/'))\"/>")
		end
		println(io, "  </Collection>")
		println(io, "</VTKFile>")
	end
	return nothing
end

function write_vtk_snapshot!(outdir, itime, t, setup, U, fractions, ρ_eff, xb, yb, zb, nvis)
	mkpath(outdir)

	# Cell-centered flow fields on pressure grid (interior only, no ghost cells).
	u = adapt(Array, U.u)
	fr = adapt(Array, fractions)
	rho = adapt(Array, ρ_eff)
	x = collect(setup.xp[1][2:end-1])
	y = collect(setup.xp[2][2:end-1])
	z = collect(setup.xp[3][2:end-1])

	ux = @views 0.5 .* (u[2:end-1, 2:end-1, 2:end-1, 1] .+ u[3:end,   2:end-1, 2:end-1, 1])
	uy = @views 0.5 .* (u[2:end-1, 2:end-1, 2:end-1, 2] .+ u[2:end-1, 3:end,   2:end-1, 2])
	uz = @views 0.5 .* (u[2:end-1, 2:end-1, 2:end-1, 3] .+ u[2:end-1, 2:end-1, 3:end,   3])
	umag = sqrt.(ux.^2 .+ uy.^2 .+ uz.^2)

	frac = @views fr[2:end-1, 2:end-1, 2:end-1]
	rho_x = @views rho[2:end-1, 2:end-1, 2:end-1, 1]
	rho_y = @views rho[2:end-1, 2:end-1, 2:end-1, 2]
	rho_z = @views rho[2:end-1, 2:end-1, 2:end-1, 3]
	rho_mean = (rho_x .+ rho_y .+ rho_z) ./ 3

	flow_name = "flow_centered_" * lpad(string(itime), 6, '0')
	vtk_flow = vtk_grid(joinpath(outdir, flow_name), x, y, z)
	vtk_flow["u_x"] = ux
	vtk_flow["u_y"] = uy
	vtk_flow["u_z"] = uz
	vtk_flow["u_mag"] = umag
	vtk_flow["phase_fraction"] = frac
	vtk_flow["rho_eff_x_centered"] = rho_x
	vtk_flow["rho_eff_y_centered"] = rho_y
	vtk_flow["rho_eff_z_centered"] = rho_z
	vtk_flow["rho_eff_mean_centered"] = rho_mean
	vtk_flow["time"] = fill(t, size(frac))
	flow_file = vtk_save(vtk_flow)

	# Staggered face fields on their native component grids.
	xu1 = collect(setup.xu[1][1][2:end-1]); yu1 = collect(setup.xu[1][2][2:end-1]); zu1 = collect(setup.xu[1][3][2:end-1])
	xu2 = collect(setup.xu[2][1][2:end-1]); yu2 = collect(setup.xu[2][2][2:end-1]); zu2 = collect(setup.xu[2][3][2:end-1])
	xu3 = collect(setup.xu[3][1][2:end-1]); yu3 = collect(setup.xu[3][2][2:end-1]); zu3 = collect(setup.xu[3][3][2:end-1])

	uface_x = @views u[2:end-1, 2:end-1, 2:end-1, 1]
	uface_y = @views u[2:end-1, 2:end-1, 2:end-1, 2]
	uface_z = @views u[2:end-1, 2:end-1, 2:end-1, 3]

	rho_face_x = @views rho[2:end-1, 2:end-1, 2:end-1, 1]
	rho_face_y = @views rho[2:end-1, 2:end-1, 2:end-1, 2]
	rho_face_z = @views rho[2:end-1, 2:end-1, 2:end-1, 3]

	sfx_name = "staggered_x_" * lpad(string(itime), 6, '0')
	sfy_name = "staggered_y_" * lpad(string(itime), 6, '0')
	sfz_name = "staggered_z_" * lpad(string(itime), 6, '0')

	vtk_sfx = vtk_grid(joinpath(outdir, sfx_name), xu1, yu1, zu1)
	vtk_sfx["u_face_x"] = uface_x
	vtk_sfx["rho_eff_face_x"] = rho_face_x
	vtk_sfx["time"] = fill(t, size(uface_x))
	sfx_file = vtk_save(vtk_sfx)

	vtk_sfy = vtk_grid(joinpath(outdir, sfy_name), xu2, yu2, zu2)
	vtk_sfy["u_face_y"] = uface_y
	vtk_sfy["rho_eff_face_y"] = rho_face_y
	vtk_sfy["time"] = fill(t, size(uface_y))
	sfy_file = vtk_save(vtk_sfy)

	vtk_sfz = vtk_grid(joinpath(outdir, sfz_name), xu3, yu3, zu3)
	vtk_sfz["u_face_z"] = uface_z
	vtk_sfz["rho_eff_face_z"] = rho_face_z
	vtk_sfz["time"] = fill(t, size(uface_z))
	sfz_file = vtk_save(vtk_sfz)

	# Bubble surface as a 3D structured grid with a degenerate Nk=1 third
	# dimension. WriteVTK requires (3, Ni, Nj, Nk) for surfaces in 3D space.
	surf_pts = zeros(eltype(xb), 3, nvis, nvis, 1)
	surf_pts[1, :, :, 1] = reshape(xb, nvis, nvis)
	surf_pts[2, :, :, 1] = reshape(yb, nvis, nvis)
	surf_pts[3, :, :, 1] = reshape(zb, nvis, nvis)
	surf_name = "bubble_" * lpad(string(itime), 6, '0')
	vtk_surf = vtk_grid(joinpath(outdir, surf_name), surf_pts)
	vtk_surf["time"] = fill(t, nvis, nvis, 1)
	surf_file = vtk_save(vtk_surf)

	return first(flow_file), first(sfx_file), first(sfy_file), first(sfz_file), first(surf_file)
end

function init_bubble_history_log(path, ncoef)
	mkpath(dirname(path))
	open(path, "w") do io
		coef_cols = join(("c" * string(i - 1) for i in 1:ncoef), "\t")
		println(io, "time\titime\t" * coef_cols * "\tcom_x\tcom_y\tcom_z\tvcom_x\tvcom_y\tvcom_z")
	end
	return nothing
end

function append_bubble_history_log(path, t, itime, coefs, centr, vcom)
	open(path, "a") do io
		coef_vals = join(string.(coefs), "\t")
		println(io,
			string(t) * "\t" * string(itime) * "\t" * coef_vals * "\t" *
			string(centr[1]) * "\t" * string(centr[2]) * "\t" * string(centr[3]) * "\t" *
			string(vcom[1]) * "\t" * string(vcom[2]) * "\t" * string(vcom[3]),
		)
	end
	return nothing
end

"""
Read `bubble_history.txt` produced by `append_bubble_history_log`.

Returns a named tuple with fields:
- `time::Vector{Float64}`
- `itime::Vector{Int}`
- `coef_names::Vector{String}`
- `coefs::Matrix{Float64}` where each row is one time sample
- `com::Matrix{Float64}` with columns `(x, y, z)`
- `vcom::Matrix{Float64}` with columns `(x, y, z)`
"""
function read_bubble_history_log(path)
	lines = readlines(path)
	isempty(lines) && error("bubble history file is empty: $path")

	header = split(chomp(lines[1]), '\t')
	expected_tail = ("com_x", "com_y", "com_z", "vcom_x", "vcom_y", "vcom_z")

	length(header) >= 9 || error("invalid bubble history header in $path")
	header[1] == "time" || error("invalid bubble history header: first column must be 'time'")
	header[2] == "itime" || error("invalid bubble history header: second column must be 'itime'")
	Tuple(header[end-5:end]) == expected_tail ||
		error("invalid bubble history header: expected trailing columns $(collect(expected_tail))")

	coef_names = header[3:end-6]
	ncoef = length(coef_names)

	data_lines = String[]
	for line in lines[2:end]
		s = strip(line)
		isempty(s) || push!(data_lines, s)
	end

	nrow = length(data_lines)
	time = Vector{Float64}(undef, nrow)
	itime = Vector{Int}(undef, nrow)
	coefs = Matrix{Float64}(undef, nrow, ncoef)
	com = Matrix{Float64}(undef, nrow, 3)
	vcom = Matrix{Float64}(undef, nrow, 3)

	for (i, line) in pairs(data_lines)
		fields = split(line, '\t')
		length(fields) == length(header) ||
			error("invalid bubble history row $i in $path: expected $(length(header)) columns, got $(length(fields))")

		time[i] = parse(Float64, fields[1])
		itime[i] = parse(Int, fields[2])

		for j in 1:ncoef
			coefs[i, j] = parse(Float64, fields[2 + j])
		end

		com[i, 1] = parse(Float64, fields[end - 5])
		com[i, 2] = parse(Float64, fields[end - 4])
		com[i, 3] = parse(Float64, fields[end - 3])

		vcom[i, 1] = parse(Float64, fields[end - 2])
		vcom[i, 2] = parse(Float64, fields[end - 1])
		vcom[i, 3] = parse(Float64, fields[end])
	end

	return (; time, itime, coef_names, coefs, com, vcom)
end
