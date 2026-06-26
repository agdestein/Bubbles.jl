using WriteVTK
using ReadVTK

function _list_vtk_series_files(vtk_dir; prefix, ext = ".vtr")
	files = filter(readdir(vtk_dir; join = true)) do f
		base = basename(f)
		startswith(base, prefix) && endswith(base, ext)
	end

	isempty(files) && error("no files matching $(prefix)*$(ext) found in $(vtk_dir)")
	sort!(files)
	return files
end

function _vtk_point_data(vtk_file)
	vtk = ReadVTK.VTKFile(vtk_file)
	return ReadVTK.get_point_data(vtk)
end

function _read_vtk_component(vtk_file, component_name)
	point_data = _vtk_point_data(vtk_file)
	component_name in point_data.names ||
		error("missing component $(component_name) in $(vtk_file)")
	return ReadVTK.get_data_reshaped(point_data[component_name])
end

function _read_vtk_time(vtk_file)
	point_data = _vtk_point_data(vtk_file)
	if "time" in point_data.names
		time_data = ReadVTK.get_data(point_data["time"])
		return isempty(time_data) ? nothing : time_data[1]
	end
	return nothing
end

function _series_suffix_map(vtk_dir; prefix, ext = ".vtr")
	files = _list_vtk_series_files(vtk_dir; prefix, ext)
	suffix_map = Dict{String,String}()
	for file in files
		base = basename(file)
		suffix = base[length(prefix) + 1:end-length(ext)]
		suffix_map[suffix] = file
	end
	return suffix_map
end

"""
Read one VTK file and extract velocity component arrays.

By default this expects the `flow_centered_*.vtr` snapshots written by
`write_vtk_snapshot!`, with point data names `u_x`, `u_y`, and `u_z`.
"""
function read_vtk_velocity_components(vtk_file; component_names = ("u_x", "u_y", "u_z"))
	vtk = ReadVTK.VTKFile(vtk_file)
	point_data = ReadVTK.get_point_data(vtk)

	all(name -> name in point_data.names, component_names) ||
		error("missing one or more velocity components $(collect(component_names)) in $(vtk_file)")

	ux = ReadVTK.get_data_reshaped(point_data[component_names[1]])
	uy = ReadVTK.get_data_reshaped(point_data[component_names[2]])
	uz = ReadVTK.get_data_reshaped(point_data[component_names[3]])

	return (; ux, uy, uz)
end

"""
Read the final flow VTK snapshot in `vtk_dir` and compute max absolute velocity
in each Cartesian component.

Returns a named tuple with
- `vtk_file`: selected file path
- `maxabs`: named tuple `(ux, uy, uz)`
- `velocities`: named tuple `(ux, uy, uz)` with full arrays
"""
function read_final_vtk_velocity_maxabs(
	vtk_dir;
	prefix = "flow_centered_",
	ext = ".vtr",
	component_names = ("u_x", "u_y", "u_z"),
)
	files = _list_vtk_series_files(vtk_dir; prefix, ext)
	sort!(files)
	vtk_file = files[end]
	vel = read_vtk_velocity_components(vtk_file; component_names)

	maxabs = (
		ux = maximum(abs, vel.ux),
		uy = maximum(abs, vel.uy),
		uz = maximum(abs, vel.uz),
	)

	return (; vtk_file, maxabs, velocities = vel)
end

"""
Read one set of face-centered staggered velocity files for a given timestep suffix.

Returns a named tuple with file paths, optional time stamp, and velocity arrays
`ux`, `uy`, `uz` read from `staggered_x/y/z_*.vtr`.
"""
function read_staggered_vtk_velocity_components(
	vtk_dir,
	suffix;
	prefixes = ("staggered_x_", "staggered_y_", "staggered_z_"),
	component_names = ("u_face_x", "u_face_y", "u_face_z"),
	ext = ".vtr",
)
	files = (
		ux = joinpath(vtk_dir, prefixes[1] * suffix * ext),
		uy = joinpath(vtk_dir, prefixes[2] * suffix * ext),
		uz = joinpath(vtk_dir, prefixes[3] * suffix * ext),
	)

	for file in values(files)
		isfile(file) || error("missing staggered VTK file: $(file)")
	end

	velocities = (
		ux = _read_vtk_component(files.ux, component_names[1]),
		uy = _read_vtk_component(files.uy, component_names[2]),
		uz = _read_vtk_component(files.uz, component_names[3]),
	)
	time = _read_vtk_time(files.ux)

	return (; suffix, time, files, velocities...)
end

"""
Read all face-centered staggered velocity snapshots in `vtk_dir`.

Returns a vector of named tuples, one per timestep, with fields:
- `suffix`: timestep identifier from the file name
- `time`: stored time value if present, otherwise `nothing`
- `files`: named tuple of x/y/z VTK file paths
- `ux`, `uy`, `uz`: face-centered velocity arrays
"""
function read_all_staggered_vtk_velocities(
	vtk_dir;
	prefixes = ("staggered_x_", "staggered_y_", "staggered_z_"),
	component_names = ("u_face_x", "u_face_y", "u_face_z"),
	ext = ".vtr",
)
	x_map = _series_suffix_map(vtk_dir; prefix = prefixes[1], ext)
	y_map = _series_suffix_map(vtk_dir; prefix = prefixes[2], ext)
	z_map = _series_suffix_map(vtk_dir; prefix = prefixes[3], ext)

	suffixes = sort!(collect(intersect(intersect(keys(x_map), keys(y_map)), keys(z_map))))
	isempty(suffixes) && error("no common staggered x/y/z timesteps found in $(vtk_dir)")

	return [
		read_staggered_vtk_velocity_components(
			vtk_dir,
			suffix;
			prefixes,
			component_names,
			ext,
		)
		for suffix in suffixes
	]
end

# Reading
############################################################
# Writing

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
