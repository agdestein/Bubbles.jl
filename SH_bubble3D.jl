include("src/SH.jl")    # import source code

# Spherical design cubature points:
npoints = 1059
r, ϕ, θ = get_points_spc(npoints)
r_test, ϕ_test, θ_test = get_points_spc(16382)  # for testing only

# Prescribed velocity field:
uzmax = [0., 0., 1.]        # max velocity in z direction 
uxmax = [1. / 3., 0., 0.]   # max velocity in x direction
C = 2.
uz = x -> 1. .- (x/C).^2    # parabolic velocity profile in the z direction 
ux = y -> C .+ y            # linear velocity profile in the x direction
V = 4. / 3. * π             # bubble volume, assumed constant

# Initial spherical harmonics coefficients
ℓₘ = 10; nbf = (ℓₘ + 1)^2  # number of basis functions (spherical harmonics)
println("# cubature points: $(length(ϕ)), #basis functions: $(nbf)")
c0 = zeros(Float64, nbf)
c0[1] = 1. * sqrt(4. * π)   # we start with a perfect sphere: only the first spherical harmonic is active; normalized to radius 1.
centr0 = zeros(Float64, 3)  # bubble centroid

dt = 1e-2; nt = 100 # time step, nr of time steps

Y, dY_dϕ, dY_dθ, ℓs, ms, one, mone = get_SH_der(ℓₘ, ϕ, θ)   # spherical harmonics (and partial derivatives) at cubature points, their identifiers (ℓ, m) and some specific indices (where m==1 and m==-1)
Y_test = get_SH(ℓₘ, ϕ_test, θ_test)                         # spherical harmonics at test points

### Comment/uncomment x vs y test case #############################################################################################################
# Parabolic velocity profile in z direction:
x0, y0, z0, x, y, z = parabolic_z(c0, centr0, Y, dY_dϕ, ϕ, θ, uz, uzmax, npoints, V, dt, nt, Y_test, ϕ_test, θ_test)

# Linear velocity profile in x direction:
# x0, y0, z0, x, y, z = linear_x(c0, centr0, Y, dY_dϕ, dY_dθ, ϕ, θ, ux, uxmax, npoints, V, dt, nt, Y_test, ϕ_test, θ_test, ℓs, ms, one, mone)
####################################################################################################################################################

scatter(x, y, z)