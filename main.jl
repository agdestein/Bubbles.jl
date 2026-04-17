# Load dependencies
using Bubbles # Bubble source code
using WGLMakie # Interactive Makie backend

# Illustrate marker masking procedure
Bubbles.illustrate_masking() |> display

# Problem definition
setup = Bubbles.lidsetup()
psolver = Bubbles.NS.default_psolver(setup)
u = Bubbles.NS.velocityfield(setup, (dim, x, y) -> zero(x));
x, xcenter = Bubbles.bubble()

# Plot nonuniform grid
Bubbles.plot_bubble_on_grid(setup, x) |> display

# Solve
(; u, x, xcenter) = Bubbles.solveandplot(u, x, xcenter, setup, psolver)

Bubbles.plotstate(Observable((; u, x, xcenter)), setup)

Bubbles.plot_insidemarkers(u, x, xcenter, setup) |> display
Bubbles.plot_fractions(u, x, xcenter, setup) |> display

# Compute integral of surface tension (it should be zero)
false && let
    npoint = length(x)
    s = similar(x)
    Bubbles.surfacetension!(s, x)
    sum(1:npoint) do i
        t = s[i]
        # p = x[i]
        # q = x[mod1(i + 1, npoint)]
        # dx = abs(q[1] - p[1])
        # dy = abs(q[2] - p[2])
        dx = 1
        dy = 1
        return Bubbles.MyPoint(t[1] * dx, t[2] * dy)
    end
end
