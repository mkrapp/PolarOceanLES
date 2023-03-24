# # Tilted ice-ocean top boundary layer
#
# This experiment simulates a two-dimensional oceanic rectangle under an ice shelf
# in a domain that's tilted with respect to gravity. We simulate the perturbation
# away from a constant along-slope (y-direction) velocity constant density stratification.
# This perturbation develops into a turbulent ice-ocean boundary layer due to momentum
# loss at the top boundary modelled with a quadratic drag law.
#
# This exampe is insipred by the `tilted_bottom_boundary_layer.jl` simulation
# https://clima.github.io/OceananigansDocumentation/stable/generated/tilted_bottom_boundary_layer/

using Printf
using CUDA
using Oceananigans
using Oceananigans.BuoyancyModels: g_Earth
using Oceananigans.Units: seconds, minute, minutes, hour, days, hours, kilometer, kilometers, meters
using Oceananigans.Grids: min_Δz
using Oceananigans.TurbulenceClosures
include("utils.jl")

# model runtime parameters: number of hours, grid size, filename, etc
const stop_time     = 10days
const Δt            = 1minute
const max_Δt        = 10minutes
const Δt_output_fld = 20minutes

ARCH = has_cuda_gpu() ? GPU() : CPU()

parsed_args = parse_experiment("experiments_ice-ocean01")
experiment = parsed_args["experiment"]
@printf(" ▷▷▷ Experiment: '%s' ◁◁◁ \n", experiment)
path = parsed_args["path"]

# GRID DIMENSIONS
const Lx = 400 # m
const Lz = 100 # m
# GRID EXTENT
const Nx = 128
const Nz =  64

z₀ = 0meters # depth of seawater parcel

# Z-GRID PROPERTIES (refinement of Δz at ice-ocean interface)
const refinement = 1.8 # controls spacing near surface (higher means finer spaced)
const stretching = 10  # controls rate of stretching at bottom

z_faces = z_levels(Nz,Lz,z₀,refinement,stretching)


grid = RectilinearGrid(ARCH; topology = (Periodic, Flat, Bounded),
                       size = (Nx, Nz),
                       x = (0, Lx),
                       z = z_faces,
                       halo = (3, 3))

println(grid)

# MODEL
# Far-field values
const α = 1   # °
const β = 90  # ° 

const V∞ = 0.1      # far field velocity                     m s⁻¹
# Parameters
const f₀ = -1.37e-4 # Coriolis parameter (for 70°S)        rad s⁻¹
const N² = 1.0e-5   # background buoyancy gradient             s⁻¹
const cᴰ = 2.5e-3   # drag coefficient
const κ  = 1.4e-4   # tracer diffusivity                    m² s⁻¹
const ν  = 1.4e-4   # molecular viscosity                   m² s⁻¹
const ĝ = (sind(α)*sind(β), sind(α)*cosd(β), cosd(α)) # tilted ocean box wrt direction of gravity
println(ĝ)

buoyancy = Buoyancy(model = BuoyancyTracer(), gravity_unit_vector = ĝ)

coriolis = ConstantCartesianCoriolis(f = f₀, rotation_axis = ĝ)
println(coriolis)

# A constant density stratification in the tilted coordinate system
@inline constant_stratification(x, y, z, t, p) = p.N² * (x * p.ĝ[1] + (z - Lz) * p.ĝ[3])
# The constant stratification as a `BackgroundField`.
B_field = BackgroundField(constant_stratification, parameters=(; ĝ, N² = N²))

# Top drag. Background flow V∞ is part of drag calculation,
# which is the only effect the background flow enters the problem.
@inline drag_u(x, y, t, u, v, p) = p.cᴰ * √(u^2 + (v + p.V∞)^2) * u
@inline drag_v(x, y, t, u, v, p) = p.cᴰ * √(u^2 + (v + p.V∞)^2) * (v + p.V∞)

# as flux boundary condition
drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=(; cᴰ, V∞))
drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=(; cᴰ, V∞))

# as boundary conditions for u 
u_bcs = FieldBoundaryConditions(top = drag_bc_u, bottom=FluxBoundaryCondition(nothing))
v_bcs = FieldBoundaryConditions(top = drag_bc_v, bottom=FluxBoundaryCondition(nothing))

closure = ScalarDiffusivity(ν=ν, κ=κ)

model = NonhydrostaticModel(; grid, buoyancy, coriolis, closure,
                            timestepper = :RungeKutta3,
                            advection = UpwindBiasedFifthOrder(),
                            tracers = :b,
                            boundary_conditions = (u=u_bcs, v=v_bcs),
                            background_fields = (; b=B_field))
println(model)

# SIMULATION
# define simulation with time stepper, and callbacks for some runtime info
simulation = Simulation(model, Δt = Δt, stop_time = stop_time)
wizard = TimeStepWizard(max_change=1.1, cfl=0.7, max_Δt=max_Δt)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(4))

# Print a progress message
progress_message(sim) = @printf(" ▷ Iteration: %06d, time: %s, Δt: %s, max(|w|) = %.1e ms⁻¹, wall time: %s\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                maximum(abs, sim.model.velocities.w), prettytime(sim.run_wall_time))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(50))

# Add outputs to the simulation.
u, v, w = model.velocities
b = model.tracers.b
B∞ = model.background_fields.tracers.b

B = b + B∞
V = v + V∞
ωy = ∂z(u) - ∂x(w) # vorticity

outputs = (; u, V, w, B, ωy)

# OUTPUTS
simulation.output_writers[:fields] = NetCDFOutputWriter(model, outputs;
                                                        filename = path * experiment * ".nc",
                                                        schedule = TimeInterval(Δt_output_fld),
                                                        overwrite_existing = true)

# Now we just run it!
run!(simulation)