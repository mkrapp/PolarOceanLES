using Printf
using CUDA
using Oceananigans
using Oceananigans.BuoyancyModels: g_Earth
using Oceananigans.Units: seconds, minute, minutes, hour, hours, kilometer, kilometers, meters
using Oceananigans.TurbulenceClosures
using Oceananigans.BoundaryConditions: getbc
using Oceananigans: fields
#using Oceanostics
using TOML
include("utils.jl")

if length(ARGS) == 0
    println("Enter name of configuration file:")
    config = TOML.parsefile(readline())
else
    config = TOML.parsefile(ARGS[1])
end

# model runtime parameters: number of hours, grid size, filename, etc
sim_params = config["simulation"]
const stop_time     = parse_units(sim_params["stop_time"])
const Δt            = parse_units(sim_params["Δt"])
const max_Δt        = parse_units(sim_params["max_Δt"])
const Δt_output_fld = parse_units(sim_params["Δt_output_fld"])

ARCH = has_cuda_gpu() ? GPU() : CPU()

path       = config["path"]
experiment = config["experiment"]
@printf(" ▷▷▷ Experiment: '%s' ◁◁◁ \n", experiment)

# GRID DIMENSIONS
grid_params = config["grid"]
const Nx = grid_params["Nx"]
const Ny = grid_params["Ny"]
const Nz = grid_params["Nz"]
const Lx = parse_units(grid_params["Lx"])
const Ly = parse_units(grid_params["Ly"])
const Lz = parse_units(grid_params["Lz"])
# Z-GRID PROPERTIES (refinement of Δz at ice-ocean interface)
const refinement = grid_params["refinement"]
const stretching = grid_params["stretching"]

const (SIZE, TOPOLOGY) = size_and_topology((Nx, Ny, Nz))

const z₀ = parse_units(grid_params["z₀"])

z_faces = z_levels(Nz,Lz,z₀,refinement,stretching)

grid = RectilinearGrid(ARCH;
                       size = SIZE,
                       x = (0, Lx),
                       y = (0, Ly),
                       z = z_faces,
                       topology=TOPOLOGY)
println(grid)

# MODEL
# Far-field values
far_field = config["far-field values"]
const u₀ = far_field["V∞"]
const N² = far_field["N²"]
# Parameters
params = config["physical parameters"]
const f₀ = params["f₀"]
const κ  = params["κ"]
const ν  = params["ν"]
const cᴰ = params["cᴰ"]

# Boundary conditions: drag at the top (mimicking a solid ice interface)
@inline u_quadratic_drag(x, y, t, u, v, p) = p.cᴰ * u * sqrt(u^2 + v^2)
@inline v_quadratic_drag(x, y, t, u, v, p) = p.cᴰ * v * sqrt(u^2 + v^2)

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(u_quadratic_drag, field_dependencies=(:u, :v), parameters=(;cᴰ=cᴰ)), bottom = GradientBoundaryCondition(0.0))
v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(v_quadratic_drag, field_dependencies=(:u, :v), parameters=(;cᴰ=cᴰ)), bottom = GradientBoundaryCondition(0.0))

buoyancy = Buoyancy(model = BuoyancyTracer())

closure = ScalarDiffusivity(ν=ν, κ=κ)

coriolis = FPlane(f=f₀)

model = NonhydrostaticModel(; grid, buoyancy,
                            advection = UpwindBiasedFifthOrder(),
                            timestepper = :RungeKutta3,
                            coriolis = coriolis,
                            tracers = (:b),
                            closure = closure,
                            boundary_conditions = (;u=u_bcs, v=v_bcs))
println(model)

# SIMULATION
# define simulation with time stepper, and callbacks for some runtime info
# Random noise damped at top and bottom
@inline Ξ(z) = randn() * (z - z₀) / model.grid.Lz * (1 + (z - z₀) / model.grid.Lz) # noise

# INITIAL CONDITIONS
# Initial stable stratification.
@inline bᵢ(x, y, z) = N² * (model.grid.Lz + z - z₀) + model.grid.Lz * 1e-6 * Ξ(z)
# Velocity initial condition: random noise scaled by the friction velocity.
@inline uᵢ(x, y, z) = 1e-5 * Ξ(z)
# `set!` the `model` fields using functions or constants:
set!(model, v=uᵢ, u=u₀, w=uᵢ, b=bᵢ)

# define simulation with time stepper, and callbacks for some runtime info
simulation = Simulation(model, Δt =  Δt, stop_time=stop_time)
wizard = TimeStepWizard(cfl=0.5, max_change=1.1, max_Δt=max_Δt)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# Print a progress message
progress_message(sim) = @printf(" ▷ Iteration: %06d, time: %s, Δt: %s, wall time: %s\n\t max(|w|) = %.1e ms⁻¹, min(b) = %.3g , max(b) %.3g\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),prettytime(sim.run_wall_time),
                                maximum(abs, sim.model.velocities.w),
                                minimum(sim.model.tracers.b), maximum(sim.model.tracers.b))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(20))

# OUTPUTS
u, v, w = model.velocities
s = sqrt(u^2 + v^2 + w^2)
ωy = ∂z(u) - ∂x(w)

#tke = Field(TurbulentKineticEnergy(model))
shear_production_op = @at (Center, Center, Center) ∂z(u)^2 + ∂z(v)^2 + ∂z(w)^2
sp = Field(shear_production_op)

## Boundary condition extractor in "kernel function form"
#@inline kernel_getbc(i, j, k, grid, boundary_condition, clock, fields) =
#    getbc(boundary_condition, i, j, grid, clock, fields)
#
## Kernel arguments
#grid = model.grid
#clock = model.clock
#model_fields = merge(fields(model), model.auxiliary_fields)
#u_bc = u.boundary_conditions.top
#v_bc = v.boundary_conditions.top
#
## Build operations
#u_bc_op=KernelFunctionOperation{Face, Center, Nothing}(kernel_getbc, grid, u_bc, clock, model_fields)
#v_bc_op=KernelFunctionOperation{Center, Face, Nothing}(kernel_getbc, grid, v_bc, clock, model_fields)
#
## Build Fields
#Qᵘ = Field(u_bc_op)
#Qᵛ = Field(v_bc_op)
#
#u★ = sqrt(sqrt(Qᵘ^2 + Qᵛ^2))

outputs = (; u, v, w, model.tracers.b, s, ωy, sp)
simulation.output_writers[:field_writer] = NetCDFOutputWriter(model, outputs, filename = path * experiment * ".nc", overwrite_existing = true, schedule=TimeInterval(Δt_output_fld), global_attributes = config2dict(config))

run!(simulation)

