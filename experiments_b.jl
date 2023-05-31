using Printf
using CUDA
using Oceananigans
using Oceananigans.BuoyancyModels: g_Earth
using Oceananigans.Units: seconds, minute, minutes, hour, hours, kilometer, kilometers, meters
using Oceananigans.TurbulenceClosures
using TOML
include("utils.jl")

config = TOML.parsefile(ARGS[1])

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

const z₀ = parse_units(grid_params["z₀"])

# Z-GRID PROPERTIES (refinement of Δz at ice-ocean interface)
const refinement = 1.2 # controls spacing near surface (higher means finer spaced)
const stretching = 12  # controls rate of stretching at bottom

z_faces = z_levels(Nz,Lz,z₀,refinement,stretching)

grid = RectilinearGrid(ARCH;
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = z_faces,
                       topology=(Periodic, Periodic, Bounded))
println(grid)

# MODEL
# Far-field values
far_field = config["far-field values"]
const u₀ = far_field["V∞"]
const T₀ = far_field["T∞"]
const S₀ = far_field["S∞"]
# Parameters
params = config["physical parameters"]
const κₜ = params["κₜ"]
const κₛ = params["κₛ"]
const ν  = params["ν"]
const α  = params["α"]
const β  = params["β"]
const f₀ = params["f₀"]
const cᴰ = params["cᴰ"]

# Boundary conditions: drag at the top (mimicking a solid ice interface)
@inline u_quadratic_drag(x, y, t, u, v, p) = p.cᴰ * u * sqrt(u^2 + v^2)
@inline v_quadratic_drag(x, y, t, u, v, p) = p.cᴰ * v * sqrt(u^2 + v^2)

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(u_quadratic_drag, field_dependencies=(:u, :v), parameters=(;cᴰ=cᴰ)))
v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(v_quadratic_drag, field_dependencies=(:u, :v), parameters=(;cᴰ=cᴰ)))

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = α, haline_contraction = β))

coriolis = FPlane(f=f₀)

#closure = AnisotropicMinimumDissipation()
closure = ScalarDiffusivity(ν=ν, κ=(S=κₛ, T=κₜ))

model = NonhydrostaticModel(; grid, buoyancy,
                            advection = UpwindBiasedFifthOrder(),
                            timestepper = :RungeKutta3,
                            tracers = (:T, :S),
                            coriolis = coriolis,
                            closure = closure,
                            boundary_conditions = (;u=u_bcs, v=v_bcs))
println(model)

# SIMULATION
# define simulation with time stepper, and callbacks for some runtime info
# Random noise damped at top and bottom
@inline Ξ(z) = randn() * (z - z₀) / model.grid.Lz * (1 + (z - z₀) / model.grid.Lz) # noise

# INITIAL CONDITIONS
# Temperature initial condition: a stable gradient with random noise superposed.
@inline Tᵢ(x, y, z) = T₀ - 9.5e-4 * (model.grid.Lz + z - z₀) + model.grid.Lz * 1e-6 * Ξ(z)
# Salinity initial condition: a stable gradient with random noise superposed.
@inline Sᵢ(x, y, z) = S₀ - 4e-4 * (model.grid.Lz + z - z₀) + model.grid.Lz * 1e-6 * Ξ(z)
# Velocity initial condition: random noise scaled by the friction velocity.
@inline uᵢ(x, y, z) = 1e-5 * Ξ(z)
# `set!` the `model` fields using functions or constants:
set!(model, v=uᵢ, u=u₀, w=uᵢ, T=Tᵢ, S=Sᵢ)

# define simulation with time stepper, and callbacks for some runtime info
simulation = Simulation(model, Δt =  Δt, stop_time=stop_time)
wizard = TimeStepWizard(cfl=0.5, max_change=1.1, max_Δt=max_Δt)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# Print a progress message
progress_message(sim) = @printf(" ▷ Iteration: %06d, time: %s, Δt: %s, wall time: %s\n\t max(|w|) = %.1e ms⁻¹, min(T) = %.3f °C, max(T) %.3f °C, min(S) = %.2f g kg⁻¹, max(S) %.2f g kg⁻¹\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),prettytime(sim.run_wall_time),
                                maximum(abs, sim.model.velocities.w),
                                minimum(sim.model.tracers.T), maximum(sim.model.tracers.T),
                                minimum(sim.model.tracers.S), maximum(sim.model.tracers.S))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(20))

# OUTPUTS
simulation.output_writers[:field_writer] = NetCDFOutputWriter(model, merge(model.velocities, model.tracers), filename = experiment * ".nc", overwrite_existing = true, schedule=TimeInterval(Δt_output_fld))

run!(simulation)

