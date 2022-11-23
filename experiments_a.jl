using Printf
using CUDA
using Oceananigans
using Oceananigans.BuoyancyModels: g_Earth
using Oceananigans.Units: seconds, minute, minutes, hour, hours, kilometer, kilometers, meters
using Oceananigans.TurbulenceClosures
include("utils.jl")

# model runtime parameters: number of hours, grid size, filename, etc
const stop_time        = 5hours
const Δt               = 1seconds
const max_Δt           = 1seconds
const Δt_output_fld    = 5seconds

ARCH = has_cuda_gpu() ? GPU() : CPU()

experiment = parse_experiment("experiments_a")
@printf(" ▷▷▷ Experiment: '%s' ◁◁◁ \n", experiment)

# GRID DIMENSIONS
const Nz = 50
const Nx = 25
const Ny = 25
# GRID EXTENT
const Lz = 10meters
const Lx = 10meters
const Ly = 10meters

z₀ = 0meters # depth of seawater parcel

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
const u₀ = 0.014    # far field velocity                 m s⁻¹
const T₀ = -1.9     # far field temperature              °C
const S₀ = 34.5     # far field salinity                 g kg⁻¹
# Parameters
const κₜ = 1.4e-7   # molecular diffusivity of heat      m² s⁻¹
const κₛ = 1.3e-9   # molecular diffusivity of salt      m² s⁻¹
const ν  = 2.0e-6   # molecular viscosity                m² s⁻¹
const α  = 3.8e-5   # thermal expansion coefficient      °C⁻¹
const β  = 7.8e-4   # haline expansion coefficient       kg g⁻¹
const f₀ = -1.37e-4 # Coriolis parameter (for 70°S)      rad s⁻¹

# Far-field forcing from constant pressure gradient
const pgrad = - f₀ * u₀
@inline pressure_gradient(x, y, z, t) = pgrad
u_forcing = Forcing(pressure_gradient)

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
                            forcing = (;u=u_forcing))
println(model)

# SIMULATION
# define simulation with time stepper, and callbacks for some runtime info
# Random noise damped at top and bottom
@inline Ξ(z) = randn() * (z - z₀) / model.grid.Lz * (1 + (z - z₀) / model.grid.Lz) # noise

# INITIAL CONDITIONS
# Temperature initial condition: a stable gradient with random noise superposed.
const z_top = CUDA.@allowscalar model.grid.zᵃᵃᶜ[model.grid.Nz]
@inline Tᵢ(x, y, z) = T₀ - 9.5e-4 * (model.grid.Lz + z - z₀) + model.grid.Lz * 1e-6 * Ξ(z) - (z == z_top ? exp(-(x-Lx/2)^2 -(y-Ly/2)^2) : 0.0)
# Salinity initial condition: a stable gradient with random noise superposed.
@inline Sᵢ(x, y, z) = S₀ - 4e-4 * (model.grid.Lz + z - z₀) + model.grid.Lz * 1e-6 * Ξ(z)
# Velocity initial condition: random noise scaled by the friction velocity.
@inline uᵢ(x, y, z) = 1e-5 * Ξ(z)
# `set!` the `model` fields using functions or constants:
set!(model, v=uᵢ, u=uᵢ, w=uᵢ, T=Tᵢ, S=Sᵢ)

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

