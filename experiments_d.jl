using Printf
using CUDA
using Oceananigans
using Oceananigans.BuoyancyModels: g_Earth
using Oceananigans.Units: seconds, minute, minutes, hour, hours, kilometer, kilometers, meters
using Oceananigans.TurbulenceClosures
using Oceananigans.BoundaryConditions: getbc
using Oceananigans: fields
using Oceanostics
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
const T₀ = far_field["T∞"]
const S₀ = far_field["S∞"]
const dTdz = far_field["dTdz"]
const dSdz = far_field["dSdz"]
# Physical parameters
params = config["physical parameters"]
const f₀ = params["f₀"]
const cᴰ = params["cᴰ"]
const κₜ = params["κₜ"]
const κₛ = params["κₛ"]
const κ  = params["κ"]
const ν  = params["ν"]
const α  = params["α"]
const β  = params["β"]
const λ₁ = params["λ₁"]
const λ₂ = params["λ₂"]
const λ₃ = params["λ₃"]
const Lf = params["Lf"]
const cₚ = params["cₚ"]
const ρₒ = params["ρₒ"]
const ρₐ = params["ρₐ"]
const ρᵢ = params["ρᵢ"]
const de = params["de"]
const df = params["df"]
const Nu = params["Nu"]
const kw = params["kw"]
# Forcing
forcing = config["forcing"]
const Qʰ = forcing["Qʰ"]
const u₁₀ = forcing["u₁₀"]
const Qf = forcing["Qf"]


# surface _temperature_ flux
Qᵀ = Qʰ / (ρₒ * cₚ) # K m s⁻¹, 
# surface _momentum_ flux (wind-stress)
Qᵘ = - ρₐ / ρₒ * cᴰ * u₁₀ * abs(u₁₀) # m² s⁻²
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ), bottom = GradientBoundaryCondition(0.0))
v_bcs = FieldBoundaryConditions(bottom = GradientBoundaryCondition(0.0))

# temperature boundary conditions: bottom temperature gradient
T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵀ),
                                bottom = GradientBoundaryCondition(dTdz))

# salinity boundary conditions: bottom salinity gradient
S_bcs = FieldBoundaryConditions(bottom = GradientBoundaryCondition(dSdz))

# frazil concentration boundary conditons: surface frazil flux
F_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qf))

buoyancy = Buoyancy(model = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion = α, haline_contraction = β)))

closure = AnisotropicMinimumDissipation()
# closure = ScalarDiffusivity(ν=ν, κ=κ)

coriolis = FPlane(f=f₀)

# frazil dynamics
# liquidus condition for seawater temperature
@inline Tf(x, y, z, t, S, params) = params.λ₁*S + params.λ₂ + params.λ₃*z
# Eq(9) in Omstedt (1984); heat transfer
@inline q(x, y, z, t, T, S, params) = params.Nu * params.kw / params.de * (Tf(x, y, z, t, S, params) - T)
# Eq(11) in Omstedt (1984); heat transfer
@inline GT(x, y, z, t, F, T, S, params) = 4 * F * q(x, y, z, t, T, S, params) / (params.df * params.ρₒ * params.cₚ)
# Eq(12) in Omstedt (1984); frazil formation
@inline GF(x, y, z, t, F, T, S, params) = 4 * F * q(x, y, z, t, T, S, params) / (params.df * params.ρᵢ * params.Lf)
# Eq(13) in Omstedt (1984); salt rejection
@inline GS(x, y, z, t, F, T, S, params) = 4 * F * q(x, y, z, t, T, S, params) * S / (params.df * params.ρₒ * params.Lf)

frazil_parameters = (λ₁ = λ₁, λ₂ = λ₂, λ₃ = λ₃, Nu = Nu, kw = kw, de = de, df = df, ρₒ = ρₒ, cₚ = cₚ, Lf = Lf, ρᵢ = ρᵢ,)

frazil_dynamics_T = Forcing(GT, field_dependencies = (:F, :T, :S), parameters = frazil_parameters)
frazil_dynamics_F = Forcing(GF, field_dependencies = (:F, :T, :S), parameters = frazil_parameters)
frazil_dynamics_S = Forcing(GS, field_dependencies = (:F, :T, :S), parameters = frazil_parameters)

V_d = π*(df/2)^2*de       # disc volume
deq = 2*(3/4*V_d/π)^(1/3) # equivalent sphere diameter
#println(deq)
Δb = g_Earth * (ρₒ - ρᵢ) / ρₒ   # m s⁻²
w_frazil = 2/9 * Δb / ν * deq^2 # m s⁻¹
#println(w_frazil)

rising = AdvectiveForcing(UpwindBiasedFifthOrder(), w=w_frazil)

model = NonhydrostaticModel(; grid, buoyancy,
                            advection = UpwindBiasedFifthOrder(),
                            timestepper = :RungeKutta3,
                            coriolis = coriolis,
                            tracers = (:T,:S,:F),
                            closure = closure,
                            forcing = (; T=frazil_dynamics_T, S=frazil_dynamics_S, F=(frazil_dynamics_F, rising)),
                            boundary_conditions = (;u=u_bcs, v=v_bcs, T=T_bcs, S=S_bcs, F=F_bcs))
println(model)

# SIMULATION
# define simulation with time stepper, and callbacks for some runtime info
# Random noise damped at top and bottom
@inline Ξ(z) = randn() * (z - z₀) / model.grid.Lz * (1 + (z - z₀) / model.grid.Lz) # noise

# INITIAL CONDITIONS
@inline Tᵢ(x, y, z) = T₀ - dTdz * (model.grid.Lz + z - z₀) + model.grid.Lz * 1e-6 * Ξ(z)
# Salinity initial condition: a stable gradient with random noise superposed.
@inline Sᵢ(x, y, z) = S₀ - dSdz * (model.grid.Lz + z - z₀) + model.grid.Lz * 1e-6 * Ξ(z)
# Velocity initial condition: random noise scaled by the friction velocity.
@inline vᵢ(x, y, z) = sqrt(abs(Qᵘ)) * 1e-3 * Ξ(z)
# `set!` the `model` fields using functions or constants:
set!(model, u=vᵢ, w=vᵢ, T=Tᵢ, S=Sᵢ, F=0.0)

# define simulation with time stepper, and callbacks for some runtime info
simulation = Simulation(model, Δt =  Δt, stop_time=stop_time)
wizard = TimeStepWizard(cfl=0.5, max_change=1.1, max_Δt=max_Δt)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(100))

# Print a progress message
progress_message(sim) = @printf(" ▷ Iteration: %06d, time: %s, Δt: %s, wall time: %s, max(|w|) = %.1e ms⁻¹\n\t T = [%.3g, %.3g], S = [%.3g, %.3g], F = [%.3g, %.3g]\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),prettytime(sim.run_wall_time),
                                maximum(abs, sim.model.velocities.w),
                                minimum(sim.model.tracers.T), maximum(sim.model.tracers.T),
                                minimum(sim.model.tracers.S), maximum(sim.model.tracers.S),
                                minimum(sim.model.tracers.F), maximum(sim.model.tracers.F))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(100))

# OUTPUTS
u, v, w = model.velocities
s = sqrt(u^2 + v^2 + w^2)
ωy = ∂z(u) - ∂x(w)

tke = Field(TurbulentKineticEnergy(model))
shear_production_op = @at (Center, Center, Center) ∂z(u)^2 + ∂z(v)^2 + ∂z(w)^2
sp = Field(shear_production_op)

# Boundary condition extractor in "kernel function form"
@inline kernel_getbc(i, j, k, grid, boundary_condition, clock, fields) =
    getbc(boundary_condition, i, j, grid, clock, fields)

# Kernel arguments
grid = model.grid
clock = model.clock
model_fields = merge(fields(model), model.auxiliary_fields)
u_bc = u.boundary_conditions.top
v_bc = v.boundary_conditions.top

# Build operations
u_bc_op=KernelFunctionOperation{Face, Center, Nothing}(kernel_getbc, grid, u_bc, clock, model_fields)
v_bc_op=KernelFunctionOperation{Center, Face, Nothing}(kernel_getbc, grid, v_bc, clock, model_fields)

# Build Fields
Qᵘ = Field(u_bc_op)
Qᵛ = Field(v_bc_op)

u★ = sqrt(sqrt(Qᵘ^2 + Qᵛ^2))

T, S, F = model.tracers

outputs = (; u, v, w, T, S, F, s, ωy, tke, sp, u★)
simulation.output_writers[:field_writer] = NetCDFOutputWriter(model, outputs, filename = experiment * ".nc", overwrite_existing = true, schedule=TimeInterval(Δt_output_fld), global_attributes = config2dict(config))

run!(simulation)

