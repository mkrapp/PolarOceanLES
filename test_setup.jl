using Printf
using CUDA
using Oceananigans
using TOML
include("utils.jl")

config = TOML.parsefile(ARGS[1])

# model runtime parameters: number of hours, grid size, filename, etc
config_sim = config["simulation"]
const stop_time = parse_units(config_sim["stop_time"])
const Δt        = parse_units(config_sim["Δt"])

ARCH = has_cuda_gpu() ? GPU() : CPU()

path       = config["path"]
experiment = config["experiment"]
@printf(" ▷ Experiment: '%s' ◁ \n", experiment)

config_grid = config["grid"]
# GRID DIMENSIONS
const Nx = config_grid["Nx"]
const Ny = config_grid["Ny"]
const Nz = config_grid["Nz"]
# GRID EXTENT
const Lx = parse_units(config_grid["Lx"])
const Ly = parse_units(config_grid["Ly"])
const Lz = parse_units(config_grid["Lz"])

grid = RectilinearGrid(ARCH; size=(Nx, Ny, Nz), x=(0, Lx), y=(0,Ly), z=(0,Lz))
println(grid)

# MODEL
model = NonhydrostaticModel(; grid)
println(model)

# SIMULATION
# define simulation with time stepper, and callbacks for some runtime info
simulation = Simulation(model, Δt =  Δt, stop_time=stop_time)

# Print a progress message
progress_message(sim) = @printf(" ▷ Iteration: %05d, time: %s, Δt: %s, wall time: %s, max(|w|) = %.1e ms⁻¹\n",
                                iteration(sim), prettytime(sim), prettytime(sim.Δt),prettytime(sim.run_wall_time),
                                maximum(abs, sim.model.velocities.w))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(20))

# OUTPUTS
simulation.output_writers[:field_writer] = NetCDFOutputWriter(model, merge(model.velocities), filename = path * experiment * ".nc", overwrite_existing = true, schedule=TimeInterval(10))

run!(simulation)

