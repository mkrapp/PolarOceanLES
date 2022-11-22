using Printf
using CUDA
using Oceananigans
include("utils.jl")

# model runtime parameters: number of hours, grid size, filename, etc
const stop_time        = 100
const Δt               = 1

ARCH = has_cuda_gpu() ? GPU() : CPU()

experiment = parse_experiment("experiments_0")
@printf(" ▷ Experiment: '%s' ◁ \n", experiment)

# GRID DIMENSIONS
const Nz = 1
const Nx = 1
const Ny = 1
# GRID EXTENT
const Lz = 1
const Lx = 1
const Ly = 1

grid = RectilinearGrid(ARCH; size=(1, 1, 1), x=(0, Lx), y=(0,Ly), z=(0,Lz))
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
simulation.output_writers[:field_writer] = NetCDFOutputWriter(model, merge(model.velocities), filename = experiment * ".nc", overwrite_existing = true, schedule=TimeInterval(10))

run!(simulation)

