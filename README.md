# Polar Ocean - Large Eddy Simulations

A suite of [*Julia*](https://julialang.org/downloads/) scripts for running polar ocean large eddy simulations on your computer (with GPU support) using [*Oceananigans.jl*](https://github.com/CliMA/Oceananigans.jl).

*Oceananigans.jl* has a fantastic [documentation](https://clima.github.io/OceananigansDocumentation/stable) with lots of examples and a very active [discussions](https://github.com/CliMA/Oceananigans.jl/discussions) page with many useful tip and tricks.

1. [Download Julia](https://julialang.org/downloads/).
2. Launch Julia and type
```
julia> using Pkg

julia> Pkg.add("Oceananigans")

julia> Pkg.add("ArgParse")
```

## Experiments 0

This is just a small (but full) non-hydrostatic ocean model on a 1x1x1 rectilinear grid, to test if everything works.

```
julia experiments_0.jl
```

Output:
```
 ▷ Experiment: 'experiments_0' ◁ 
1×1×1 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on GPU with 3×3×3 halo
├── Periodic x ∈ [0.0, 1.0) regularly spaced with Δx=1.0
├── Periodic y ∈ [0.0, 1.0) regularly spaced with Δy=1.0
└── Bounded  z ∈ [0.0, 1.0] regularly spaced with Δz=1.0
NonhydrostaticModel{GPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── grid: 1×1×1 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on GPU with 3×3×3 halo
├── timestepper: QuasiAdamsBashforth2TimeStepper
├── tracers: ()
├── closure: Nothing
├── buoyancy: Nothing
└── coriolis: Nothing
[ Info: Initializing simulation...
 ▷ Iteration: 00000, time: 0 seconds, Δt: 1 second, wall time: 0 seconds, max(|w|) = 0.0e+00 ms⁻¹
[ Info:     ... simulation initialization complete (15.441 seconds)
[ Info: Executing initial time step...
[ Info:     ... initial time step complete (1.025 minutes).
 ▷ Iteration: 00020, time: 20 seconds, Δt: 1 second, wall time: 1.294 minutes, max(|w|) = 0.0e+00 ms⁻¹
 ▷ Iteration: 00040, time: 40 seconds, Δt: 1 second, wall time: 1.298 minutes, max(|w|) = 0.0e+00 ms⁻¹
 ▷ Iteration: 00060, time: 1 minute, Δt: 1 second, wall time: 1.300 minutes, max(|w|) = 0.0e+00 ms⁻¹
 ▷ Iteration: 00080, time: 1.333 minutes, Δt: 1 second, wall time: 1.303 minutes, max(|w|) = 0.0e+00 ms⁻¹
[ Info: Simulation is stopping. Model time 1.667 minutes has hit or exceeded simulation stop time 1.667 minutes.
 ▷ Iteration: 00100, time: 1.667 minutes, Δt: 1 second, wall time: 1.305 minutes, max(|w|) = 0.0e+00 ms⁻¹
```

## Experiments A

A basic experiment where we cool the surface of a 10mx10mx10m ocean box.

```
julia experiments_a.jl -e <path and prefix for this experiment>
```

## Experiments B

A basic experiment similar to A but where we also add a drag to the surface, mimicking a solid ice interface at the top of the 10mx10mx10m ocean box.

```
julia experiments_b.jl -e <path and prefix for this experiment>
```

## Experiments C

*Coming soon...*

Similar to B with melting at the ice-ocean interface.

## Experiments D

*Coming soon...*

A tilted ocean box with drag at the ice-ocean interface.

## Experiments E

*Coming soon...*

An open ocean box with imposed wind stress, cooling, and evaporation at the surface.

## Experiments F

*Coming soon...*

A bigger open ocean box like E but with a circular sea ice-free mask imposed. Fluxes are basically zero where the surface is masked by "sea ice".

