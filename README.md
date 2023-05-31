# Polar Ocean - Large Eddy Simulations

A suite of [*Julia*](https://julialang.org/downloads/) scripts for running polar ocean large eddy simulations on your computer (with GPU support) using [*Oceananigans.jl*](https://github.com/CliMA/Oceananigans.jl).

*Oceananigans.jl* has a fantastic [documentation](https://clima.github.io/OceananigansDocumentation/stable) with lots of examples and a very active [discussions](https://github.com/CliMA/Oceananigans.jl/discussions) page with many useful tip and tricks.

1. [Download Julia](https://julialang.org/downloads/).
2. Launch Julia and type
```
julia> using Pkg

julia> Pkg.add("Oceananigans")

julia> Pkg.add("Oceanostics")
```

## Test setup

This is just a small (but full) non-hydrostatic ocean model on a 1x1x1 rectilinear grid, to test if everything works.

```bash
julia test_setup.jl test_setup.toml
```

Output:
```
 ▷ Experiment: 'test_setup' ◁
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

A basic experiment where we simulate the cooling of a surface of a 10mx10mx10m ocean box for 5 hours. It is initialized with a far-field velocity of 10.0 cm/s, a far-field temperature of -1.9°C (and $\partial T/\partial z$ = 9.5e-4 °C/m) and a salinity of 34.5 g/kg (and $\partial S/\partial z$ = 4.e-4 g/kg/m).

```bash
julia experiments_a.jl experiments_a.toml
```

## Experiments B

A basic experiment similar to A but instead of surface cooling we apply a surface drag, mimicking a solid ice interface (such as sea ice cover or an ice shelf) at the top of the 10mx10mx10m ocean box.

```bash
julia experiments_b.jl experiments_b.toml
```

## Experiments C

Like B but only a slice in the x-z plane, without Coriolis. And only buoyancy as a tracer (the combined effect of temperature and salinity variations due to gravity). Top drag follows Monin-Obukhov theory.

```bash
julia experiments_c.jl experiments_c.toml
```


## Other experiments

*Coming soon...*

- [ ] Similar to B with melting at the ice-ocean interface.
- [ ] A tilted ocean box with drag at the ice-ocean interface.
- [ ] An open ocean box with imposed wind stress, cooling, and evaporation at the surface.
- [ ] A bigger open ocean box like E but with a circular sea ice-free mask imposed. Fluxes are basically zero where the surface is masked by "sea ice".

