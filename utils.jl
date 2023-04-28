using Oceananigans.Units: second, seconds, minute, minutes, hour, hours, day, days, meter, meters, kilometer, kilometers

function z_levels(Nz,Lz,z₀,refinement,stretching)
    # Normalized height ranging from 0 to 1
    @inline h(k) = (k - 1) / Nz

    # Linear near-surface generator
    @inline ζ₀(k) = 1 + (h(k) - 1) / refinement

    # Bottom-intensified stretching function
    @inline Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

    # Generating function
    @inline z_faces(k) = z₀ + Lz * (ζ₀(k) * Σ(k) - 1)
    return z_faces
end

function parse_units(x)
    #val, unit = split(x)
    #return parse(Float64,val) * eval(Symbol(unit))
    return eval(Meta.parse(x))
end

function size_and_topology(dims)
    (Nx, Ny, Nz) = dims
    # determine grid size from dimensions -> (tuple)
    SIZE = Tuple([x for x in dims if x > 1])
    # determine grid topology from dimensions -> (tuple)
    TOPOLOGY = (Nx > 1 ? Periodic : Flat, Ny > 1 ? Periodic : Flat, Bounded)
    return (SIZE, TOPOLOGY)
end

function config2dict(config)
    buf = IOBuffer()
    TOML.print(buf, config)
    return Dict("config" => String(take!(buf)))
end

