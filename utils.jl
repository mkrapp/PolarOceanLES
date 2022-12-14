using ArgParse

function parse_experiment(experiment)
    s = ArgParseSettings()
    @add_arg_table s begin
        "--experiment", "-e"
            help = "starting value of array"
            arg_type = String
            default = experiment
    end
    return parse_args(s)["experiment"]
end

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
