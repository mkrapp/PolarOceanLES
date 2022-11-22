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
