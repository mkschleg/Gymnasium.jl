# Gymnasium

Minimal Interface for [Gymnasium](https://gymnasium.farama.org/content/basic_usage/), an RL environment repository which replaces OpenAI Gym. Might implement compatibility layer for CommonRLInterface or MinimalRLCore if there is interest.

## Interface

```julia
make(id::String;
     unwrap=false, # whether the environment is returned with the default wrappers or unwrapped
     render_mode=nothing, # corresponds to the render_mode in the gymnasium environment
     max_episode_steps=nothing, 
     autoreset=false,
     disable_env_checker=nothing, 
     kwargs... # Untested, but technically should send optional kwargs to environment constructor in python.
     )
make(name::String, version::Int; kwargs...) # calls the above function
```

Then the interface is juliafied in the usual ways:

```julia
step!(env::GymnasiumEnv, action)
reset!(env::GymnasiumEnv{T}; seed::Union{Nothing, Int}=nothing, options::Union{Nothing, Dict}=nothing) where {T}
render(env::GymnasiumEnv)
close!(env::GymnasiumEnv)
```



## Environment Compatibility
I have not tested on every environment. To add compatibility for an observation type that I don't implement `convert_obs` similar to the ndarray function:

```julia
convert_obs(::Val{:ndarray}, pyobs::Py) = pyconvert(Vector, pyobs)
```

where `:ndarray` comes from the call of `convert_obs` in the GymnasiumEnv wrapper class:

```julia
function GymnasiumEnv(id::String, pyenv::Py)
    obs, info = pyenv.reset() # reset to get the obs type
    env = GymnasiumEnv(pyenv, id, convert_obs(obs), false, 0.0, info)
    return env
end

function convert_obs(pyobs::Py)
    t_str = pyconvert(String, @py type(pyobs).__name__)
    convert_obs(Val(Symbol(t_str)), pyobs)
end
```

The info dict is returned as a `PythonCall.Py` object, and the reward and terminal values are assumed to be Float64 and Bool respectively. 

