# Gymnasium

Minimal Interface for [Gymnasium](https://gymnasium.farama.org/content/basic_usage/), an RL environment repository which replaces OpenAI Gym. 


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
