module Gymnasium

# Write your package code here.
import PythonCall: PythonCall, Py, @py, pyconvert, pybuiltins, pyimport, pyis

pygym::Py = PythonCall.pynew() # initially NULL
function __init__()
    PythonCall.pycopy!(pygym, pyimport("gymnasium"))
    PythonCall.pyconvert_add_rule("gymnasium.spaces.discrete:Discrete", DiscreteSpace, convert_discrete_space)
end


######## Spaces #########

struct DiscreteSpace
    n::Int
    start::Int
end

function convert_discrete_space(::Type{DiscreteSpace}, pyobj::Py)
    n = Gymnasium.pyconvert(Int, pyobj.n)
    start = Gymnasium.pyconvert(Int, pyobj.start)
    ds = DiscreteSpace(n, start)
    return Gymnasium.PythonCall.pyconvert_return(ds)
end

################

######## Environment ########

mutable struct GymnasiumEnv{T, AS, OS}
    pyenv::Py
    const id::String
    const action_space::AS
    const observation_space::OS

    _ex_obs::T
end

function GymnasiumEnv(id::String, pyenv::Py)
    obs, info = pyenv.reset() # reset to get the obs type
    as = pyconvert(Any, pyenv.action_space)
    os = pyconvert(Any, pyenv.observation_space)
    env = GymnasiumEnv(pyenv, id, as, os, convert_obs(obs))
    return env
end

GymnasiumEnv(id::String; kwargs...) = make(id; kwargs...)
GymnasiumEnv(name::String, version::Int; kwargs...) = make(name, version; kwargs...)

ispy(::GymnasiumEnv) = true
Py(env::GymnasiumEnv) = env.pyenv

function convert_obs(::GymnasiumEnv{T}, pyobs::Py) where T
    pyconvert(T, pyobs)
end

convert_obs(::T, pyobs::Py) where T = pyconvert(Any, pyobs)

function convert_obs(::NamedTuple{N, T}, pyobs::Py) where {N, T}
    types = T.parameters
    NamedTuple{N}(pyconvert(types[i], pyobs[string(N[i])]) for i in 1:length(N))
end


function convert_obs(pyobs::Py)
    t_str = pyconvert(String, @py type(pyobs).__name__)
    convert_obs(Val(Symbol(t_str)), pyobs)
end

convert_obs(::Val{:ndarray}, pyobs::Py) = pyconvert(Array, pyobs)

function convert_obs(::Val{:dict}, pyobs::Py)
    d = Dict{String, Any}()
    for key in pyobs.keys()
        d[string(key)] = convert_obs(pyobs[key])
    end
    (; zip(Symbol.(keys(d)), values(d))...)
end


function make(id::String;
              unwrap=false,
              render_mode=nothing,
              max_episode_steps=nothing,
              autoreset=false,
              disable_env_checker=nothing, kwargs...)
    pyenv = pygym.make(id,
                       render_mode=render_mode,
                       max_episode_steps=max_episode_steps,
                       disable_env_checker=disable_env_checker,
                       kwargs...)

    if unwrap
        pyenv = pyenv.unwrapped
    end

    GymnasiumEnv(id, pyenv)
end

function make(name::String, version::Int; kwargs...)
    id = name * "-v" * string(version)
    make(id::String; kwargs...)
end

function step!(env::GymnasiumEnv, action)

    if pyis(env.pyenv, pybuiltins.None)
        throw("GymnasiumEnv: pyenv None in step!")
    end
    
    observation, reward, terminal, truncated, info = env.pyenv.step(action)
    obs = convert_obs(env, observation)
    term = pyconvert(Bool, terminal)
    rew = pyconvert(Float64, reward)
    trun = pyconvert(Bool, truncated)

    obs, rew, term, trun, info
end

function reset!(env::GymnasiumEnv{T}; seed::Union{Nothing, Int}=nothing, options::Union{Nothing, Dict}=nothing) where {T}
    
    if pyis(env.pyenv, pybuiltins.None)
        throw("GymnasiumEnv: pyenv None in reset!")
    end
    obs, info = env.pyenv.reset(seed=seed, options=options)
    return convert_obs(env, obs), info
end

function render(env::GymnasiumEnv)
    if pyis(env.pyenv, pybuiltins.None)
        throw("GymnasiumEnv: pyenv None in render")
    end
    env.pyenv.render()
end


function close!(env::GymnasiumEnv)
    if pyis(env.pyenv, pybuiltins.None)
        return
    end
    env.pyenv.close()
    env.pyenv = pybuiltins.None
    return
end


# # CommonRLInterface
# module _CRLI
# import CommonRLInterface
# end



# # MinimalRLCore
# module _MRLC
# import MinimalRLCore
# end



end
