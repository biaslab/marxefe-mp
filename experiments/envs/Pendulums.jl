module Pendulums
using Distributions # MvNormal
using Markdown
using LinearAlgebra

#using .SUtils
#using .DynamicSystemBase
#import .DynamicSystemBase: get_eom_md, ddzdt, inv_inertia_matrix
#import .SystemBase: state_transition!, measure, get_eom_md, get_state_dim, get_observation_dim, get_action_dim, get_control_dim
function not_implemented_error(datatype::DataType)
    func_name = stacktrace()[2].func
    error("$(func_name) not implemented for $datatype")
end
function has_nan_or_inf(vector)
    return any(x -> isnan(x) || isinf(x), vector)
end

function check_nan_or_inf(vector, errormsg::String)
    if ! has_nan_or_inf(vector) return end
    error("The vector $vector = $errormsg contains NaN or Inf.")
end

#using .SUtils

export System
# implemented functions
export step!
export get_realtime_xticks, get_state_dim, get_observation_dim, get_action_dim, get_control_dim
export get_control, get_state, get_tsteps, convert_action
# need to be implemented by inheriting module
export get_eom_md, state_transition!, measure
export DPendulum
export DynamicSystem

# Define shared interface functions as abstract
export ddzdt, inv_inertia_matrix, update_params


abstract type System end

# -------------------
# Core System Functions
# -------------------

function step!(sys::System, a::Vector{Float64}, ulims::Tuple{Float64, Float64})
    update!(sys, a, ulims)
    sys.k += 1
end

function update!(sys::System, a::Vector{Float64}, ulims::Tuple{Float64, Float64})
    u = convert_action(a, ulims)
    state_transition!(sys, u)
end

# -------------------
# Action and Time Management
# -------------------

convert_action(a::Vector{Float64}, ulims::Tuple{Float64, Float64}) = clamp.(a, ulims...)
get_tsteps(sys::System) = range(0, step=sys.Δt, length=sys.N)

function get_realtime_xticks(sys::System)
    xticks_pos = collect(0:(sys.N/5):sys.N) .* sys.Δt
    xticks_labels = string.(Int.(round.(xticks_pos)))
    return xticks_pos, xticks_labels
end


# -------------------
# State and Control Queries
# -------------------

get_control(sys::System) = sys.torque
get_state(sys::System) = sys.z

# -------------------
# Not Implemented Placeholders
# -------------------

get_state_dim(sys_type::Type{System}) = not_implemented_error(sys_type)
get_observation_dim(sys_type::Type{System}) = not_implemented_error(sys_type)
get_action_dim(sys_type::Type{System}) = not_implemented_error(sys_type)
get_control_dim(sys_type::Type{System}) = not_implemented_error(sys_type)
get_eom_md(sys::System) = not_implemented_error(typeof(sys))
state_transition!(sys::System, u::Vector{Float64}) = not_implemented_error(typeof(sys))
measure(sys::System) = not_implemented_error(typeof(sys))

#using .SUtils
#using .SystemBase
#import .SystemBase: get_eom_md, state_transition!, measure

# Define abstract types or interfaces for systems
abstract type DynamicSystem <: System end
abstract type Pendulum <: DynamicSystem end

const DEFAULTS_SYS = (
    DPendulum = (
        D_z = 2::Int,
        D_y = 2::Int,
        D_a = 2::Int,
        D_u = 2::Int,
        Δt = 0.01::Float64, # time step duration in s
        N = 500::Int, # number of episode steps
        z = zeros(4)::Vector{Float64}, # initial (hidden) state
        W = 1e-5*diagm(ones(2))::Matrix{Float64}, # measurement noise std matrix
        mass = [1.0, 1.0]::Vector{Float64}, # mass of each link
        length = [1.0, 1.0]::Vector{Float64}, # length of each link
        damping = 0.0::Float64 # damping value
    ),
)

mutable struct DPendulum <: Pendulum
    # time
    k           ::Integer
    Δt          ::Float64
    N           ::Integer
    # state + observation
    z           ::Vector{Float64}
    W           ::Matrix{Float64}
    # parameters
    mass        ::Vector{Float64}
    length      ::Vector{Float64}
    damping     ::Float64

    function DPendulum(;
        Δt::Float64=DEFAULTS_SYS.DPendulum.Δt,
        N::Int=DEFAULTS_SYS.DPendulum.N,
        z::Vector{Float64}=DEFAULTS_SYS.DPendulum.z,
        W::Matrix{Float64}=DEFAULTS_SYS.DPendulum.W,
        mass::Vector{Float64}=DEFAULTS_SYS.DPendulum.mass,
        length::Vector{Float64}=DEFAULTS_SYS.DPendulum.length,
        damping::Float64=DEFAULTS_SYS.DPendulum.damping
    )
        return new(1, Δt, N, z, W, mass, length, damping)
    end
end


function ddzdt(sys::DPendulum, z1::Float64, z2::Float64, dz1::Float64, dz2::Float64, u::Vector{Float64})
    m1, m2 = sys.mass
    l1, l2 = sys.length
    c1, c2 = sys.damping, sys.damping
    κ1, κ2 = 0.0, 0.0
    gravity = 9.81

    z1mz2 = z1 - z2
    check_nan_or_inf(z1mz2, "z1 - z2: $z1 - $z2")
    z2mz1 = z2 - z1
    check_nan_or_inf(z2mz1, "z2 - z1: $z2 - $z1")

    # Components system dynamics matrix
    Jx = 1/2 * m2 * l1 * l2
    μ1 = (1/2 * m1 + m2) * gravity * l1
    μ2 = 1/2 * m2 * gravity * l2

    sinz1mz2 = sin(z1mz2)
    sinz2mz1 = sin(z2mz1)

    ddz1 = -Jx * sinz1mz2 * dz2^2 - μ1 * sin(z1) + κ1 * sinz2mz1 - c1 * dz1 / l1 + u[1]
    ddz2 =  Jx * sinz1mz2 * dz1^2 - μ2 * sin(z2) + κ2 * sinz2mz1 - c2 * dz2 / l2 + u[2]

    return inv_inertia_matrix(sys)*[ddz1, ddz2]
end


function state_transition!(sys::DynamicSystem, u::Vector{Float64})
    check_nan_or_inf(sys.z, "sys.z contains NaN or Inf: $(sys.z)")

    D_z = get_state_dim(typeof(sys))
    D_z = 4
    # TODO: assert D_z mod 2 ≡ 0
    #D_zh = div(D_z,2)
    D_zh = 2

    z, dz = sys.z[1:D_zh], sys.z[D_zh+1:end]
    z_new = z + sys.Δt * dz
    dz_new = dz + sys.Δt * ddzdt(sys, z[1], z[2], dz[1], dz[2], u)

    sys.z = vcat(z_new, dz_new)

    #update_params(sys)
    σ = 1e-3*ones(2)

    Q = [sys.Δt^3/3*σ[1]          0.0  sys.Δt^2/2*σ[1]          0.0;
                 0.0  sys.Δt^3/3*σ[2]          0.0  sys.Δt^2/2*σ[2];
         sys.Δt^2/2*σ[1]          0.0      sys.Δt*σ[1]          0.0;
                 0.0  sys.Δt^2/2*σ[2]          0.0      sys.Δt*σ[2]]
    sys.z = rand(MvNormal(sys.z, Q))
    return sys.z
end

function measure(sys::DynamicSystem)
    D_y = get_observation_dim(typeof(sys))
    z = sys.z[1:2]
    
    y = rand(MvNormal(z, sys.W))
    return y
end

# ----------------------------
# Abstract Interface Functions
# ----------------------------
get_eom_md(sys::DynamicSystem) = not_implemented_error(typeof(sys))
#ddzdt(sys::DynamicSystem, z::Vector{Float64}, dz::Vector{Float64}, u::Vector{Float64}) = not_implemented_error(typeof(sys))
inv_inertia_matrix(sys::DynamicSystem) = not_implemented_error(typeof(sys))
update_params(sys::DynamicSystem) = not_implemented_error(typeof(sys))

# ------------------------
# System Query Functions
# ------------------------

function get_state_dim(sys_type::Type{DPendulum})
    return DEFAULTS_SYS.DPendulum.D_z
end

function get_observation_dim(sys_type::Type{DPendulum})
    return DEFAULTS_SYS.DPendulum.D_y
end

function get_action_dim(sys_type::Type{DPendulum})
    return DEFAULTS_SYS.DPendulum.D_a
end

function get_control_dim(sys_type::Type{DPendulum})
    return DEFAULTS_SYS.DPendulum.D_u
end

function get_eom_md(sys::DPendulum)
    return md"""
    # System dynamics
    $$\begin{aligned}
    J_a = \frac{1}{3}m_1 l_1^2 + m_2 l_1^2 \, , \quad \\
    J_b = \frac{1}{3}m_2 l_2^2 \, , \quad \\
    J_x = \frac{1}{2}m_2 l_1 l_2 \, , \quad \\
    \mu_1 = (\frac{1}{2}m_1 + m_2)Gl_2 \, , \quad \\
    \mu_2 = \frac{1}{2}m_2 G l_2 \, . \\
    \underbrace{\begin{bmatrix} J_a & J_x \cos(\theta_1 - \theta_2) \\ J_x \cos(\theta_1 - \theta_2) & J_b \end{bmatrix}}_{M(\theta_1, \theta_2)}
    \begin{bmatrix} \ddot{\theta}_1 \\ \ddot{\theta}_2 \end{bmatrix} =
    \begin{bmatrix}
    - J_x \sin(\theta_1 - \theta_2)\dot{\theta}_2^2 - \mu_1 \sin \theta_1 + K_t (\theta_2 - \theta_1) \\
    J_x \sin(\theta_1 - \theta_2)\dot{\theta}_1^2 - \mu_2 \sin \theta_2 + K_t (\theta_2 - \theta_1)
    \end{bmatrix}
    \end{aligned}$$
    """
end

# ------------------------
# System Dynamics
# ------------------------

function inv_inertia_matrix(sys::DPendulum)
    z1, z2, _, _ = sys.z
    m1, m2 = sys.mass
    l1, l2 = sys.length

    # Components of the inertia matrix
    Ja = 1/3 * m1 * l1^2 + m2*l1^2
    Jb = 1/3 * m2 * l2^2
    Jx = 1/2 * m2 * l1 * l2

    z1mz2 = z1 - z2
    check_nan_or_inf(z1mz2, "z1 - z2: $z1 - $z2")

    # Inverse mass (inertia) matrix
    detM = Ja * Jb - (Jx * cos(z1mz2))^2
    invM = (1/detM) * [Jb -Jx * cos(z1mz2);-Jx * cos(z1mz2) Ja]
    return invM
end

end
