module MARXControllers

using Optim
using Distributions
using SpecialFunctions
using LinearAlgebra

export MARXController, update!, predictions, posterior_predictive, MPC, minimizeMPC, update_goals!


mutable struct MARXController
    """
    Model predictive controller based on a Multivariate Auto-Regressive eXogenous model.

    Parameters are inferred through Bayesian filtering and controls through minimizing expected free energy.
    """

    Dy              ::Integer
    Dx              ::Integer
    Du              ::Integer
    ybuffer         ::Matrix{Float64}
    ubuffer         ::Matrix{Float64}
    delay_inp       ::Integer
    delay_out       ::Integer

    M               ::Matrix{Float64}   # Coefficients mean matrix
    Λ               ::Matrix{Float64}   # Coefficients row-covariance
    Ω               ::Matrix{Float64}   # Precision scale matrix
    ν               ::Float64           # Precision degrees-of-freedom
    Υ               ::Matrix{Float64}   # Control prior precision matrix

    goal_prior      ::Distribution{Multivariate, Continuous}
    thorizon        ::Integer
    num_iters       ::Integer

    free_energy     ::Float64

    function MARXController(coefficients_mean_matrix,
                       coefficients_row_covariance,
                       precision_scale,
                       precision_degrees,
                       control_prior_precision,
                       goal_prior;
                       Dy::Integer=2,
                       Du::Integer=2,
                       delay_inp::Integer=1, 
                       delay_out::Integer=1, 
                       time_horizon::Integer=1,
                       num_iters::Integer=10)

        ybuffer = zeros(Dy,delay_out)
        ubuffer = zeros(Du,delay_inp)
        Dx = Du*delay_inp + Dy*delay_out

        free_energy = Inf

        return new(Dy,
                   Dx,
                   Du,
                   ybuffer,
                   ubuffer,
                   delay_inp,
                   delay_out,
                   coefficients_mean_matrix,
                   coefficients_row_covariance,
                   precision_scale,
                   precision_degrees,
                   control_prior_precision,
                   goal_prior,
                   time_horizon,
                   num_iters,
                   free_energy)
    end
end

function update!(agent::MARXController, y_k::Vector, u_k::Vector)

    # Short-hand
    M = agent.M
    Λ = agent.Λ
    Ω = agent.Ω
    ν = agent.ν

    # Update input buffer
    agent.ubuffer = backshift(agent.ubuffer, u_k)
    x_k = [agent.ubuffer[:]; agent.ybuffer[:]]

    # Auxiliary variables
    X = x_k*x_k'
    Ξ = (x_k*y_k' + Λ*M)

    # Update rules
    agent.ν = ν + 1
    agent.Λ = Λ + X
    agent.Ω = Ω + y_k*y_k' + M'*Λ*M - Ξ'*inv(Λ+X)*Ξ
    agent.M = inv(Λ+X)*Ξ

    # Update output buffer
    agent.ybuffer = backshift(agent.ybuffer, y_k)

    # Update performance metric
    agent.free_energy = -logevidence(agent, y_k, x_k)
end

function params(agent::MARXController)
    return agent.M, agent.U, agent.V, agent.ν
end

function logevidence(agent::MARXController, y,x)
    η, μ, Ψ = posterior_predictive(agent, x)
    return -1/2*(agent.Dy*log(η*π) -logdet(Ψ + 1e-6*I(agent.Dy)) - 2*logmultigamma(agent.Dy, (η+agent.Dy)/2) + 2*logmultigamma(agent.Dy, (η+agent.Dy-1)/2) + (η+agent.Dy)*log(1 + 1/η*(y-μ)'*Ψ*(y-μ)) )
end

function posterior_predictive(agent::MARXController, x_t)
    "Posterior predictive distribution is multivariate T-distributed."

    η_t = agent.ν - agent.Dy + 1
    μ_t = agent.M'*x_t
    # Ψ_t = (agent.ν-agent.Dy+1)*inv(agent.Ω)*inv(1 + x_t'*inv(agent.Λ + 1e-6*I(agent.Dy))*x_t)
    Ψ_t = (agent.ν-agent.Dy+1)*inv(agent.Ω)*inv(1 + x_t'*x_t)

    return η_t, μ_t, Ψ_t
end

function predictions(agent::MARXController, controls; time_horizon=1)
    
    m_y = zeros(agent.Dy,time_horizon)
    S_y = zeros(agent.Dy,agent.Dy,time_horizon)

    ybuffer = agent.ybuffer
    ubuffer = agent.ubuffer
    
    for t in 1:time_horizon
        
        # Update control buffer
        ubuffer = backshift(ubuffer, controls[:,t])
        x_t = [ubuffer[:]; ybuffer[:]]

        # Prediction
        η_t, μ_t, Ψ_t = posterior_predictive(agent, x_t)
        m_y[:,t] = μ_t
        S_y[:,:,t] = inv(Ψ_t) * η_t/(η_t - 2)
        
        # Update previous 
        ybuffer = backshift(ybuffer, m_y[:,t])
        
    end
    return m_y, S_y
end

function MPC(agent::MARXController, controls)
    "Model predictive control"

    ybuffer = agent.ybuffer
    ubuffer = agent.ubuffer

    m_star = mean(agent.goal_prior)

    J = 0
    for t in 1:agent.thorizon

        # Current control
        u_t = controls[(t-1)*agent.Du+1:t*agent.Du]
        
        # Update control buffer
        ubuffer = backshift(ubuffer, u_t)
        x_t = [ubuffer[:]; ybuffer[:]]

        # Calculate posterior predictive
        η_t, μ_t, Ψ_t = posterior_predictive(agent, x_t)

        # Calculate and accumulate EFE
        J += 1/2*(μ_t - m_star)'*(μ_t - m_star) + 1/2*u_t'*agent.Υ*u_t
        
        # Update previous 
        ybuffer = backshift(ybuffer, μ_t)        
    end
    return J
end

function minimizeMPC(agent::MARXController; u_0=nothing, time_limit=10, verbose=false, control_lims::Tuple=(-Inf,Inf))
    "Minimize MPC objective and return policy."

    if isnothing(u_0); u_0 = 1e-8*randn(agent.thorizon); end
    opts = Optim.Options(time_limit=time_limit, 
                         show_trace=verbose, 
                         allow_f_increases=true, 
                         g_tol=1e-12, 
                         show_every=10,
                         iterations=10_000)

    # Objective function
    C(u) = MPC(agent, u)
    gradC(J,u) = ForwardDiff.gradient!(J,C,u)

    # Constrained minimization procedure
    results = optimize(C, gradC, control_lims..., u_0, Fminbox(LBFGS()), opts)

    return Optim.minimizer(results)
end

function backshift(x::AbstractMatrix, a::Vector)
    "Shift elements rightwards and add element"
    return [a x[:,1:end-1]]
end

function backshift(x::AbstractVector, a::Number)
    "Shift elements down and add element"

    N = size(x,1)

    # Shift operator
    S = Tridiagonal(ones(N-1), zeros(N), zeros(N-1))

    # Basis vector
    e = [1.0; zeros(N-1)]

    return S*x + e*a
end

function update_goals!(x::AbstractVector, g::Distribution{Univariate, Continuous})
    "Move goals forward and add a final goal"
    circshift!(x,-1)
    x[end] = g
end

function multigamma(p,a)
    result = π^(p*(p-1)/4)
    for j = 1:p 
        result *= gamma(a + (1-j)/2)
    end
    return result
end

function logmultigamma(p,a)
    result = p*(p-1)/4*log(π)
    for j = 1:p 
        result += loggamma(a + (1-j)/2)
    end
    return result
end

function sqrtm(M::AbstractMatrix)
    "Square root of matrix"

    if size(M) == (2,2)
        "https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix"

        A,C,B,D = M

        # Determinant
        δ = A*D - B*C
        s = sqrt(δ)

        # Trace
        τ = A+D
        t = sqrt(τ + 2s)

        return 1/t*(M+s*Matrix{eltype(M)}(I,2,2))
    else
        "Babylonian method"

        Xk = Matrix{eltype(M)}(I,size(M))
        Xm = zeros(eltype(M), size(M))

        while sum(abs.(Xk[:] .- Xm[:])) > 1e-3
            Xm = Xk
            Xk = (Xm + M/Xm)/2.0
        end
        return Xk
    end
end

end
