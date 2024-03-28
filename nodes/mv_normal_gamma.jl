
@node MvNormalGamma Stochastic [out, μ,Λ,α,β]

@rule MvNormalGamma(:out, Marginalisation) (q_μ::Any, q_Λ::Any, q_α::Any, q_β::Any,) = begin
    m_μ = mean(q_μ)
    m_Λ = mean(q_Λ)
    m_α = mean(q_α)
    m_β = mean(q_β)
    return MvNormalGamma(m_μ, m_Λ, m_α, m_β)
end