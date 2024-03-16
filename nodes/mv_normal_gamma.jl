
struct MvNormalGamma end
@node MvNormalGamma Stochastic [out, μ,Λ,α,β]


@rule MvNormalGamma(:out, Marginalisation) (q_μ::PointMass, q_Λ::PointMass, q_α::PointMass, q_β::PointMass) = begin
    return MvNormalGamma(q_μ, q_Λ, q_α, q_β)
end