
struct NUV_Box end

@node NUV_Box Stochastic [out, σ2a, σ2b, a, b,  γ]


@rule NUV_Box(:σ2a, Marginalisation) (q_out::PointMass, q_σ2b::PointMass, q_a::PointMass,q_b::PointMass, q_γ::PointMass, ) = begin
    return PointMass( 1 / mean(q_γ) * sqrt(abs2(mean(q_out) - mean(q_a)) + var(q_out))) 
end

@rule NUV_Box(:σ2b, Marginalisation) (q_out::PointMass, q_σ2a::PointMass, q_a::PointMass,q_b::PointMass, q_γ::PointMass, ) = begin
    return PointMass( 1 / mean(q_γ) * sqrt(abs2(mean(q_out) - mean(q_b)) + var(q_out)))
end