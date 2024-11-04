
@rule NUV_Box(:out, Marginalisation) (q_σ2a::PointMass, 
                                      q_σ2b::PointMass, 
                                      q_a::PointMass, 
                                      q_b::PointMass, 
                                      q_γ::PointMass) = begin

    out_mean = (mean(q_a)*mean(q_σ2b) + mean(q_b)*mean(q_σ2a)) /(mean(q_σ2a) + mean(q_σ2b))
    out_var = mean(q_σ2a) * mean(q_σ2b) /(mean(q_σ2a) + mean(q_σ2b)) +1e-6
    
    return NormalMeanVariance(out_mean, out_var)

end