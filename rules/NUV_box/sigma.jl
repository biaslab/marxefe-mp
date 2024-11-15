
@rule NUV_Box(:σ2a, Marginalisation) (q_out::PointMass, 
                                      q_σ2b::PointMass, 
                                      q_a::PointMass,
                                      q_b::PointMass, 
                                      q_γ::PointMass, ) = begin
    return PointMass( 1 / mean(q_γ) * sqrt(abs2(mean(q_out) - mean(q_a)) + var(q_out))) 
end

@rule NUV_Box(:σ2a, Marginalisation) (q_out::UnivariateGaussianDistributionsFamily, 
                                      q_σ2b::PointMass, 
                                      q_a::PointMass,
                                      q_b::PointMass, 
                                      q_γ::PointMass, ) = begin
    return PointMass( 1 / mean(q_γ) * sqrt(abs2(mean(q_out) - mean(q_a)) + var(q_out))) 
end

@rule NUV_Box(:σ2a, Marginalisation) (q_out::ContinuousUnivariateLogPdf, 
                                      q_σ2b::PointMass, 
                                      q_a::PointMass,
                                      q_b::PointMass, 
                                      q_γ::PointMass, ) = begin

    q_out_box = ContinuousUnivariateLogPdf(Interval(mean(q_a)..mean(q_b)), q_out.logpdf)
    return PointMass( 1 / mean(q_γ) * sqrt(abs2(mean(q_out_box) - mean(q_a)) + var(q_out_box)))
end

@rule NUV_Box(:σ2b, Marginalisation) (q_out::PointMass, 
                                      q_σ2a::PointMass, 
                                      q_a::PointMass,
                                      q_b::PointMass, 
                                      q_γ::PointMass, ) = begin
    return PointMass( 1 / mean(q_γ) * sqrt(abs2(mean(q_out) - mean(q_b)) + var(q_out)))
end

@rule NUV_Box(:σ2b, Marginalisation) (q_out::UnivariateGaussianDistributionsFamily, 
                                      q_σ2a::PointMass, 
                                      q_a::PointMass,
                                      q_b::PointMass, 
                                      q_γ::PointMass, ) = begin
    return PointMass( 1 / mean(q_γ) * sqrt(abs2(mean(q_out) - mean(q_b)) + var(q_out)))
end

@rule NUV_Box(:σ2b, Marginalisation) (q_out::ContinuousUnivariateLogPdf, 
                                      q_σ2a::PointMass, 
                                      q_a::PointMass,
                                      q_b::PointMass, 
                                      q_γ::PointMass, ) = begin

    q_out_box = ContinuousUnivariateLogPdf(Interval(mean(q_a)..mean(q_b)), q_out.logpdf)
    return PointMass( 1 / mean(q_γ) * sqrt(abs2(mean(q_out_box) - mean(q_b)) + var(q_out_box)))
end