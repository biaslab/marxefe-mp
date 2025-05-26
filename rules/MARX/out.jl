@rule MARX(:out, Marginalisation) (q_outprev1::PointMass, 
                                   q_outprev2::PointMass, 
                                   q_in::PointMass,
                                   q_inprev1::PointMass, 
                                   q_inprev2::PointMass,
                                   m_Φ::MatrixNormalWishart) = begin

    # Construct buffer vector
    x = [mean(q_outprev1); mean(q_outprev2); mean(q_in); mean(q_inprev1); mean(q_inprev2)]

    # Parameters of multivariate location-scale T posterior predictive distribution
    η = ν - Dy + 1
    μ = M'*x
    Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)
      
    return MvLocationScaleT(η,μ,Σ)
end

@rule MARX(:out, Marginalisation) (q_outprev1::PointMass, 
                                   q_outprev2::PointMass, 
                                   m_in::MvNormalMeanPrecision,
                                   q_inprev1::PointMass, 
                                   q_inprev2::PointMass,
                                   m_Φ::MatrixNormalWishart) = begin
      
    return Uninformative()
end