import BayesBase: params

@rule ARXEFE(:out, Marginalisation) (q_outprev1::PointMass, 
                                     q_outprev2::PointMass, 
                                     q_in::PointMass, 
                                     q_inprev1::PointMass, 
                                     q_inprev2::PointMass, 
                                     m_ζ::MvNormalGamma) = begin

    μk,Λk,αk,βk = params(m_ζ)
    xk = [mean(q_outprev1), mean(q_outprev2), mean(q_in), mean(q_inprev1), mean(q_outprev2)]

    ν = 2αk
    μ = μk'*xk
    σ = sqrt(βk/αk*(xk'*inv(Λk)*xk + 1))

    return LocationScaleT(ν,μ,σ)
end

@rule ARXEFE(:out, Marginalisation) (q_outprev1::PointMass, 
                                     q_outprev2::PointMass, 
                                     m_in::UnivariateGaussianDistributionsFamily, 
                                     q_inprev1::PointMass, 
                                     q_inprev2::PointMass, 
                                     m_ζ::MvNormalGamma) = begin

    μk,Λk,αk,βk = params(m_ζ)
    xk = [mean(q_outprev1), mean(q_outprev2), mode(m_in), mean(q_inprev1), mean(q_outprev2)]

    ν = 2αk
    μ = μk'*xk
    σ = sqrt(βk/αk*(xk'*inv(Λk)*xk + 1))

    return LocationScaleT(ν,μ,σ)
end