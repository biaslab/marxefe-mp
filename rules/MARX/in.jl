@rule MARX(:in, Marginalisation) (m_out::MvNormalMeanCovariance,
                                  q_outprev1::Union{PointMass,AbstractMvNormal,MvLocationScaleT}, 
                                  q_outprev2::Union{PointMass,AbstractMvNormal,MvLocationScaleT},
                                  q_inprev1::PointMass, 
                                  q_inprev2::PointMass,
                                  m_Φ::MatrixNormalWishart) = begin

    m_star,S_star = mean_cov(m_out)
    M,Λ,Ω,ν = params(m_Φ)
    Dy = length(m_star)
                         
    function G(u)
    
        # Construct buffer vector
        x = [mode(q_outprev1); mode(q_outprev2); u; mode(q_inprev1); mode(q_inprev2)]

        # Parameters of multivariate location-scale T posterior predictive distribution
        η = ν - Dy + 1
        μ = M'*x
        Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)

        # Mutual information
        MI = -1/2*logdet(Σ)

        # Cross entropy
        CE = 1/2*( η/(η-2)*tr( S_star\(Σ + (μ-m_star)*(μ-m_star)' )) )

        return MI + CE
    end
    return unBoltzmann(G,Dy)
end

@rule MARX(:in, Marginalisation) (m_out::MvNormalMeanCovariance, 
                                  m_outprev1::MvLocationScaleT, 
                                  q_outprev2::PointMass,
                                  m_inprev1::unBoltzmann, 
                                  q_inprev2::PointMass,
                                  m_Φ::MatrixNormalWishart,) = begin 

    m_star,S_star = mean_cov(m_out)
    M,Λ,Ω,ν = params(m_Φ)
    Dy = length(m_star)
                            
    function G(u)

        # Construct buffer vector
        x = [mode(m_outprev1); mode(q_outprev2); u; mode(m_inprev1, u_lims=u_lims); mode(q_inprev2)]

        # Parameters of multivariate location-scale T posterior predictive distribution
        η = ν - Dy + 1
        μ = M'*x
        Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)

        # Mutual information
        MI = -1/2*logdet(Σ)

        # Cross entropy
        CE = 1/2*( η/(η-2)*tr( S_star\(Σ + (μ-m_star)*(μ-m_star)' )) )

        return MI + CE
    end
    return unBoltzmann(G,Dy)
end

@rule MARX(:inprev1, Marginalisation) (m_out::MvNormalMeanCovariance,
                                       q_outprev1::Union{PointMass,AbstractMvNormal,MvLocationScaleT}, 
                                       q_outprev2::Union{PointMass,AbstractMvNormal,MvLocationScaleT},
                                       q_in::unBoltzmann, 
                                       q_inprev2::PointMass,
                                       m_Φ::MatrixNormalWishart) = begin

    m_star,S_star = mean_cov(m_out)
    M,Λ,Ω,ν = params(m_Φ)
    Dy = length(m_star)
                         
    function G(u)
    
        # Construct buffer vector
        x = [mode(q_outprev1); mode(q_outprev2); mode(q_in, u_lims=u_lims); u; mode(q_inprev2)]

        # Parameters of multivariate location-scale T posterior predictive distribution
        η = ν - Dy + 1
        μ = M'*x
        Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)

        # Mutual information
        MI = -1/2*logdet(Σ)

        # Cross entropy
        CE = 1/2*( η/(η-2)*tr( S_star\(Σ + (μ-m_star)*(μ-m_star)' )) )

        return MI + CE
    end
    return unBoltzmann(G,Dy)
end

@rule MARX(:inprev1, Marginalisation) (m_out::MvNormalMeanCovariance, 
                                       m_outprev1::MvNormalMeanCovariance, 
                                       q_outprev2::PointMass, 
                                       m_in::MvNormalMeanPrecision,  
                                       q_inprev2::PointMass,
                                       m_Φ::MatrixNormalWishart,) = begin 

    m_star,S_star = mean_cov(m_out)
    M,Λ,Ω,ν = params(m_Φ)
    Dy = length(m_star)
                            
    function G(u)

        # Construct buffer vector
        x = [mode(m_outprev1); mode(q_outprev2); mode(m_in); u; mode(q_inprev2)]

        # Parameters of multivariate location-scale T posterior predictive distribution
        η = ν - Dy + 1
        μ = M'*x
        Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)

        # Mutual information
        MI = -1/2*logdet(Σ)

        # Cross entropy
        CE = 1/2*( η/(η-2)*tr( S_star\(Σ + (μ-m_star)*(μ-m_star)' )) )

        return MI + CE
    end
    return unBoltzmann(G,Dy)
end

@rule MARX(:inprev1, Marginalisation) (m_out::MvNormalMeanCovariance, 
                                       m_outprev1::MvLocationScaleT, 
                                       q_outprev2::PointMass, 
                                       m_in::MvNormalMeanPrecision, 
                                       q_inprev2::PointMass,
                                       m_Φ::MatrixNormalWishart, ) = begin 
    
    m_star,S_star = mean_cov(m_out)
    M,Λ,Ω,ν = params(m_Φ)
    Dy = length(m_star)
                            
    function G(u)

        # Construct buffer vector
        x = [mode(m_outprev1); mode(q_outprev2); mode(m_in); u; mode(q_inprev2)]

        # Parameters of multivariate location-scale T posterior predictive distribution
        η = ν - Dy + 1
        μ = M'*x
        Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)

        # Mutual information
        MI = -1/2*logdet(Σ)

        # Cross entropy
        CE = 1/2*( η/(η-2)*tr( S_star\(Σ + (μ-m_star)*(μ-m_star)' )) )

        return MI + CE
    end
    return unBoltzmann(G,Dy)
end