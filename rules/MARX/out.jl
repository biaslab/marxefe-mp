@rule MARX(:out, Marginalisation) (q_outprev1::Union{PointMass,AbstractMvNormal},
                                   q_outprev2::Union{PointMass,AbstractMvNormal}, 
                                   q_in::PointMass,
                                   q_inprev1::PointMass, 
                                   q_inprev2::PointMass,
                                   m_Φ::MatrixNormalWishart) = begin

    # Extract parameters 
    M,Λ,Ω,ν = params(m_Φ)

    # Construct buffer vector
    x = [mode(q_outprev1); mode(q_outprev2); mode(q_in); mode(q_inprev1); mode(q_inprev2)]

    # Parameters of multivariate location-scale T posterior predictive distribution
    η = ν - Dy + 1
    μ = M'*x
    Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)
      
    return MvLocationScaleT(η,μ,Σ)
end

@rule MARX(:out, Marginalisation) (q_outprev1::PointMass, 
                                   q_outprev2::PointMass, 
                                   m_in::unBoltzmann,
                                   q_inprev1::PointMass, 
                                   q_inprev2::PointMass,
                                   m_Φ::MatrixNormalWishart,) = begin 

    # Extract parameters 
    M,Λ,Ω,ν = params(m_Φ)

    # Construct buffer vector
    x = [mode(q_outprev1); mode(q_outprev2); mode(m_in, u_lims=u_lims); mode(q_inprev1); mode(q_inprev2)]

    # Parameters of multivariate location-scale T posterior predictive distribution
    η = ν - Dy + 1
    μ = M'*x
    Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)
    
    return MvLocationScaleT(η,μ,Σ)
end

@rule MARX(:out, Marginalisation) (m_outprev1::MvLocationScaleT, 
                                   q_outprev2::PointMass,
                                   m_in::AbstractMvNormal, 
                                   m_inprev1::unBoltzmann, 
                                   q_inprev2::PointMass,
                                   m_Φ::MatrixNormalWishart,) = begin 

    # Extract parameters 
    M,Λ,Ω,ν = params(m_Φ)

    # Construct buffer vector
    x = [mode(m_outprev1); mode(q_outprev2); mode(m_in); mode(m_inprev1, u_lims=u_lims); mode(q_inprev2)]

    # Parameters of multivariate location-scale T posterior predictive distribution
    η = ν - Dy + 1
    μ = M'*x
    Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)
    
    return MvLocationScaleT(η,μ,Σ)
end

@rule MARX(:outprev1, Marginalisation) (q_out::Union{AbstractMvNormal,MvLocationScaleT}, 
                                        q_outprev2::Union{PointMass,AbstractMvNormal}, 
                                        q_in::PointMass,
                                        q_inprev1::PointMass, 
                                        q_inprev2::PointMass,
                                        m_Φ::MatrixNormalWishart) = begin

    # Extract parameters 
    M,Λ,Ω,ν = params(m_Φ)                                            

    y = mode(q_out)
    x = [mode(q_outprev2); mode(q_in); mode(q_inprev1); mode(q_inprev2)]

    Dy = length(y)
    Dx = length(x)

    B1 = M[1:Dy,:]
    B_ = M[Dy+1:end,:]

    μ = inv(B1)'*(y - B_'*x)
    Σ = ν*inv(Ω)
      
    return MvNormalMeanCovariance(μ,Σ)
end

@rule MARX(:outprev1, Marginalisation) (m_out::Union{AbstractMvNormal,MvLocationScaleT}, 
                                        q_outprev2::Union{PointMass,AbstractMvNormal}, 
                                        m_in::Union{PointMass,AbstractMvNormal},
                                        m_inprev1::Union{PointMass,unBoltzmann}, 
                                        q_inprev2::PointMass,
                                        m_Φ::MatrixNormalWishart) = begin

    # Extract parameters 
    M,Λ,Ω,ν = params(m_Φ)                                            

    y = mode(m_out)
    x = [mode(q_outprev2); mode(m_in); mode(m_inprev1, u_lims=u_lims); mode(q_inprev2)]
    
    Dy = length(y)
    Dx = length(x)

    B1 = M[1:Dy,:]
    B_ = M[Dy+1:end,:]

    μ = inv(B1)'*(y - B_'*x)
    Σ = ν*inv(Ω)
      
    return MvNormalMeanCovariance(μ,Σ)
end
