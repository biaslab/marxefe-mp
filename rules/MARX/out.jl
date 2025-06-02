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

@rule MARX(:out, Marginalisation) (m_outprev1::Union{AbstractMvNormal,MvLocationScaleT}, 
                                   m_outprev2::Union{AbstractMvNormal,MvLocationScaleT},
                                   m_in::Union{AbstractMvNormal,MvLocationScaleT}, 
                                   m_inprev1::Union{AbstractMvNormal,MvLocationScaleT}, 
                                   m_inprev2::Union{AbstractMvNormal,MvLocationScaleT},
                                   m_Φ::MatrixNormalWishart,) = begin 

    # Extract parameters 
    M,Λ,Ω,ν = params(m_Φ)

    # Construct buffer vector
    x = [mode(m_outprev1); mode(m_outprev2); mode(m_in); mode(m_inprev1); mode(m_inprev2)]

    # Parameters of multivariate location-scale T posterior predictive distribution
    η = ν - Dy + 1
    μ = M'*x
    Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)
    
    return MvLocationScaleT(η,μ,Σ)
end

@rule MARX(:out, Marginalisation) (m_outprev1::AbstractMvNormal, 
                                   q_outprev2::PointMass, 
                                   m_in::unBoltzmann, 
                                   m_inprev1::unBoltzmann, 
                                   q_inprev2::PointMass,
                                   m_Φ::MatrixNormalWishart, ) = begin 
    # Extract parameters 
    M,Λ,Ω,ν = params(m_Φ)

    # Construct buffer vector
    x = [mode(m_outprev1); mode(q_outprev2); mode(m_in, u_lims=u_lims); mode(m_inprev1, u_lims=u_lims); mode(q_inprev2)]

    # Parameters of multivariate location-scale T posterior predictive distribution
    η = ν - Dy + 1
    μ = M'*x
    Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)
    
    return MvLocationScaleT(η,μ,Σ)
end

@rule MARX(:out, Marginalisation) (m_outprev1::MvLocationScaleT, 
                                   m_outprev2::AbstractMvNormal, 
                                   m_in::AbstractMvNormal, 
                                   m_inprev1::unBoltzmann, 
                                   m_inprev2::unBoltzmann, 
                                   m_Φ::MatrixNormalWishart, ) = begin 
    # Extract parameters 
    M,Λ,Ω,ν = params(m_Φ)

    # Construct buffer vector
    x = [mode(m_outprev1); mode(m_outprev2); mode(m_in); mode(m_inprev1, u_lims=u_lims); mode(m_inprev2, u_lims=u_lims)]

    # Parameters of multivariate location-scale T posterior predictive distribution
    η = ν - Dy + 1
    μ = M'*x
    Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)

    return MvLocationScaleT(η,μ,Σ)
end

@rule MARX(:out, Marginalisation) (q_outprev1::Union{PointMass,AbstractMvNormal}, 
                                   q_outprev2::Union{PointMass,AbstractMvNormal}, 
                                   q_in::Union{PointMass,AbstractMvNormal}, 
                                   q_inprev1::Union{PointMass,AbstractMvNormal}, 
                                   q_inprev2::Union{PointMass,AbstractMvNormal}, 
                                   q_Φ::MatrixNormalWishart, ) = begin 

    # Extract parameters 
    M,Λ,Ω,ν = params(q_Φ)

    # Construct buffer vector
    x = [mode(q_outprev1); mode(q_outprev2); mode(q_in); mode(q_inprev1); mode(q_inprev2)]

    # Parameters of multivariate location-scale T posterior predictive distribution
    η = ν - Dy + 1
    μ = M'*x
    Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)

    return MvLocationScaleT(η,μ,Σ)
end

@rule MARX(:out, Marginalisation) (q_outprev1::Union{PointMass,AbstractMvNormal}, 
                                   q_outprev2::Union{PointMass,AbstractMvNormal}, 
                                   q_in::unBoltzmann, 
                                   q_inprev1::Union{PointMass,AbstractMvNormal}, 
                                   q_inprev2::Union{PointMass,AbstractMvNormal}, 
                                   q_Φ::MatrixNormalWishart, ) = begin 

    # Extract parameters 
    M,Λ,Ω,ν = params(q_Φ)

    # Construct buffer vector
    x = [mode(q_outprev1); mode(q_outprev2); mode(q_in, u_lims=u_lims); mode(q_inprev1); mode(q_inprev2)]

    # Parameters of multivariate location-scale T posterior predictive distribution
    η = ν - Dy + 1
    μ = M'*x
    Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)

    return MvLocationScaleT(η,μ,Σ)
end

@rule MARX(:out, Marginalisation) (q_outprev1::Union{PointMass,AbstractMvNormal}, 
                                   q_outprev2::Union{PointMass,AbstractMvNormal}, 
                                   q_in::unBoltzmann, 
                                   q_inprev1::unBoltzmann, 
                                   q_inprev2::Union{PointMass,AbstractMvNormal}, 
                                   q_Φ::MatrixNormalWishart, ) = begin 

    # Extract parameters 
    M,Λ,Ω,ν = params(q_Φ)

    # Construct buffer vector
    x = [mode(q_outprev1); mode(q_outprev2); mode(q_in, u_lims=u_lims); mode(q_inprev1, u_lims=u_lims); mode(q_inprev2)]

    # Parameters of multivariate location-scale T posterior predictive distribution
    η = ν - Dy + 1
    μ = M'*x
    Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)

    return MvLocationScaleT(η,μ,Σ)
end

@rule MARX(:out, Marginalisation) (q_outprev1::Union{PointMass,AbstractMvNormal}, 
                                   q_outprev2::Union{PointMass,AbstractMvNormal}, 
                                   q_in::unBoltzmann, 
                                   q_inprev1::unBoltzmann, 
                                   q_inprev2::unBoltzmann, 
                                   q_Φ::MatrixNormalWishart, ) = begin 

    # Extract parameters 
    M,Λ,Ω,ν = params(q_Φ)

    # Construct buffer vector
    x = [mode(q_outprev1); mode(q_outprev2); mode(q_in, u_lims=u_lims); mode(q_inprev1, u_lims=u_lims); mode(q_inprev2, u_lims=u_lims)]

    # Parameters of multivariate location-scale T posterior predictive distribution
    η = ν - Dy + 1
    μ = M'*x
    Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)

    return MvLocationScaleT(η,μ,Σ)
end

@rule MARX(:outprev1, Marginalisation) (q_out::Union{AbstractMvNormal,MvLocationScaleT}, 
                                        q_outprev2::Union{PointMass,AbstractMvNormal}, 
                                        q_in::Union{PointMass,AbstractMvNormal},
                                        q_inprev1::Union{PointMass,AbstractMvNormal}, 
                                        q_inprev2::Union{PointMass,AbstractMvNormal},
                                        m_Φ::MatrixNormalWishart) = begin
 
    M,Λ,Ω,ν = params(m_Φ)                                            

    m_star,S_star = mean_cov(q_out)
    x = [mode(q_outprev2); mode(q_in); mode(q_inprev1); mode(q_inprev2)]

    Dy = length(m_star)
    Dx = length(x)

    B1  = M[1:Dy,:]
    iB1 = inv(B1 + diagm(ones(Dy)))
    B_  = M[Dy+1:end,:]

    μ = iB1'*(m_star - B_'*x)
    Σ = ν*Ω

    # @info "iB1 = ", iB1
    # @info "μ = ", μ
    # @info "Σ = ", Σ
      
    return MvNormalMeanCovariance(μ,Σ)
end

@rule MARX(:outprev1, Marginalisation) (q_out::Union{PointMass,AbstractMvNormal}, 
                                        q_outprev2::Union{PointMass,AbstractMvNormal}, 
                                        q_in::unBoltzmann, 
                                        q_inprev1::Union{PointMass,AbstractMvNormal}, 
                                        q_inprev2::Union{PointMass,AbstractMvNormal}, 
                                        q_Φ::MatrixNormalWishart, ) = begin 

    M,Λ,Ω,ν = params(q_Φ)                                            

    y = mode(q_out)
    x = [mode(q_outprev2); mode(q_in, u_lims=u_lims); mode(q_inprev1); mode(q_inprev2)]

    Dy = length(y)
    Dx = length(x)

    B1  = M[1:Dy,:]
    iB1 = inv(B1 + diagm(ones(Dy)))
    B_  = M[Dy+1:end,:]

    μ = iB1'*(y - B_'*x)
    Σ = ν*Ω

    # @info "iB1 = ", iB1
    # @info "μ = ", μ
    # @info "Σ = ", Σ
      
    return MvNormalMeanCovariance(μ,Σ)
end

@rule MARX(:outprev1, Marginalisation) (q_out::Union{PointMass,AbstractMvNormal}, 
                                        q_outprev2::Union{PointMass,AbstractMvNormal}, 
                                        q_in::unBoltzmann, 
                                        q_inprev1::unBoltzmann, 
                                        q_inprev2::Union{PointMass,AbstractMvNormal}, 
                                        q_Φ::MatrixNormalWishart, ) = begin 

    M,Λ,Ω,ν = params(q_Φ)                                            

    y = mode(q_out)
    x = [mode(q_outprev2); mode(q_in, u_lims=u_lims); mode(q_inprev1, u_lims=u_lims); mode(q_inprev2)]

    Dy = length(y)
    Dx = length(x)

    B1  = M[1:Dy,:]
    iB1 = inv(B1 + diagm(ones(Dy)))
    B_  = M[Dy+1:end,:]

    μ = iB1'*(y - B_'*x)
    Σ = ν*Ω

    # @info "iB1 = ", iB1
    # @info "μ = ", μ
    # @info "Σ = ", Σ
      
    return MvNormalMeanCovariance(μ,Σ)
end

@rule MARX(:outprev1, Marginalisation) (q_out::Union{PointMass,AbstractMvNormal}, 
                                        q_outprev2::Union{PointMass,AbstractMvNormal}, 
                                        q_in::unBoltzmann, 
                                        q_inprev1::unBoltzmann, 
                                        q_inprev2::unBoltzmann, 
                                        q_Φ::MatrixNormalWishart, ) = begin 

    M,Λ,Ω,ν = params(q_Φ)                                            

    y = mode(q_out)
    x = [mode(q_outprev2); mode(q_in, u_lims=u_lims); mode(q_inprev1, u_lims=u_lims); mode(q_inprev2, u_lims=u_lims)]

    Dy = length(m_star)
    Dx = length(x)

    B_  = M[Dy+1:end,:]
    B1  = M[1:Dy,:]
    iB1 = inv(B1 + diagm(ones(Dy)))

    μ = iB1'*(y - B_'*x)
    Σ = ν*Ω

    # @info "iB1 = ", iB1
    # @info "μ = ", μ
    # @info "Σ = ", Σ
      
    return MvNormalMeanCovariance(μ,Σ)
end

@rule MARX(:outprev1, Marginalisation) (m_out::Union{AbstractMvNormal,MvLocationScaleT}, 
                                        q_outprev2::Union{PointMass,AbstractMvNormal}, 
                                        m_in::Union{PointMass,AbstractMvNormal},
                                        m_inprev1::unBoltzmann, 
                                        q_inprev2::PointMass,
                                        m_Φ::MatrixNormalWishart) = begin

    M,Λ,Ω,ν = params(m_Φ)                                            

    m_star,S_star = mean_cov(m_out)
    x = [mode(q_outprev2); mode(m_in); mode(m_inprev1, u_lims=u_lims); mode(q_inprev2)]
    
    Dy = length(m_star)
    Dx = length(x)

    B1  = M[1:Dy,:]
    iB1 = inv(B1 + diagm(ones(Dy)))
    B_  = M[Dy+1:end,:]

    μ = iB1'*(m_star - B_'*x)
    Σ = ν*Ω

    # @info "iB1 = ", iB1
    # @info "μ = ", μ
    # @info "Σ = ", Σ
      
    return MvNormalMeanCovariance(μ,Σ)
end

@rule MARX(:outprev1, Marginalisation) (m_out::AbstractMvNormal, 
                                        m_outprev2::AbstractMvNormal, 
                                        m_in::AbstractMvNormal, 
                                        m_inprev1::AbstractMvNormal, 
                                        m_inprev2::AbstractMvNormal, 
                                        m_Φ::MatrixNormalWishart, ) = begin 
    
    M,Λ,Ω,ν = params(m_Φ)                                            

    y = mode(m_out)
    x = [mode(m_outprev2); mode(m_in); mode(m_inprev1); mode(m_inprev2)]
    
    Dy = length(y)
    Dx = length(x)

    B1  = M[1:Dy,:]
    iB1 = inv(B1 + diagm(ones(Dy)))
    B_  = M[Dy+1:end,:]

    μ = iB1'*(y - B_'*x)
    Σ = ν*Ω

    # @info "iB1 = ", iB1
    # @info "μ = ", μ
    # @info "Σ = ", Σ
      
    return MvNormalMeanCovariance(μ,Σ)
end

@rule MARX(:outprev1, Marginalisation) (m_out::MvNormalMeanCovariance, 
                                        q_outprev2::PointMass, 
                                        m_in::unBoltzmann, 
                                        m_inprev1::unBoltzmann, 
                                        q_inprev2::PointMass,
                                        m_Φ::MatrixNormalWishart,) = begin 
    
    M,Λ,Ω,ν = params(m_Φ)                                            

    y = mode(m_out)
    x = [mode(q_outprev2); mode(m_in, u_lims=u_lims); mode(m_inprev1, u_lims=u_lims); mode(q_inprev2)]
    
    Dy = length(y)
    Dx = length(x)

    B1  = M[1:Dy,:]
    iB1 = inv(B1 + diagm(ones(Dy)))
    B_  = M[Dy+1:end,:]

    μ = iB1'*(y - B_'*x)
    Σ = ν*Ω

    # @info "iB1 = ", iB1
    # @info "μ = ", μ
    # @info "Σ = ", Σ
      
    return MvNormalMeanCovariance(μ,Σ)
end

@rule MARX(:outprev1, Marginalisation) (m_out::AbstractMvNormal, 
                                        m_outprev2::AbstractMvNormal, 
                                        m_in::AbstractMvNormal, 
                                        m_inprev1::unBoltzmann, 
                                        m_inprev2::unBoltzmann, 
                                        m_Φ::MatrixNormalWishart, ) = begin 

    M,Λ,Ω,ν = params(m_Φ)                                            

    y = mode(m_out)
    x = [mode(m_outprev2); mode(m_in); mode(m_inprev1, u_lims=u_lims); mode(m_inprev2, u_lims=u_lims)]
    
    Dy = length(y)
    Dx = length(x)

    B1  = M[1:Dy,:]
    iB1 = inv(B1 + diagm(ones(Dy)))
    B_  = M[Dy+1:end,:]

    μ = iB1'*(y - B_'*x)
    Σ = ν*Ω

    # @info "iB1 = ", iB1
    # @info "μ = ", μ
    # @info "Σ = ", Σ
      
    return MvNormalMeanCovariance(μ,Σ)      
end


@rule MARX(:outprev2, Marginalisation) (m_out::AbstractMvNormal, 
                                        m_outprev1::AbstractMvNormal, 
                                        m_in::AbstractMvNormal, 
                                        m_inprev1::AbstractMvNormal, 
                                        m_inprev2::AbstractMvNormal, 
                                        m_Φ::MatrixNormalWishart, ) = begin 
    
    # M,Λ,Ω,ν = params(m_Φ)                                            

    # y_k = mode(m_out)
    # y_kmin1 = mode(m_outprev1)                                    
    # x = [mode(m_in); mode(m_inprev1); mode(m_inprev2)]
    
    # Dy = length(y_k)
    # Dx = length(x)

    # B1 = M[1:Dy,:]
    # B2 = M[Dy+1:2Dy,:]
    # B_ = M[2Dy+1:end,:]

    # μ = inv(B2)'*(y_k - B1'*y_kmin1 - B_'*x)
    # Σ = ν*inv(Ω)
      
    # return MvNormalMeanCovariance(μ,Σ)
    return Uninformative()
end

@rule MARX(:outprev2, Marginalisation) (m_out::AbstractMvNormal, 
                                        m_outprev1::AbstractMvNormal, 
                                        m_in::AbstractMvNormal, 
                                        m_inprev1::unBoltzmann, 
                                        m_inprev2::unBoltzmann, 
                                        m_Φ::MatrixNormalWishart, ) = begin 
    
    # M,Λ,Ω,ν = params(m_Φ)                                            

    # y_k = mode(m_out)
    # y_kmin1 = mode(m_outprev1)                                    
    # x = [mode(m_in); mode(m_inprev1, u_lims=u_lims); mode(m_inprev2, u_lims=u_lims)]
    
    # Dy = length(y_k)
    # Dx = length(x)

    # B1 = M[1:Dy,:]
    # B2 = M[Dy+1:2Dy,:]
    # B_ = M[2Dy+1:end,:]

    # μ = inv(B2 + 1e-3*diagm(ones(Dy)))'*(y_k - B1'*y_kmin1 - B_'*x)
    # Σ = ν*inv(Ω)
      
    # return MvNormalMeanCovariance(μ,Σ)
    return Uninformative()
end

@rule MARX(:outprev2, Marginalisation) (q_out::AbstractMvNormal, 
                                        q_outprev1::Union{PointMass,AbstractMvNormal}, 
                                        q_in::unBoltzmann, 
                                        q_inprev1::unBoltzmann, 
                                        q_inprev2::Union{PointMass,AbstractMvNormal}, 
                                        q_Φ::MatrixNormalWishart, ) = begin 
   
#    M,Λ,Ω,ν = params(q_Φ)                                            

#    y_k = mode(q_out)
#    y_kmin1 = mode(q_outprev1)                                    
#    x = [mode(q_in, u_lims=u_lims); mode(q_inprev1, u_lims=u_lims); mode(q_inprev2)]
   
#    Dy = length(y_k)
#    Dx = length(x)

#    B1 = M[1:Dy,:]
#    B2 = M[Dy+1:2Dy,:]
#    B_ = M[2Dy+1:end,:]

#    μ = inv(B2 + 1e-3*diagm(ones(Dy)))'*(y_k - B1'*y_kmin1 - B_'*x)
#    Σ = ν*inv(Ω)
     
#    return MvNormalMeanCovariance(μ,Σ)
    return Uninformative()
end

@rule MARX(:outprev2, Marginalisation) (q_out::AbstractMvNormal, 
                                        q_outprev1::AbstractMvNormal, 
                                        q_in::unBoltzmann, 
                                        q_inprev1::unBoltzmann, 
                                        q_inprev2::unBoltzmann, 
                                        q_Φ::MatrixNormalWishart, ) = begin 

    # M,Λ,Ω,ν = params(q_Φ)                                            

    # y_k = mode(q_out)
    # y_kmin1 = mode(q_outprev1)                                    
    # x = [mode(q_in, u_lims=u_lims); mode(q_inprev1, u_lims=u_lims); mode(q_inprev2, u_lims=u_lims)]
    
    # Dy = length(y_k)
    # Dx = length(x)

    # B1 = M[1:Dy,:]
    # B2 = M[Dy+1:2Dy,:]
    # B_ = M[2Dy+1:end,:]

    # μ = inv(B2 + 1e-3*diagm(ones(Dy)))'*(y_k - B1'*y_kmin1 - B_'*x)
    # Σ = ν*inv(Ω)
      
    # return MvNormalMeanCovariance(μ,Σ)
    return Uninformative()
end
