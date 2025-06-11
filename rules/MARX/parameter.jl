
@rule MARX(:Φ, Marginalisation) (q_out::PointMass,
                                 q_outprev1::PointMass, 
                                 q_outprev2::PointMass, 
                                 q_in::PointMass, 
                                 q_inprev1::PointMass, 
                                 q_inprev2::PointMass) = begin

    y_k = mean(q_out)                                
    x_k = [mean(q_outprev1); mean(q_outprev2); mean(q_in); mean(q_inprev1); mean(q_inprev2)]

    Dy = length(y_k)
    Dx = length(x_k)

    M_ = inv(x_k*x_k' + diagm(1e-12*ones(Dx)))*(x_k*y_k')
    Λ_ = x_k*x_k' + diagm(1e-12*ones(Dx))
    Ω_ = zeros(Dy,Dy)
    ν_ = 2 - Dx + Dy

    return MatrixNormalWishart(M_, Λ_, Ω_, ν_)
end

@rule MARX(:Φ, Marginalisation) (m_out::AbstractMvNormal,
                                 m_outprev1::Union{PointMass,AbstractMvNormal,MvLocationScaleT},
                                 q_outprev2::PointMass,
                                 m_in::Union{PointMass,AbstractMvNormal,unBoltzmann},
                                 m_inprev1::Union{PointMass,AbstractMvNormal,unBoltzmann}, 
                                 q_inprev2::PointMass) = begin

    return Uninformative()
end

@rule MARX(:Φ, Marginalisation) (m_out::AbstractMvNormal,
                                 m_outprev1::Union{PointMass,AbstractMvNormal,MvLocationScaleT},
                                 m_outprev2::Union{PointMass,AbstractMvNormal},
                                 m_in::AbstractMvNormal, 
                                 m_inprev1::Union{PointMass,AbstractMvNormal,unBoltzmann}, 
                                 m_inprev2::Union{PointMass,AbstractMvNormal,unBoltzmann}) = begin

    return Uninformative()
end

@rule MARX(:Φ, Marginalisation) (m_out::AbstractMvNormal, 
                                 q_outprev1::PointMass, 
                                 q_outprev2::PointMass, 
                                 m_in::Union{PointMass,AbstractMvNormal,unBoltzmann}, 
                                 q_inprev1::PointMass, 
                                 q_inprev2::PointMass, ) = begin 
    return Uninformative()
end

@rule MARX(:Φ, Marginalisation) (q_out::AbstractMvNormal, 
                                 q_outprev1::Union{PointMass,AbstractMvNormal},
                                 q_outprev2::Union{PointMass,AbstractMvNormal}, 
                                 q_in::Union{PointMass,AbstractMvNormal,unBoltzmann}, 
                                 q_inprev1::Union{PointMass,AbstractMvNormal,unBoltzmann}, 
                                 q_inprev2::Union{PointMass,AbstractMvNormal,unBoltzmann}, ) = begin 
    return Uninformative()
end