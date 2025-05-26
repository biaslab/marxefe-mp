@rule MARX(:in, Marginalisation) (m_out::MvNormalMeanCovariance,
                                  q_outprev1::PointMass, 
                                  q_outprev2::PointMass, 
                                  q_inprev1::PointMass, 
                                  q_inprev2::PointMass,
                                  m_Φ::MatrixNormalWishart) = begin

    # Extract goal prior params
    m_star = mean(m_out)
    S_star = cov(m_out)

    Dy = length(m_star)
    M,Λ,Ω,ν = params(m_Φ)
                         
    function G(u)
    
        # Construct buffer vector
        x = [mean(q_outprev1); mean(q_outprev2); u; mean(q_inprev1); mean(q_inprev2)]

        # Parameters of multivariate location-scale T posterior predictive distribution
        η = ν - Dy + 1
        μ = M'*x
        Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)

        # Mutual information
        MI = -1/2*logdet(Σ)

        # Cross entropy
        CE = 1/2*( η/(η-2)*tr(inv(S_star)*Σ) + (μ-m_star)'*inv(S_star)*(μ-m_star) ) 

        return MI + CE
    end

    # Maximize 
    opts = Optim.Options(time_limit=10., 
                         show_trace=false, 
                         allow_f_increases=true, 
                         g_tol=1e-12, 
                         show_every=10,
                         iterations=1000)
    results = optimize(G, u_lims..., zeros(2), Fminbox(LBFGS()), opts, autodiff=:forward)

    # Return mode
    return MvNormalMeanCovariance(Optim.minimizer(results), 1e-8diagm(ones(Dy)))
end

@rule MARX(:in, Marginalisation) (m_Φ::MatrixNormalWishart,
                                  q_out::PointMass,
                                  q_outprev1::PointMass, 
                                  q_outprev2::PointMass, 
                                  q_inprev1::PointMass, 
                                  q_inprev2::PointMass,) = begin

    # Extract goal prior params
    m_star = mean(q_out)
    Dy = length(m_star)
    S_star = 1e-12*diagm(ones(Dy))
    
    M,Λ,Ω,ν = params(m_Φ)
                         
    function G(u)
    
        # Construct buffer vector
        x = [mean(q_outprev1); mean(q_outprev2); u; mean(q_inprev1); mean(q_inprev2)]

        # Parameters of multivariate location-scale T posterior predictive distribution
        η = ν - Dy + 1
        μ = M'*x
        Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)

        # Mutual information
        MI = 1/2*logdet(Σ)

        # Cross entropy
        CE = 1/2*( η/(η-2)*tr(inv(S_star)*Σ) + (μ-m_star)'*inv(S_star)*(μ-m_star) ) 

        return MI + CE
    end

    # Maximize 
    opts = Optim.Options(time_limit=10., 
                         show_trace=true, 
                         allow_f_increases=true, 
                         g_tol=1e-12, 
                         show_every=10,
                         iterations=1000)
    results = optimize(G, -10,10, zeros(2), Fminbox(LBFGS()), opts, autodiff=:forward)

    # Return mode
    return PointMass(Optim.minimizer(results))
end