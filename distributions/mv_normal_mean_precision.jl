import Distributions: logpdf
import BayesBase
import ExponentialFamily

function logpdf(dist::MvNormalMeanCovariance, x::AbstractVector)
    d = ndims(dist)
    m,S = mean_cov(dist)
    return -d/2*(2*π) - 1/2*logdet(S) - 1/2*(x - m)'*inv(S)*(x-m)
end