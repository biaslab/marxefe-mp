module buffer

using LinearAlgebra
using Distributions
using SpecialFunctions

export Buffer, dimensions

mutable struct Buffer
 
    D    ::Integer
    data ::Vector{Any}

    function Buffer(data::Vector)
        return new(data)
    end
end

function backshift(x::AbstractVector, a::Number)
    "Shift elements down and add element"

    N = size(x,1)

    # Shift operator
    S = Tridiagonal(ones(N-1), zeros(N), zeros(N-1))

    # Basis vector
    e = [1.0; zeros(N-1)]

    return S*x + e*a
end

function backshift(M::AbstractMatrix, a::Number)
    return diagm(backshift(diag(M), a))
end

end