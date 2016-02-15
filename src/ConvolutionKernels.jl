abstract AbstractConvolutionKernel

immutable SquaredExponentialKernel <: AbstractConvolutionKernel
    Σ::AbstractPDMat
    SquaredExponentialKernel(Σ::AbstractPDMat) = new(Σ)
    SquaredExponentialKernel(Σ::Vector) = new(PDiagMat(Σ))
    SquaredExponentialKernel(Σ::Array) = new(PDMat(Σ))
end

dim(kern::AbstractConvolutionKernel) = kern.Σ.dim

## Calculate weights
function conv_wt(kern::SquaredExponentialKernel, d::Array)
    if size(d, 1) != dim(kern)
        throw(DimensionMismatch)
    end
    exp(-0.5 * invquad(kern.Σ, d))
end

