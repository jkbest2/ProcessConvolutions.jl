module GaussianProcessConvolutions

using Base
using Distributions
using PDMats

import PDMats: dim

export
    # Convolution Kernels
    AbstractConvolutionKernel,
    SquaredExponentialKernel,
    # Basic type
    GaussianProcessConvolution,
    # Methods
    dim,		# get dimensionality of process or kernel
    unnorm_wt,	# Get unnormalized weights for a new location
    norm_wt,	# Get normalized weights for a new location
    nknot,	# Return number of knots
    predict,
    unorm_predict


# include convolution kernels
include("ConvolutionKernels.jl")

#----------------------------------------------------------------------------
# Basic type
type GaussianProcessConvolution
    knot_locs::Array
    knot_values::Array
    dim::Integer
    nknot::Integer
end
    # GaussianProcessConvolution(knot_locs::Array, knot_values::vector) =
    #     new(knot_locs, knot_values, size(knot_locs)[1], size(knot_locs)[2])
# end

nknot(GPC::GaussianProcessConvolution) = length(GPC.knot_values)

#------------------------------------------------------------------------------
# Convolution kernels
abstract AbstractConvolutionKernel

type SquaredExponentialKernel <: AbstractConvolutionKernel
    Σ::AbstractPDMat

    SquaredExponentialKernel(Σ::AbstractPDMat) = new(Σ)
    SquaredExponentialKernel(Σ::Vector) = new(PDiagMat(Σ))
    SquaredExponentialKernel(Σ::Array) = new(PDMat(Σ))
end

## Get dimensionality of kernel
dim(kern::AbstractConvolutionKernel) = kern.Σ.dim

## Normalize weights so they add to 1
function norm_wt(kern::AbstractConvolutionKernel, d::Array)
    un_wt = zeros(size(d, 2))
    un_wt = unnorm_wt(kern, d)
    un_wt ./ sum(un_wt)
end

#------------------------------------------------------------------------------
# Putting them together
function predict(GPC::GaussianProcessConvolution, kern::AbstractConvolutionKernel, new_loc::Array)

        nnew = size(new_loc, 2)
        new_val = zeros(nnew)

        for l in 1:nnew
                d = GPC.knot_locs' .- new_loc[:, l]
                new_val[l] = dot(norm_wt(kern, d), GPC.knot_values)
        end
        new_val
end

function unorm_predict(GPC::GaussianProcessConvolution, kern::AbstractConvolutionKernel, new_loc::Array)

        nnew = size(new_loc, 2)
        new_val = zeros(nnew)

        for l in 1:nnew
                d = GPC.knot_locs' .- new_loc[:, l]
                new_val[l] = dot(unnorm_wt(kern, d), GPC.knot_values)
        end
        new_val
end

end  # module
