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
    dim,		  # get dimensionality of process or kernel
    conv_wt,	# Get convolution weights for a new location
    nknot,	  # Return number of knots
    predict   # Give value of GP at new locations

# include convolution kernels
include("ConvolutionKernels.jl")

#----------------------------------------------------------------------------
# Basic type
type GaussianProcessConvolution
    knot_locs::Array
    knot_values::Array
    dim::Integer
    nknot::Integer
    GaussianProcessConvolution(knot_locs::Array, knot_values::Vector) =
      new(knot_locs,
          knot_values,
          size(knot_locs, 2),
          size(knot_locs, 1))
    GaussianProcessConvolution(knot_locs::Array, dist::UnivariateDistribution) =
      new(knot_locs,
          rand(dist, size(knot_locs, 1)),
          size(knot_locs, 2),
          size(knot_locs, 1))
    GaussianProcessConvolution(knot_locs::Array) =
      GaussianProcessConvolution(knot_locs, Normal(0, 1))
end

knot_locs(GPC::GaussianProcessConvolution) = GPC.knot_locs
knot_values(GPC::GaussianProcessConvolution) = GPC.knot_values
nknot(GPC::GaussianProcessConvolution) = GPC.nknot
dim(GPC::GaussianProcessConvolution) = GPC.dim

#------------------------------------------------------------------------------
# Putting them together
function predict(GPC::GaussianProcessConvolution,
                 kern::AbstractConvolutionKernel,
                 new_loc::Array)

    nnew = size(new_loc, 1)
    new_val = zeros(nnew)

    for l in 1:nnew
      d = GPC.knot_locs' .- new_loc'[:, l]
      new_val[l] = dot(conv_wt(kern, d), GPC.knot_values)
    end
    new_val
end

# Include efficient sample storage type
include("sampling.jl")
end  # module
