VERSION >= v"0.4" && __precompile__()

module ProcessConvolutions

using Base
using Distributions
using PDMats

import PDMats: dim

export
    # Convolution Kernels
    AbstractConvolutionKernel,
    GaussianKernel,
    SquaredExponentialKernel,
    # Basic type
    ProcessConvolution,
    GaussianProcessSample,
    # Methods
    dim,		    # get dimensionality of process or kernel
    conv_wt,	    # Get convolution weights for a new location
    knot_wt,        # Return matrix of conv weights for new locations
    nknot,	        # Return number of knots
    predict        # Give value of GP at new locations

# include convolution kernels
include("ConvolutionKernels.jl")

#----------------------------------------------------------------------------
# Basic type
abstract type AbstractProcess <: Any end

immutable ProcessConvolution{F <: AbstractFloat} <: AbstractProcess
    knot_locs::Array{F, 2}
    knot_values::Vector{F}
    dim::Int
    nknot::Int

    function ProcessConvolution(knot_locs, knot_values)
      new(knot_locs,
          knot_values,
          size(knot_locs, 2),
          size(knot_locs, 1))
    end
end

function ProcessConvolution{F <: AbstractFloat}(knot_locs::Array{F, 2},
                                                dist::UnivariateDistribution)
     ProcessConvolution(knot_locs,
                        rand(dist, size(knot_locs, 1)))
end

ProcessConvolution{F <: AbstractFloat}(knot_locs::AbstractArray) =
    ProcessConvolution(knot_locs, Normal(0, 1))

knot_locs(pc::ProcessConvolution) = pc.knot_locs
knot_values(pc::ProcessConvolution) = pc.knot_values
nknot(pc::ProcessConvolution) = pc.nknot
dim(pc::ProcessConvolution) = pc.dim

immutable ZeroProcess <: AbstractProcess end

immutable ConstantProcess{F <: AbstractFloat} <: AbstractProcess
    v::F
end

#-----------------------------------------------------------------------------
# abstract PredictiveProcessConvolution
#
# type ContinuousPredictivePC{F <: AbstractFloat} <: PredictiveProcessConvolution
#     ProcConv::AbstractProcess
#     ConvKern::AbstractConvolutionKernel
#     PredLocs::Array{F}
#     KnotWt::Array{F}
#     Transform::Function
# end
#
# type DiscretePredictivePC{T <: Any} <: PredictiveProcessConvolution
#     KnotLocs::Array{F}
#     PredLocs::Array{Float}
#     Process::Array{T}
#     KnotValue::Dict{T, Array{Float}}
#     ConvKernel::Dict{T, AbstractConvolutionKernel}
#     KnotWt::Dict{AbstractConvolutionKernel, Array{Float}}
# end
#
# # Outer constructor for simplest case: no knot values specified, one
# # shared kernel for all processes.
# function DiscretePredictivePC(knotlocs::Array{Float},
#                               predlocs::Array{Float},
#                               processlist::Vector{T},
#                               kernel::AbstractConvolutionKernel)
#     nkts = size(knotlocs, 1)
#     npred = size(predlocs, 1)
#     nproc = length(processlist)
#
#     kern = Dict{T, AbstractConvolutionKernel}()
#     for proc in processlist
#         kern[proc] = kernel
#     end
#
#     kv = Dict{T, Vector{Float}}()
#     for proc in processlist
#         kv[proc] = randn(nkts)
#     end
#
#     kw = Dict{AbstractConvolutionKernel,
#               Array{Float}}(kernel => knot_wt(knotlocs, kernel, predlocs))
#
#     DiscretePredictivePC(knotlocs,
#                          predlocs,
#                          processlist,
#                          kv,
#                          kw)
# end
#
# # Outer constructor for different kernel case: no knot values specified, one
# # shared kernel for all processes.
# function DiscretePredictivePC(knotlocs::Array{Float},
#                               predlocs::Array{Float},
#                               kernel::Dict{T, AbstractConvolutionKernel})
#
#     processlist = collect(keys(kernel))
#     nkts = size(knotlocs, 1)
#     npred = size(predlocs, 1)
#     nproc = length(processlist)
#
#     kv = Dict{T, Vector{Float}}()
#     for proc in processlist
#         kv[proc] = randn(nkts)
#     end
#
#     kw = Dict{AbstractConvolutionKernel,
#               Array{Float64, 2}}()
#     for kern in unique(values(kernel))
#         kw[kern] = knot_wt(knotlocs, kern, predlocs)
#     end
#
#     DiscretePredictivePC(knotlocs,
#                          predlocs,
#                          processlist,
#                          kv,
#                          kw)
# end
#
#------------------------------------------------------------------------------
# Putting them together
function predict(pc::ProcessConvolution,
                 kern::AbstractConvolutionKernel,
                 new_loc::AbstractArray)

    nnew = size(new_loc, 1)
    new_val = zeros(nnew)

    for l in 1:nnew
      d = pc.knot_locs' .- new_loc'[:, l]
      new_val[l] = dot(conv_wt(kern, d), pc.knot_values)
    end
    new_val
end

function predict(pc::ProcessConvolution,
                 knot_wt::AbstractArray)
    knot_wt * pc.knot_values
end

## Calculate knot weight matrix
function knot_wt(pc::ProcessConvolution,
                 kern::AbstractConvolutionKernel,
                 new_locs::AbstractArray)
    nloc = size(new_locs, 1)
    nk = nknot(pc)

    k_wt = Array{Float64, 2}(nloc, nk)
    for l in 1:nloc
        k_wt[l, :] = conv_wt(kern, knot_locs(pc)' .- new_locs[l, :]')'
    end
    k_wt
end

function knot_wt(knot_locs::Array,
                 kern::AbstractConvolutionKernel,
                 new_locs::Array)
    nloc = size(new_locs, 1)
    nk = size(knot_locs, 1)

    k_wt = Array{Float64, 2}(nloc, nk)
    for l in 1:nloc
        k_wt[l, :] = conv_wt(kern, knot_locs' .- new_locs[l, :]')'
    end
    k_wt
end

# Include efficient sample storage type
include("sampling.jl")

end  # module
