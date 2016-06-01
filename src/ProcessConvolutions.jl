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
abstract AbstractProcess <: Any

immutable ProcessConvolution <: AbstractProcess
    knot_locs::AbstractArray
    knot_values::Vector
    dim::Integer
    nknot::Integer

    ProcessConvolution(knot_locs::AbstractArray, knot_values::Vector) =
      new(knot_locs,
          knot_values,
          size(knot_locs, 2),
          size(knot_locs, 1))
    ProcessConvolution(knot_locs::AbstractArray, dist::UnivariateDistribution) =
      new(knot_locs,
          rand(dist, size(knot_locs, 1)),
          size(knot_locs, 2),
          size(knot_locs, 1))
    ProcessConvolution(knot_locs::AbstractArray) =
      ProcessConvolution(knot_locs, Normal(0, 1))
end

knot_locs(pc::ProcessConvolution) = pc.knot_locs
knot_values(pc::ProcessConvolution) = pc.knot_values
nknot(pc::ProcessConvolution) = pc.nknot
dim(pc::ProcessConvolution) = pc.dim

immutable ZeroProcess <: AbstractProcess end

immutable ConstantProcess <: AbstractProcess
    v::Float64
end

#-----------------------------------------------------------------------------
abstract PredictiveProcessConvolution

type ContinuousPredictivePC <: PredictiveProcessConvolution
    ProcConv::AbstractProcess
    ConvKern::AbstractConvolutionKernel
    PredLocs::Array{Float64}
    KnotWt::Array{Float64}
    Transform::Function
end

type DiscretePredictivePC <: PredictiveProcessConvolution
    KnotLocs::Array{Float64}
    PredLocs::Array{Float64}
    Process::Array{Symbol}
    KnotValue::Dict{Symbol, Array{Float64}}
    ConvKernel::Dict{Symbol, AbstractConvolutionKernel}
    KnotWt::Dict{AbstractConvolutionKernel, Array{Float64}}
end

# Outer constructor for simplest case: no knot values specified, one
# shared kernel for all processes.
function DiscretePredictivePC(knotlocs::Array{Float64},
                              predlocs::Array{Float64},
                              processlist::Vector{Symbol},
                              kernel::AbstractConvolutionKernel)
    nkts = size(knotlocs, 1)
    npred = size(predlocs, 1)
    nproc = length(processlist)

    kern = Dict{Symbol, AbstractConvolutionKernel}()
    for proc in processlist
        kern[proc] = kernel
    end

    kv = Dict{Symbol, Vector{Float64}}()
    for proc in processlist
        kv[proc] = randn(nkts)
    end

    kw = Dict{AbstractConvolutionKernel,
              Array{Float64}}(kernel => knot_wt(knotlocs, kernel, predlocs))

    DiscretePredictivePC(knotlocs,
                         predlocs,
                         processlist,
                         kv,
                         kw)
end

# Outer constructor for different kernel case: no knot values specified, one
# shared kernel for all processes.
function DiscretePredictivePC(knotlocs::Array{Float64},
                              predlocs::Array{Float64},
                              kernel::Dict{Symbol, AbstractConvolutionKernel})

    processlist = collect(keys(kernel))
    nkts = size(knotlocs, 1)
    npred = size(predlocs, 1)
    nproc = length(processlist)

    kv = Dict{Symbol, Vector{Float64}}()
    for proc in processlist
        kv[proc] = randn(nkts)
    end

    kw = Dict{AbstractConvolutionKernel,
              Array{Float64, 2}}()
    for kern in unique(values(kernel))
        kw[kern] = knot_wt(knotlocs, kern, predlocs)
    end

    DiscretePredictivePC(knotlocs,
                         predlocs,
                         processlist,
                         kv,
                         kw)
end

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
