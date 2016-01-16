###############################################################################
# Type for efficient storage of GPC realizations
type GaussianProcessSample
    knot_locs::Array
    knot_values::Array
    dim::Integer
    nknot::Integer
    iter::Integer
    warmup::Integer

    GaussianProcessSample(knot_locs::Array, iters::Integer, warmup::Integer) =
      new(knot_locs,
          Array(Float64, size(knot_locs, 1), iters),
          size(knot_locs, 2),
          size(knot_locs, 1),
          iters,
          warmup)
end

# Extract a specific realization as a GPC
GaussianProcessConvolution(GPS::GaussianProcessSample, realization::Integer) = 
  new(GPS.knot_locs,
      GPS.knot_values[:, realization],
      GPS.dim,
      GPS.nknot)

