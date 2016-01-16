using PyPlot
import PyPlot: contourf

inv_logit(x) = 1 ./ (1 .+ exp(-x))

function contourf(GPC::GaussianProcessConvolution, kern::AbstractConvolutionKernel; levels = 7, transform = identity)
    ndim = dim(GPC)
    if ndim != dim(kern)
        throw(DimensionMismatch)
    elseif ndim > 2
        error("Too many dimensions")
    end

    xmin, xmax = extrema(GPC.knot_locs[:, 1])
    ymin, ymax = extrema(GPC.knot_locs[:, 2])

    aspect = (xmax - xmin) / (ymax - ymin)

    n_x = 1025
    x_interp = linspace(xmin, xmax, n_x)
    n_y = Int(ceil(n_x / aspect))
    y_interp = linspace(ymax, ymin, n_y)
    
    interp = Array(Float64, n_y, n_x)
    
    for i in 1:n_x
        for j in 1:n_y
            interp[j, i] = predict(GPC, kern, [x_interp[i] y_interp[j]])[1]
        end
    end
    
    interp = transform(interp)

    PyPlot.contourf(x_interp, y_interp, interp, levels)
end
