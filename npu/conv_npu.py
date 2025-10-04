import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A convolution kernel that you need to implement.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height
out_pool_width = out_width

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""


@nki.jit
def conv2d(X, W, bias):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_height, out_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax
    c_out_pmax = nl.tile_size.pmax
    n_tiles_c_out = out_channels // c_out_pmax

    tile_h = min(out_height, nl.tile_size.gemm_moving_fmax // out_width)
    tile_h_w = tile_h * out_width

    n_tiles_c_in = in_channels // c_in_pmax
    n_tiles_c_out = out_channels // c_out_pmax
    n_tiles_h_w = (out_height + tile_h - 1) // tile_h

    # load W
    W_tiles = nl.ndarray(
        (c_out_pmax, n_tiles_c_out, in_channels, filter_height, filter_width),
        dtype=bias.dtype,
        buffer=nl.sbuf,
    )
    for c_out in nl.affine_range(n_tiles_c_out):
        start_c_out = c_out * c_out_pmax
        end_c_out = (c_out + 1) * c_out_pmax
        for c_in in nl.affine_range(in_channels):
            W_tiles[:, c_out, c_in, :, :] = nl.load(
                W[start_c_out:end_c_out, c_in, :, :]
            )

    # load bias
    bias_tiles = nl.ndarray(
        (c_out_pmax, n_tiles_c_out), dtype=bias.dtype, buffer=nl.sbuf
    )
    for c_out in nl.affine_range(n_tiles_c_out):
        start_c_out = c_out * c_out_pmax
        end_c_out = (c_out + 1) * c_out_pmax
        bias_tiles[:, c_out] = nl.load(bias[start_c_out:end_c_out])

    res_c_out = nl.arange(c_out_pmax)[:, None]
    res_h_w = nl.arange(tile_h_w)[None, :]

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        for h_w in nl.affine_range(n_tiles_h_w):
            start_h = h_w * tile_h
            height = tile_h + filter_height - 1

            # load X by input channel tiles
            X_tiles = nl.ndarray(
                (n_tiles_c_in, nl.par_dim(c_in_pmax), height, input_width),
                dtype=X.dtype,
                buffer=nl.sbuf,
            )
            for c_in in nl.affine_range(n_tiles_c_in):
                par, row, col = nl.mgrid[0:c_in_pmax, 0:height, 0:input_width]
                X_tiles[c_in] = nl.load(
                    X[b, c_in * c_in_pmax + par, start_h + row, col],
                    mask=((start_h + row) < input_height),
                )

            for c_out in nl.affine_range(n_tiles_c_out):
                W_tile = nl.copy(W_tiles[:, c_out, :, :])

                bias_tile = nl.ndarray(
                    (c_out_pmax, 1), dtype=bias.dtype, buffer=nl.sbuf
                )
                bias_tile[:, 0] = nl.copy(bias_tiles[:, c_out])

                res = nl.zeros((c_out_pmax, tile_h_w), dtype=nl.float32, buffer=nl.psum)

                for fh in nl.affine_range(filter_height):
                    for fw in nl.affine_range(filter_width):
                        for c_in in nl.affine_range(n_tiles_c_in):
                            w_start_c_in = c_in * c_in_pmax
                            w_end_c_in = (c_in + 1) * c_in_pmax
                            W_tile_slice = W_tile[:, w_start_c_in:w_end_c_in, fh, fw]

                            X_tile = nl.ndarray(
                                (nl.par_dim(c_in_pmax), tile_h_w),
                                dtype=X.dtype,
                                buffer=nl.sbuf,
                            )

                            for h in nl.affine_range(tile_h):
                                if start_h + h >= out_height:
                                    break

                                x_start_h = h * out_width
                                x_end_h = (h + 1) * out_width
                                X_tile[:, x_start_h:x_end_h] = X_tiles[
                                    c_in, :, h + fh, fw : fw + out_width
                                ]

                            res[res_c_out, res_h_w] += nl.matmul(W_tile_slice, X_tile)

                res = nl.add(res, bias_tile)
                out = nl.copy(res.reshape((c_out_pmax, tile_h, out_width)))

                par, row, col = nl.mgrid[0:c_out_pmax, 0:tile_h, 0:out_width]
                nl.store(
                    X_out[b, c_out * c_out_pmax + par, start_h + row, col],
                    value=out,
                    mask=((start_h + row) < out_height),
                )

    return X_out
