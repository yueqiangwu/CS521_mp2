import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity


class ConvModel(nn.Module):
    def __init__(
        self, H, W, in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding

        self.H = H
        self.W = W

        # TO DO: Define static shapes here.
        self.TILE_SIZE = 1024

        # Precompute output size
        # self.out_h = ...
        # self.out_w = ...
        self.out_h = (H + 2 * padding - kernel_size) // stride + 1
        self.out_w = (W + 2 * padding - kernel_size) // stride + 1

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def im2col_manual(self, x):
        N = x.shape[0]  # batch size can remain dynamic
        C = self.in_channels
        KH = KW = self.kernel_size
        S = self.stride
        P = self.padding
        out_h = self.out_h
        out_w = self.out_w

        # Pad input
        x_pad = F.pad(x, (P, P, P, P))

        # TO DO: Convert input (x) into shape (N, out_h*out_w, C*KH*KW).
        # Refer to Lecture 3 for implementing this operation.

        # patches = ...
        patches = []
        for i in range(0, KH):
            i_end = i + S * out_h
            for j in range(0, KW):
                j_end = j + S * out_w
                t = x_pad[:, :, i:i_end:S, j:j_end:S]
                patches.append(t)

        patches = torch.stack(patches, dim=0)
        patches = patches.permute(1, 3, 4, 2, 0)
        patches = patches.reshape(N, out_h * out_w, C * KH * KW)

        # return patches
        return patches

    def conv2d_manual(self, x):
        N = x.shape[0]
        C_out = self.out_channels
        KH = KW = self.kernel_size
        out_h = self.out_h
        out_w = self.out_w
        T = self.TILE_SIZE

        # TO DO: 1) convert input (x) into shape (N, out_h*out_w, C*KH*KW).
        # cols = self.im2col_manual(x)
        cols = self.im2col_manual(x)

        # TO DO: 2) flatten self.weight into shape (C_out, C*KH*KW).
        w_flat_t = self.weight.view(C_out, -1).t()

        # TO DO: 3) perform tiled matmul after required reshaping is done.
        out_s = out_h * out_w
        out = torch.empty((N, out_s, C_out), device=x.device)

        for start in range(0, out_s, T):
            end = min(out_s, start + T)
            tile_cols = cols[:, start:end, :]
            tile_out = torch.matmul(tile_cols, w_flat_t)
            out[:, start:end, :] = tile_out

        # TO DO: 4) Add bias.
        out = out + self.bias.view(1, 1, C_out)

        # TO DO: 5) reshape output into shape (N, C_out, out_h, out_w).
        out = out.permute(0, 2, 1)
        out = out.reshape(N, C_out, out_h, out_w)

        # return out
        return out

    def forward(self, x):
        return self.conv2d_manual(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N, C, H, W = 2, 4, 220, 220
    x = torch.randn(N, C, H, W)
    out_channels = 8
    kernel_size = 7
    model = ConvModel(H, W, C, out_channels, kernel_size, stride=1, padding=1)

    # out = model(x)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with record_function("myconv_forward"):
            out = model(x)

    # Test your solution
    conv_ref = F.conv2d(x, model.weight, model.bias, stride=1, padding=1)
    print("PyTorch --- shape check:", out.shape == conv_ref.shape)
    print("PyTorch --- correctness check:", torch.allclose(out, conv_ref, atol=1e-4))
