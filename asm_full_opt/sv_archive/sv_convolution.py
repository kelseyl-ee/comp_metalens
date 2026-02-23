import math
import torch
import torch.nn as nn

from generate_waves import GenerateWaves
from phase_mask import PhaseMask
from asm_prop import ASMPropagator
import config
from torch.nn import functional as F


class SVConvolution(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_  = torch.float32

        self.H_obj = config.H_OBJ
        self.W_obj = config.W_OBJ
        self.block_N = config.BLOCK_SIZE
        self.PSF_wn = config.PSF_WINDOW_N
        self.dx = config.PIX_SIZE

    def images_to_blocks(self, imgs):
        imgs = torch.as_tensor(imgs)
        assert imgs.ndim == 3, f"Expected [N,H,W], got {tuple(imgs.shape)}"
        N, H, W = imgs.shape
        B = self.block_N

        nBH = H // B
        nBW = W // B
        nBlocks = nBH * nBW

        # [N, H, W] -> [N, nBH, B, nBW, B]
        x = imgs.reshape(N, nBH, B, nBW, B)
        # -> [N, B, B, nBW, nBH]  (u-major block layout)
        x = x.permute(0, 2, 4, 3, 1).contiguous()
        # -> [N, B, B, nBlocks]
        x = x.reshape(N, B, B, nBlocks)
        # -> [N, nBlocks, B, B]
        blocks = x.permute(0, 3, 1, 2).contiguous()

        return blocks

    def upsample_blocks(self, blocks, upsample_N=None, mode="nearest", align_corners=None):
        """
        Upsample per-region blocks.
        Returns
        -------
        blocks_up : torch.Tensor
            Shape [N, P, upsample_N, upsample_N]
        """
        x = torch.as_tensor(blocks, device=self.device_, dtype=self.dtype_)
        assert x.ndim == 4, f"Expected [N,P,B,B], got {tuple(x.shape)}"

        N, P, B1, B2 = x.shape
        assert B1 == B2, "Blocks must be square [B,B]."

        if upsample_N is None:
            if not hasattr(self, "PSF_wn"):
                raise AttributeError(
                    "upsample_N is None but self.PSF_wn not set. "
                    "Set self.PSF_wn = config.PSF_WINDOW_N in __init__."
                )
            upsample_N = int(self.PSF_wn)
        upsample_N = int(upsample_N)

        # [N,P,B,B] -> [N*P,1,B,B]
        x = x.reshape(N * P, 1, B1, B1)
        # Upsample
        x_up = F.interpolate(
            x,
            size=(upsample_N, upsample_N),
            mode=mode,
            align_corners=align_corners
        )
        # Back to [N, P, upsample_N, upsample_N]
        blocks_up = x_up.reshape(N, P, upsample_N, upsample_N)

        return blocks_up

    

    def convolve_with_first_psf(self, imgs, psf_stack):
        """
        Convolve a batch of images with the first PSF in a PSF stack.
        """
        imgs = torch.as_tensor(imgs)
        psf_stack = torch.as_tensor(psf_stack)

        N, H, W = imgs.shape
        _, K, P = psf_stack.shape
        assert P >= 1, "PSF stack must have at least one PSF"

        # Take first PSF
        psf = psf_stack[0, :, :]            # [K, K]

        # Prepare for conv2d
        x = imgs.unsqueeze(1)               # [N, 1, H, W]
        w = psf.unsqueeze(0).unsqueeze(0)   # [1, 1, K, K]

        # Same-size convolution
        pad = K - 1
        out = F.conv2d(x, w, padding=pad)   # [N, 1, H, W]

        return out.squeeze(1)


    def convolve_blocks_with_matching_psfs(self, blocks, psf_stack):
        """
        Depthwise (per-channel) full convolution:
        for each region i, convolve blocks[:, i] with psf_stack[i].
        Returns
        -------
        out : torch.Tensor
            Shape [N, P, (2K-1), (2K-1)]   (full convolution)
        """
        x = torch.as_tensor(blocks, device=self.device_, dtype=self.dtype_)
        w0 = torch.as_tensor(psf_stack, device=self.device_, dtype=self.dtype_)

        assert x.ndim == 4, f"blocks must be [N,P,K,K], got {tuple(x.shape)}"
        assert w0.ndim == 3, f"psf_stack must be [P,K,K], got {tuple(w0.shape)}"

        N, P, K1, K2 = x.shape
        assert K1 == K2, "blocks must be square in the last two dims"
        assert w0.shape[0] == P, f"psf_stack first dim must match P={P}, got {w0.shape[0]}"
        assert w0.shape[1] == K1 and w0.shape[2] == K2, "psf_stack must match block kernel size"
        # w0 = torch.flip(w0, dims=(1,2))

        # Prepare weights for grouped conv:
        # weight shape must be [out_channels, in_channels/groups, kH, kW]
        # For depthwise: out_channels=P, in_channels=P, groups=P -> in_channels/groups = 1
        w = w0.unsqueeze(1)  # [P,1,K,K]

        pad = K1 - 1  # full convolution
        out = F.conv2d(x, w, padding=pad, groups=P)  # [N,P,2K-1,2K-1]

        return out

    def splat_blocks_to_sensor(
        self,
        X,
        Y,
        x_centers,
        y_centers,
        block_out,
    ):
        """
        Splat per-region convolution outputs onto the sensor grid.

        Parameters
        ----------
        X, Y : torch.Tensor
            Sensor meshgrid, shape [X_N, Y_N], physical coords (meters)
        x_centers, y_centers : torch.Tensor
            Physical sensor-plane centers for each region, shape [P]
        block_out : torch.Tensor
            Per-region outputs, shape [N, P, Hk, Wk]
        dx : float
            Sensor pixel pitch (meters)

        Returns
        -------
        sensor_img : torch.Tensor
            Accumulated sensor image, shape [N, X_N, Y_N]
        """

        X_N, Y_N = X.shape
        N, P, Hk, Wk = block_out.shape
        dx = self.dx

        assert Hk == Wk, "Block outputs must be square"
        hk = Hk // 2

        # Sensor grid center index
        cx = X_N // 2
        cy = Y_N // 2

        # Allocate output
        sensor_img = torch.zeros((N, X_N, Y_N), device=self.device_, dtype=self.dtype_)

        # Convert physical centers -> pixel indices
        # NOTE: x -> row index, y -> col index (matches your earlier convention)
        i_centers = cx + torch.round(x_centers / dx).to(torch.long)
        j_centers = cy + torch.round(y_centers / dx).to(torch.long)

        for p in range(P):
            ic = i_centers[p].item()
            jc = j_centers[p].item()

            # Sensor slice
            i0 = ic - hk
            i1 = ic + hk + 1
            j0 = jc - hk
            j1 = jc + hk + 1

            # Block slice (handle boundaries)
            bi0 = max(0, -i0)
            bj0 = max(0, -j0)
            bi1 = Hk - max(0, i1 - X_N)
            bj1 = Wk - max(0, j1 - Y_N)

            si0 = max(i0, 0)
            sj0 = max(j0, 0)
            si1 = min(i1, X_N)
            sj1 = min(j1, Y_N)

            if si0 >= si1 or sj0 >= sj1:
                continue  # completely off sensor

            # Accumulate
            sensor_img[:, si0:si1, sj0:sj1] += block_out[:, p, bi0:bi1, bj0:bj1]

        return sensor_img



