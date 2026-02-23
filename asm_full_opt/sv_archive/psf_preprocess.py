import torch
import torch.nn as nn
import torch.nn.functional as F

from generate_waves import GenerateWaves
from phase_mask import PhaseMask
from asm_prop import ASMPropagator
import config



class PSFPreProcessor(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_  = torch.float32

        self.grid_N = config.GRID_N
        self.dx = config.PIX_SIZE
        self.psf_window_N = config.PSF_WINDOW_N
        self.tukey_alpha = config.TUKEY_ALPHA  # Tukey window alpha parameter

        self.build_window()


    def build_window(self):
        """
        Build a 2D apodization window for the cropped PSF.
        Saved as self.W2d with shape [Wn, Wn].
        """
        Wn = int(self.psf_window_N)

        # Flat top Tukey window
        a = float(self.tukey_alpha)
        n = torch.arange(Wn, device=self.device_, dtype=self.dtype_)
        x = n / (Wn - 1)
        w1 = torch.ones_like(x)
        left = x < (a / 2)
        right = x > (1 - a / 2)
        w1[left] = 0.5 * (1 + torch.cos(2 * torch.pi * (x[left] / a - 0.5)))
        w1[right] = 0.5 * (1 + torch.cos(2 * torch.pi * ((x[right] - 1) / a + 0.5)))


        W2d = w1[:, None] * w1[None, :]
        self.register_buffer("W2d", W2d)

        return self.W2d


    def crop_center(self, psf_stack, pixel_map, gw, uv_samples=None, hfov_deg=None):
        psf = torch.as_tensor(psf_stack, device=self.device_, dtype=self.dtype_)  # [P,N,N]
        P, N, _ = psf.shape
        Wn = int(self.psf_window_N)

        if uv_samples is None:
            uv_samples = gw.uv_samples
        uv_samples = torch.as_tensor(uv_samples, device=pixel_map.device_, dtype=pixel_map.dtype_).reshape(P, 2)

        x_m, y_m = pixel_map.angles_to_sensor_xy(gw.theta_x, gw.theta_y)

        # meters -> *pixel centers* on the PSF grid 
        dx = float(self.dx)
        c = (N - 1) / 2.0  # optical axis pixel center (float)
        i_c = c + (x_m / dx)  
        j_c = c + (y_m / dx)   

        # Build local crop offsets (pixel units)
        half = (Wn - 1) / 2.0
        offs = torch.arange(Wn, device=self.device_, dtype=self.dtype_) - half  # [Wn]
        dI, dJ = torch.meshgrid(offs, offs, indexing="ij")  # each [Wn,Wn]

        # Absolute sample positions on full grid
        I = i_c.view(P, 1, 1) + dI.view(1, Wn, Wn)  # [P,Wn,Wn]
        J = j_c.view(P, 1, 1) + dJ.view(1, Wn, Wn)  # [P,Wn,Wn]

        # Normalize to [-1,1] for grid_sample (align_corners=True convention)
        j_norm = (2.0 * J / (N - 1)) - 1.0
        i_norm = (2.0 * I / (N - 1)) - 1.0
        grid = torch.stack([j_norm, i_norm], dim=-1)  # [P,Wn,Wn,2] = (x,y) = (col,row)

        # Sample crops (bilinear handles subpixel centers)
        psf_in = psf.unsqueeze(1)  # [P,1,N,N]
        psf_crop = F.grid_sample(
            psf_in, grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        ).squeeze(1)  # [P,Wn,Wn]

        return psf_crop


    def crop_center2(self, psf_stack):
        """
        Center-crop each PSF from [P, N, N] to [P, Wn, Wn] using the optical axis pixel as the center.
        Returns
        -------
        psf_crop : torch.Tensor, shape [P, Wn, Wn]
        """
        psf = torch.as_tensor(psf_stack, device=self.device_, dtype=self.dtype_)

        N = int(self.grid_N)
        Wn = int(self.psf_window_N)

        c = N // 2
        h = Wn // 2

        y0 = c - h
        y1 = y0 + Wn
        x0 = c - h
        x1 = x0 + Wn

        psf_crop = psf[:, y0:y1, x0:x1]

        return psf_crop


    def normalize(self, psf_crop, eps=1e-12):
        """
        Normalize each cropped PSF so sum over pixels = 1.
        Returns
        -------
        psf_norm : torch.Tensor, shape [P, Wn, Wn]
        """
        psf = torch.as_tensor(psf_crop, device=self.device_, dtype=self.dtype_)
        denom = psf.sum(dim=(-2, -1), keepdim=True) + eps
        return psf / denom


    def apply_apodization(self, psf_crop):
        psf = torch.as_tensor(psf_crop, device=self.device_, dtype=self.dtype_)
        return psf * self.W2d


    def forward(self, psf_stack, pixel_map, gw, uv_samples=None, hfov_deg=None):
        """
        Full preprocessing pipeline:
            full PSFs -> center crop -> window -> optional renormalize

        Returns
        -------
        psf_out : torch.Tensor, shape [P, Wn, Wn]
        """
        psf_crop = self.crop_center(psf_stack, pixel_map, gw, uv_samples=None, hfov_deg=None)
        psf_crop = self.apply_apodization(psf_crop)

        return psf_crop
