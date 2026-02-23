import numpy as np
import matplotlib.pyplot as plt
import config
from scipy.special import j1
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import j1


MM = 1E-3
UM = 1E-6
NM = 1E-9

class GenerateWaves(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_  = torch.float32

        self.wavl = config.WAVL  
        self.grid_N = config.GRID_N
        self.efl = config.EFL
        self.lens_D = config.LENS_D
        self.hfov = config.HFOV
        self.dx = config.PIX_SIZE

        self.H_obj = config.H_OBJ
        self.W_obj = config.W_OBJ
        self.field_strategy = config.FIELD_STRATEGY
        self.block_size = config.BLOCK_SIZE

        self.build_spatial_grid()
        
    

    def build_spatial_grid(self):
        """
        Builds a centered spatial grid at the metalens plane.
        """
        N = self.grid_N
        dx = self.dx
        coords = (torch.arange(N, device=self.device_, dtype=self.dtype_)
                - (N // 2)) * dx

        x = coords.clone()
        y = coords.clone()
        X, Y = torch.meshgrid(x, y, indexing="ij")

        self.x = x
        self.y = y
        self.X = X
        self.Y = Y

        return x, y, X, Y

    
    def sample_field_points(self, strategy=None):
        """
        Choose which field points to simulate PSFs at, and build a pixel->sample routing map.

        Returns
        -------
        uv_samples : torch.Tensor, shape [P, 2], dtype float
            Sample/reference locations in OBJECT pixel coordinates (u, v), using pixel-center convention.
            u increases along width (x-ish), v increases along height (y-ish).

        pixel_to_sample : torch.Tensor, shape [H_obj, W_obj], dtype long
            For each object pixel (v,u), gives the PSF sample index p in [0..P-1] that should be used.

        Notes
        -----
        - Pixel-center convention: pixel (u,v) has center at (u+0.5, v+0.5).
        """
        H = int(self.H_obj)
        W = int(self.W_obj)
        if strategy is None:
            strategy = self.field_strategy
        strategy = strategy.lower()

        if strategy == "full":
            # One sample per pixel center
            u = torch.arange(W, device=self.device_, dtype=self.dtype_) + 0.5  # [W]
            v = torch.arange(H, device=self.device_, dtype=self.dtype_) + 0.5  # [H]
            U, V = torch.meshgrid(u, v, indexing="ij")  # [W,H]

            uv_samples = torch.stack([U.reshape(-1), V.reshape(-1)], dim=-1)  # [P,2], P=W*H

            # Build pixel_to_sample[v,u] that matches p = u*H + v
            u_idx = torch.arange(W, device=self.device_, dtype=torch.long).view(1, W)  # [1,W]
            v_idx = torch.arange(H, device=self.device_, dtype=torch.long).view(H, 1)  # [H,1]
            pixel_to_sample = (u_idx * H + v_idx).to(torch.long)  # broadcast -> [H,W]

            self.uv_samples = uv_samples
            self.pixel_to_sample = pixel_to_sample
            return uv_samples, pixel_to_sample

        elif strategy == "block":
            B = int(getattr(self, "block_size", 4))  # block size in pixels (NxN)

            # number of blocks along each dimension (ceil division)
            nBH = (H + B - 1) // B
            nBW = (W + B - 1) // B

            # Block reference points = block centers (in pixel-center coords)
            # block i spans v in [i*B, (i+1)*B - 1], similarly for u
            # center (in pixel-center coords) = (start + end + 1)/2
            # where pixel centers are at index+0.5.
            u_centers = []
            for j in range(nBW):
                u0 = j * B
                u1 = min((j + 1) * B - 1, W - 1)
                u_centers.append(0.5 * ((u0 + 0.5) + (u1 + 0.5)))
            v_centers = []
            for i in range(nBH):
                v0 = i * B
                v1 = min((i + 1) * B - 1, H - 1)
                v_centers.append(0.5 * ((v0 + 0.5) + (v1 + 0.5)))

            u_centers = torch.tensor(u_centers, device=self.device_, dtype=self.dtype_)  # [nBW]
            v_centers = torch.tensor(v_centers, device=self.device_, dtype=self.dtype_)  # [nBH]

            U_c, V_c = torch.meshgrid(u_centers, v_centers, indexing="ij")  # [nBW,nBH]
            uv_samples = torch.stack([U_c.reshape(-1), V_c.reshape(-1)], dim=-1)  # [P,2], P=nBW*nBH

            # block_v = v//B (row index), block_u = u//B (col index)
            # u-major sample indexing (matches uv_samples flattening): p = block_u*nBH + block_v
            v_idx = torch.arange(H, device=self.device_, dtype=torch.long).view(H, 1)  # [H,1]
            u_idx = torch.arange(W, device=self.device_, dtype=torch.long).view(1, W)  # [1,W]

            block_v = torch.div(v_idx, B, rounding_mode="floor")  # [H,1] in [0..nBH-1]
            block_u = torch.div(u_idx, B, rounding_mode="floor")  # [1,W] in [0..nBW-1]
            pixel_to_sample = (block_u * nBH + block_v).to(torch.long)  # broadcast -> [H,W]

            self.uv_samples = uv_samples
            self.pixel_to_sample = pixel_to_sample
            self.nBH = nBH
            self.nBW = nBW
            self.block_size = B

            return uv_samples, pixel_to_sample

        else:
            raise ValueError(f"Unknown field_strategy='{strategy}'. Use 'full' or 'block'.")

  
    def uv_to_angles(self, uv_samples, hfov_deg=None):
            """
            Map object-plane sample coordinates (u,v) to field angles (theta_x, theta_y),
            enforcing: (u,v) = (0.5, 0.5) -> ( +HFOV/sqrt(2), -HFOV/sqrt(2) )
                    (u,v) = (W-0.5, 0.5) -> ( -HFOV/sqrt(2), -HFOV/sqrt(2) )

            Returns
            -------
            theta_x : torch.Tensor, shape [P]
            theta_y : torch.Tensor, shape [P]
                Angles in radians.
            """
            if hfov_deg is None:
                hfov_deg = float(self.hfov)

            uv = torch.as_tensor(uv_samples, device=self.device_, dtype=self.dtype_)

            u = uv[:, 0]
            v = uv[:, 1]
            H = float(self.H_obj)
            W = float(self.W_obj)

            # Center in pixel-center coords 
            u_c = W / 2.0
            v_c = H / 2.0
            du = u - u_c
            dv = v - v_c

            # Normalize so that u=0.5 -> -1 and u=W-0.5 -> +1 (same for v)
            r_u = (W / 2.0) - 0.5
            r_v = (H / 2.0) - 0.5
            u_n = du / r_u
            v_n = dv / r_v

            theta_diag = math.radians(hfov_deg)
            a = theta_diag / math.sqrt(2.0)

            theta_x = torch.atan(-u_n * math.tan(a))
            theta_y = torch.atan( v_n * math.tan(a))
            self.theta_x = theta_x
            self.theta_y = theta_y

            return theta_x, theta_y


    def angles_to_k(self, theta_x, theta_y):
        """
        Convert field angles to transverse wavevectors (kx, ky).
        """
        wavl = float(self.wavl)
        k0 = (2.0 * math.pi) / wavl

        tx = torch.as_tensor(theta_x, device=self.device_, dtype=self.dtype_)
        ty = torch.as_tensor(theta_y, device=self.device_, dtype=self.dtype_)

        kx = k0 * torch.sin(tx)
        ky = k0 * torch.sin(ty)

        return kx, ky

    
    def make_plane_waves(self, kx, ky, X=None, Y=None):
        """
        Generate a batch of plane waves at the metalens plane:
            U0[p,:,:] = exp(i*(kx[p]*X + ky[p]*Y))
        """
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y

        X = torch.as_tensor(X, device=self.device_, dtype=self.dtype_)
        Y = torch.as_tensor(Y, device=self.device_, dtype=self.dtype_)

        kx = torch.as_tensor(kx, device=self.device_, dtype=self.dtype_).view(-1, 1, 1)  # [P,1,1]
        ky = torch.as_tensor(ky, device=self.device_, dtype=self.dtype_).view(-1, 1, 1)  # [P,1,1]

        phase = kx * X + ky * Y  # [P,N,N] by broadcasting

        # Make complex field
        U0 = torch.exp(1j * phase.to(torch.float32))  # complex64
        self.U0 = U0

        return U0


    def generate_plane_wave_stack(self, strategy=None, block_size=None, hfov_deg=None):
            """
            Convenience wrapper that builds a batch of plane waves U0 for the current config.
            Returns
            -------
            U0 : torch.Tensor, shape [P,N,N], complex
            uv_samples : torch.Tensor, shape [P,2]
            pixel_to_sample : torch.Tensor, shape [H_obj,W_obj]
            theta_x, theta_y : torch.Tensor, shape [P]
            kx, ky : torch.Tensor, shape [P]
            """
            if not hasattr(self, "X"):
                self.build_spatial_grid()

            if strategy is None:
                strategy = getattr(self, "field_strategy", "full")
            if block_size is not None:
                self.block_size = int(block_size)

            uv_samples, pixel_to_sample = self.sample_field_points(strategy=strategy)
            theta_x, theta_y = self.uv_to_angles(uv_samples, hfov_deg=hfov_deg)
            kx, ky = self.angles_to_k(theta_x, theta_y)
            U0 = self.make_plane_waves(kx, ky, self.X, self.Y)

            return U0, uv_samples, pixel_to_sample, theta_x, theta_y, kx, ky
