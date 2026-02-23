import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import j1

import config

MM = 1E-3
UM = 1E-6
NM = 1E-9


class PhaseMask(nn.Module):

    def __init__(
        self,
        config,
        X=None,
        Y=None,
        init="hyperbolic",
        noise_std=0.0,
        wrap_phase=False,
        trainable=True,
        use_aperture=True,
        hard_aperture=True,
    ):
        super().__init__()

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_  = torch.float32

        self.wavl = config.WAVL  
        self.grid_N = config.GRID_N
        self.efl = config.EFL
        self.lens_D = config.LENS_D
        self.hfov = config.HFOV
        self.dx = config.PIX_SIZE

        self.wrap_phase = wrap_phase
        self.use_aperture = use_aperture
        self.hard_aperture = hard_aperture

        if (X is None) or (Y is None):
            self.build_spatial_grid()
        else:
            X = torch.as_tensor(X, device=self.device_, dtype=self.dtype_)
            Y = torch.as_tensor(Y, device=self.device_, dtype=self.dtype_)

            self.register_buffer("X", X)
            self.register_buffer("Y", Y)

        # Build aperture mask (1 inside, 0 outside)
        R = torch.sqrt(self.X * self.X + self.Y * self.Y)
        A = (R <= (self.lens_D / 2.0)).to(self.dtype_)
        self.register_buffer("A", A)

        # Initialize phase
        init = init.lower()
        phi0 = self._make_init_phase(init=init)

        if noise_std > 0.0:
            phi0 = phi0 + torch.randn_like(phi0) * noise_std

        if self.use_aperture:
            phi0 = phi0 * self.A

        if self.wrap_phase:
            phi0 = self._wrap(phi0)

        if trainable:
            self.phi = nn.Parameter(phi0)
        else:
            self.register_buffer("phi", phi0)
    
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

        self.register_buffer("X", X)
        self.register_buffer("Y", Y)

        return x, y, self.X, self.Y


    def _make_init_phase(self, init):
        if init == "hyperbolic":
            return self.hyperbolic_phase(self.X, self.Y, self.wavl, self.efl)
        elif init == "random":
            return (2.0 * math.pi) * torch.rand(
                (self.grid_N, self.grid_N), device=self.device_, dtype=self.dtype_
            ) - math.pi
        elif init in ("zeros", "zero"):
            return torch.zeros((self.grid_N, self.grid_N), device=self.device_, dtype=self.dtype_)
        else:
            raise ValueError(f"Unknown init='{init}'. Use 'hyperbolic', 'random', or 'zeros'.")


    @staticmethod
    def hyperbolic_phase(X, Y, wavl, efl):
        """
        Ideal focusing phase of a thin lens (hyperbolic) for a focus at distance f.

        phi(x,y) = -k * (sqrt(x^2+y^2+f^2) - f)
        """
        k = (2.0 * math.pi) / wavl
        r2 = X * X + Y * Y
        opd = torch.sqrt(r2 + efl * efl) - efl
        return -k * opd


    @staticmethod
    def _wrap(phi):
        return torch.atan2(torch.sin(phi), torch.cos(phi))


    def forward(self):
        phi = self.phi

        if self.wrap_phase:
            phi = self._wrap(phi)

        if self.use_aperture:
            phi = phi * self.A

        return phi


    def apply(self, U):
        phi = self.forward().to(device=U.device)

        if self.use_aperture and self.hard_aperture:
            A = self.A.to(device=U.device)
            return U * A * torch.exp(1j * phi)
        else:
            return U * torch.exp(1j * phi)
        phi = self.forward().to(device=U.device)
        return U * torch.exp(1j * phi)