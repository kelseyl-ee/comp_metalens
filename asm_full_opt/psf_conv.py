import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


MM = 1E-3
UM = 1E-6
NM = 1E-9

class PSFConv(nn.Module):
    """
    Takes input object and on-axis PSFs, returns image convolved w PSFs 
    """

    def __init__(self, 
        config, 
        pixel_map,
        X=None, 
        Y=None,
        ):

        super().__init__()
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_  = torch.float32

        self.pixel_map = pixel_map

        self.H_obj = int(config.H_OBJ)
        self.W_obj = int(config.W_OBJ)
        self.hfov = config.HFOV
        self.efl = config.EFL

        self.wavl = config.WAVL  
        self.grid_N = config.GRID_N
        self.lens_D = config.LENS_D
        self.dx = config.PIX_SIZE

        if X is None and Y is None:
            self.build_spatial_grid()
        else:
            self.x = None  
            self.y = None
            self.register_buffer("X", X)
            self.register_buffer("Y", Y)
    
    def render_sensor_ideal(self, obj):
        pm = self.pixel_map
        im_sensor_ideal = pm.render_sensor_ideal(obj)
        return im_sensor_ideal
    
    def make_otf(self, psf, fft_shape=None, normalize=True, eps=1e-12):
        psf = torch.as_tensor(psf, device=self.device_, dtype=self.dtype_)

        if psf.ndim != 2:
            raise ValueError(f"psf must be 2D [H,W], got shape {tuple(psf.shape)}")

        if normalize:
            psf = psf / (psf.sum() + eps)

        if fft_shape is None:
            fft_shape = (int(self.grid_N), int(self.grid_N))

        psf0 = torch.fft.ifftshift(psf)                    # center -> (0,0)
        otf  = torch.fft.fft2(psf0, s=fft_shape)           # complex

        return otf
    
    def make_otfs(self, psfs, normalize=True, eps=1e-12):
        """
        Can support psf stack [K, N, N]
        """
        psfs = torch.as_tensor(psfs, device=self.device_, dtype=self.dtype_)

        if psfs.ndim != 3:
            raise ValueError(f"psf must be (K, N, N), got shape {tuple(psf.shape)}")

        if normalize:
            psfs = psfs / (psfs.sum(dim=(-2, -1), keepdim=True) + eps)

        psf0 = torch.fft.ifftshift(psfs, dim=(-2, -1))
        otfs  = torch.fft.fft2(psf0, dim=(-2, -1))

        return otfs
    
    def sensor_image(self, obj, psfs):
        """
        Render final sensor image:
            object -> ideal sensor image -> PSF convolution
        """
        # 1) ideal sensor image (geometry only)
        im_ideal = self.render_sensor_ideal(obj)           # [B,1,N,N]
        # 2) OTF from PSF
        otfs = self.make_otfs(psfs)                            # [K,N,N] complex

        # 3) FFT-based convolution
        F_img = torch.fft.fft2(im_ideal, dim=(-2, -1))  # [B, 1, N, N]
        imgs = torch.fft.ifft2(
            F_img * otfs[None, :, :, :],                 # [B, K, N, N]
            dim=(-2, -1)
        ).real

        return imgs


    #--------------------------------------------------------------------
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

        return x, y, X, Y
    
    
