import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from generate_waves import GenerateWaves
from phase_mask import PhaseMask
from asm_prop import ASMPropagator
import config


MM = 1E-3
UM = 1E-6
NM = 1E-9

class PixelMap(nn.Module):
    
    def __init__(self, config):

        super().__init__()
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_  = torch.float32

    import torch
import torch.nn as nn

class PixelMap(nn.Module):
    """
    Utility class to map object pixels -> angles -> sensor shifts/pixels.
    """

    def __init__(self, config, device=None, dtype=torch.float32):
        super().__init__()
        self.device_ = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype_ = dtype

        self.H_obj = int(config.H_OBJ)
        self.W_obj = int(config.W_OBJ)
        self.efl = config.EFL

  
    def pixel_uv_grid(self, flatten=True):
        """
        Returns uv coordinates for every object pixel center.
        """
        H, W = self.H_obj, self.W_obj

        u = torch.arange(W, device=self.device_, dtype=self.dtype_) + 0.5  
        v = torch.arange(H, device=self.device_, dtype=self.dtype_) + 0.5  

        # full grid of centers
        U, V = torch.meshgrid(u, v, indexing="ij")  # U,V: [W, H]
        uv_hw = torch.stack([U, V], dim=-1)

        if flatten:
            return uv_hw.reshape(-1, 2)          # [H*W, 2]
        return uv_hw


    def uv_to_angles_pixels(self, uv_pixels, generate_waves: "GenerateWaves", hfov_deg=None):
        """
        Use GenerateWaves.uv_to_angles() to map object pixel centers -> field angles.
        """
        uv = torch.as_tensor(uv_pixels, device=self.device_, dtype=self.dtype_).reshape(-1, 2)
        theta_x, theta_y = generate_waves.uv_to_angles(uv, hfov_deg=hfov_deg)

        theta_x = theta_x.to(device=self.device_, dtype=self.dtype_).reshape(-1)
        theta_y = theta_y.to(device=self.device_, dtype=self.dtype_).reshape(-1)

        return theta_x, theta_y


    def angles_to_sensor_xy(self, theta_x, theta_y, efl_m=None):
        """
        Convert angles -> sensor-plane shifts using:
            x = f * tan(theta_x)
            y = f * tan(theta_y)
        """
        f = float(self.efl if efl_m is None else efl_m)

        tx = torch.as_tensor(theta_x, device=self.device_, dtype=self.dtype_)
        ty = torch.as_tensor(theta_y, device=self.device_, dtype=self.dtype_)

        sensor_x = f * torch.tan(tx)
        sensor_y = f * torch.tan(ty)

        return sensor_x, sensor_y
