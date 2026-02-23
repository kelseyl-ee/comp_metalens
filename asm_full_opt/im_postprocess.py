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

class PostProcess(nn.Module):
    """
    Takes result images and process through lightweight digital backend. 
    """

    def __init__(self, 
        config, 
        pixel_map,
        X=None, 
        Y=None,
        centers=None,
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
        
        self.compute_img_crop()


    def compute_img_crop(self):
        """
        Compute and cache the tight bbox around the in-FoV region defined by
        self.pixel_map.obj_valid_mask.
        """
        if not hasattr(self.pixel_map, "obj_valid_mask"):
            self.pixel_map.build_obj_sampling_grid(store=True)

        valid = self.pixel_map.obj_valid_mask  # expected [1,1,N,N]
        valid = torch.as_tensor(valid, device=self.device_)

        if valid.ndim == 4:
            v2d = valid[0, 0] > 0  # [N,N] bool
        elif valid.ndim == 2:
            v2d = valid > 0
        else:
            raise ValueError(f"obj_valid_mask should be [1,1,N,N] or [N,N]. Got {tuple(valid.shape)}")

        ys, xs = torch.where(v2d)
        if ys.numel() == 0:
            raise ValueError("obj_valid_mask has no valid pixels (all zeros).")

        y0 = int(ys.min().item())
        y1 = int(ys.max().item()) + 1  # exclusive
        x0 = int(xs.min().item())
        x1 = int(xs.max().item()) + 1  # exclusive

        self.valid_img_crop = (y0, y1, x0, x1)
        return self.valid_img_crop


    def crop_imgs(self, imgs):
        """
        Crop images [B, K, N, N] using cached self.valid_img_box.
        Returns [B, K, H_crop, W_crop].
        """
        if not hasattr(self, "valid_img_box"):
            self.compute_img_crop()

        imgs = imgs.to(device=self.device_, dtype=self.dtype_) 

        if imgs.ndim != 4:
            raise ValueError(f"Expected imgs with shape [B,K,N,N]. Got {tuple(imgs.shape)}")

        y0, y1, x0, x1 = self.valid_img_crop
        return imgs[:, :, y0:y1, x0:x1]


    def downsample_imgs(self, imgs, H=None, W=None):
        """
        Downsample images to (self.H_obj, self.W_obj).
        """
        imgs = imgs.to(device=self.device_, dtype=self.dtype_)

        if imgs.ndim != 4:
            raise ValueError(f"Expected imgs with shape [B,K,H,W]. Got {tuple(imgs.shape)}")
        
        H_out = self.H_obj if H is None else int(H)
        W_out = self.W_obj if W is None else int(W)

        imgs_ds = F.interpolate(imgs, size=(H_out, W_out), mode="area")

        return imgs_ds

    
    def subtract_imgs(self, imgs):
        pos = imgs[:, 0::2]  # e.g. [B,8,28,28]
        neg = imgs[:, 1::2]  # e.g. [B,8,28,28]
        fm = pos - neg       # e.g. [B,8,28,28]

        fm = torch.rot90(fm, k=3, dims=(-2, -1))  # rotate spatial dims 
        return fm     


    def forward(self, imgs, subtract=True, H=None, W=None):
        crop = self.crop_imgs(imgs)
        imgs_ds = self.downsample_imgs(crop)
        
        if subtract:
            fm = self.subtract_imgs(imgs_ds)
            imgs_pp = fm
        else:
            imgs_pp = imgs_ds

        return imgs_pp
