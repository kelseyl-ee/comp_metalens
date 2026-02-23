import math
import torch
import torch.nn as nn


class ASMPropagator(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_  = torch.float32

        self.wavl = config.WAVL
        self.grid_N = config.GRID_N
        self.dx = config.PIX_SIZE
        self.z = config.Z

        self.build_spatial_grid()
        self.build_frequency_grids()
        self.build_transfer_function()

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

    def build_frequency_grids(self):
        """
        Build spatial-frequency grids matching torch.fft.fft2 ordering.
        """
        N = self.grid_N
        dx = self.dx

        fx = torch.fft.fftfreq(N, d=dx).to(device=self.device_, dtype=self.dtype_)  # cycles/m
        fy = torch.fft.fftfreq(N, d=dx).to(device=self.device_, dtype=self.dtype_)  # cycles/m

        FX, FY = torch.meshgrid(fx, fy, indexing="ij")  # [N,N]

        self.register_buffer("fx", fx)
        self.register_buffer("fy", fy)
        self.register_buffer("FX", FX)
        self.register_buffer("FY", FY)

        return fx, fy, FX, FY
    

    def generate_on_axis_plane_wave(self, normalize=True):
        """
        Generate a single on-axis plane wave U0(x,y) = A * exp(i*phase).
        """
        N = self.grid_N
        U0 = torch.ones((N, N), device=self.device_, dtype=torch.complex64)
        if normalize:
            eps = 1e-12
            power = (U0.abs() ** 2).sum() + eps
            U0 = U0 / torch.sqrt(power)
        return U0



    def build_transfer_function(self, z=None, evanescent=True, store=True):
        """
        Build ASM transfer function H = exp(i*kz*z).

        If z is None, uses self.z.
        """
        if z is None:
            z = self.z

        k0 = (2.0 * math.pi) / self.wavl

        kx = (2.0 * math.pi) * self.FX  # rad/m
        ky = (2.0 * math.pi) * self.FY  # rad/m

        kz_sq = (k0 * k0) - (kx * kx + ky * ky)

        if evanescent:
            kz = torch.sqrt(kz_sq.to(torch.complex64))
            H  = torch.exp(1j * kz * z)
        else:
            kz = torch.sqrt(torch.clamp(kz_sq, min=0.0))
            H = torch.exp(1j * kz.to(torch.float32) * z)

        if store:
            self.register_buffer("H", H)

        return H


    def forward(self, phase_mask, U0_stack=None, normalize=False, return_field=False, H=None, apply_phase=True):
        """
        Apply phase mask and ASM propagate a batch of fields, returning PSFs.

        Returns
        -------
        psf_stack : torch.Tensor, shape [P,N,N], float
        (optional) Uz_stack : torch.Tensor, shape [P,N,N], complex
        """
        if U0_stack is None:
            U0 = self.generate_on_axis_plane_wave()  
        else:
            U0 = torch.as_tensor(U0_stack, device=self.device_)

        # Ensure complex
        if not torch.is_complex(U0):
            U0 = U0.to(torch.complex64)

        if H is None:
            H = self.H
        H = H.to(device=U0.device)
        if not torch.is_complex(H):
            H = H.to(torch.complex64)

        # Apply phase 
        if apply_phase:
            U_lens = phase_mask.apply(U0)
        else:
            U_lens = U0

        # FFT -> multiply -> IFFT 
        U_f = torch.fft.fft2(U_lens)
        U_f_prop = U_f * H  
        Uz = torch.fft.ifft2(U_f_prop)

        psf = torch.abs(Uz) ** 2  # [P,N,N], float

        if normalize:
            eps = 1e-12
            energy = Uz.abs().square().sum(dim=(-2, -1), keepdim=True) + eps
            Uz = Uz * torch.rsqrt(energy)
            psf = Uz.abs().square()

        if return_field:
            return psf, Uz
        else:
            return psf
