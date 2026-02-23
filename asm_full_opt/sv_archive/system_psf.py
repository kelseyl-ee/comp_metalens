import torch
import torch.nn as nn

from generate_waves import GenerateWaves
from phase_mask import PhaseMask
from asm_prop import ASMPropagator
from pixel_map import PixelMap


class SystemPSF(nn.Module):

    def __init__(
        self,
        config,
        phase_init="hyperbolic",
        phase_noise_std=0.0,
        wrap_phase=False,
        use_aperture=True,
        hard_aperture=True,
        test_orientation=False,
    ):
        super().__init__()

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_  = torch.float32

        self.config = config

        self.phase = PhaseMask(
            config,
            init=phase_init,
            noise_std=phase_noise_std,
            wrap_phase=wrap_phase,
            trainable=True,
            use_aperture=use_aperture,
            hard_aperture=hard_aperture,
            test_orientation=test_orientation,
        )

        self.asm = ASMPropagator(config)

        self.pixel_map = PixelMap(config)


    def forward(
        self,
        strategy=None,
        block_size=None,
        hfov_deg=None,
        normalize=True,
        return_field=False,
        return_meta=False,
    ):
        """
        End-to-end: generate plane wave -> apply phase mask -> ASM propagate -> PSFs.

        Returns
        -------
        psf torch.Tensor, shape [1,N,N]
        """
        U0, uv_samples, pixel_to_sample, theta_x, theta_y, kx, ky = self.gw.generate_plane_wave_stack(
            strategy=strategy,
            block_size=block_size,
            hfov_deg=hfov_deg,
        )

        if return_field:
            psf_stack, Uz_stack = self.asm(U0, self.phase, normalize=normalize, return_field=True)
        else:
            psf_stack = self.asm(U0, self.phase, normalize=normalize, return_field=False)

        if return_meta:
            meta = {
                "uv_samples": uv_samples,
                "pixel_to_sample": pixel_to_sample,
                "theta_x": theta_x,
                "theta_y": theta_y,
                "kx": kx,
                "ky": ky,
            }

            if return_field:
                return psf_stack, Uz_stack, meta
            else:
                return psf_stack, meta

        else:
            if return_field:
                return psf_stack, Uz_stack
            else:
                return psf_stack
