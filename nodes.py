import logging
import math
from enum import Enum
from typing import Optional

import torch
from torch import Tensor, nn

from comfy.model_sampling import EDM, V_PREDICTION
from comfy.supported_models_base import BASE
from comfy.ldm.modules.diffusionmodules.util import make_beta_schedule


class ModelSamplingMode(str, Enum):
    EDM = "edm"
    V_PRED = "v_prediction"


class ModelSamplingWaifuEDM(nn.Module):
    sigmas: Tensor
    log_sigmas: Tensor

    def __init__(self, model_config: Optional[BASE] = None):
        super().__init__()
        if model_config is not None:
            sampling_settings = model_config.sampling_settings
        else:
            sampling_settings = {}

        sigma_min = sampling_settings.get("sigma_min", 0.001)
        sigma_max = sampling_settings.get("sigma_max", 1000.0)
        sigma_data = sampling_settings.get("sigma_data", 1.0)
        use_tan = sampling_settings.get("use_tan", True)
        tan_scaling = sampling_settings.get("tan_scaling", 1.6)

        self.set_parameters(sigma_min, sigma_max, sigma_data, use_tan, tan_scaling)

    def set_parameters(self, sigma_min, sigma_max, sigma_data, use_tan, tan_scaling: float = 1.0):
        self.sigma_data = sigma_data
        # needs to be float64 to avoid overflow in later calculations
        half_pi_t = torch.acos(torch.zeros(1, dtype=torch.float64)).squeeze(0)

        if use_tan:
            # needs to be float64 or this will overflow
            sigmas = (
                torch.tan(torch.linspace(sigma_min, half_pi_t - sigma_min, 1000, dtype=torch.float64))
                * tan_scaling
            )
            sigmas = sigmas.to(torch.float32)
        else:
            sigmas = torch.linspace(math.log(sigma_min), math.log(sigma_max - sigma_min), 1000).exp()

        self.register_buffer("sigmas", sigmas)  # for compatibility with some schedulers
        self.register_buffer("log_sigmas", sigmas.log())

    @property
    def sigma_min(self) -> Tensor:
        return self.sigmas[0]

    @property
    def sigma_max(self) -> Tensor:
        return self.sigmas[-1]

    """
    This is actually models's input for time perception, so it's "timestep" only for sdxl.
    It's actually c_noise
    """

    def timestep(self, sigma):
        # COSXL
        c_noise = 0.25 * sigma.log()
        return c_noise

    """
    Warning: sigma() is not exact for this and so schedule reinterpretation methods that use it may be inaccurate
    """

    def sigma(self, timestep):
        if math.isclose(timestep, self.timestep(self.sigma_min), abs_tol=1e-6):
            return self.sigma_min
        elif math.isclose(timestep, self.timestep(self.sigma_max), abs_tol=1e-6):
            return self.sigma_max
        else:
            return None

    def percent_to_sigma(self, percent):
        return None



class ModelSamplingCosXLTC(nn.Module):
    sigmas: Tensor
    log_sigmas: Tensor

    def __init__(self, model_config: Optional[BASE] = None):
        super().__init__()
        if model_config is not None:
            sampling_settings = model_config.sampling_settings
        else:
            sampling_settings = {}

        sigma_min = sampling_settings.get("sigma_min", 0.001)
        sigma_max = sampling_settings.get("sigma_max", 1000.0)
        sigma_data = sampling_settings.get("sigma_data", 1.0)
        use_tan = sampling_settings.get("use_tan", True)
        tan_scaling = sampling_settings.get("tan_scaling", 1.6)

        self.set_parameters(sigma_min, sigma_max, sigma_data, use_tan, tan_scaling)

    def set_parameters(self, sigma_min, sigma_max, sigma_data, use_tan, tan_scaling: float = 1.6):
        cosxl_tan_sigmas = tan_scaling * torch.tan(
            torch.linspace(0.001, torch.acos(torch.zeros(1)).item() - 0.001, 1000)
        )

        linear_start = 0.00085
        linear_end = 0.012
        beta_schedule = "linear"
        timesteps = 1000
        cosine_s = 8e-3
        betas = make_beta_schedule(
            beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        (timesteps,) = betas.shape
        num_timesteps = int(timesteps)
        linear_start = linear_start
        linear_end = linear_end

        xl10_sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5

        threshold_sigma_max = xl10_sigmas[-1]
        threshold_sigma = threshold_sigma_max

        # sigmas is in forward diffusion order
        valid_indices_cosxl = torch.nonzero(cosxl_tan_sigmas >= threshold_sigma, as_tuple=True)[0]
        cosxl_start_index = valid_indices_cosxl[0] if len(valid_indices_cosxl) > 0 else len(cosxl_tan_sigmas)

        valid_indices_xl10 = torch.nonzero(xl10_sigmas <= threshold_sigma, as_tuple=True)[0]
        xl10_end_index = valid_indices_xl10[-1] if len(valid_indices_xl10) > 0 else -1

        sigmas_full = torch.cat([xl10_sigmas[: xl10_end_index + 1], cosxl_tan_sigmas[cosxl_start_index:]])
        self.register_buffer("sigmas", sigmas_full)
        self.register_buffer("log_sigmas", sigmas_full.log())

        self.cosxl_start_index = cosxl_start_index
        self.xl10_end_index = xl10_end_index
        self.real_threshold_sigma = sigmas_full[xl10_end_index]
        self.sigma_data = 1.0

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    """
    This is actually models's input for time perception, so it's "timestep" only for sdxl.
    It's actually c_noise
    """

    def timestep(self, sigma):
        # COSXL
        c_noise = 0.25 * sigma.log()
        return c_noise

    """
    Warning: sigma() is not exact for this and so schedule reinterpretation methods that use it may be inaccurate
    """

    def sigma(self, timestep):
        if math.isclose(timestep, self.timestep(self.sigma_min), abs_tol=1e-6):
            return self.sigma_min
        elif math.isclose(timestep, self.timestep(self.sigma_max), abs_tol=1e-6):
            return self.sigma_max
        else:
            return torch.exp(timestep / 0.25) # inexact but it'll do i guess

    def percent_to_sigma(self, percent):
        index = round(percent * 1000) - 1
        return self.sigmas[index]


class ModelSamplingWaifuDiffusionV:
    """
    Class methods
    -------------
    INPUT_TYPES (dict):
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.
    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "sampling": ([x.value for x in list(ModelSamplingMode)],),
                "schedule": (["Tan", "XLTC", "XL10"], {"default":"XLTC"}),
                "sigma_min": (
                    "FLOAT",
                    {"default": 0.001, "min": 0.0, "max": 1000.0, "step": 0.001, "round": False},
                ),
                "sigma_max": (
                    "FLOAT",
                    {"default": 1000.0, "min": 0.0, "max": 1000.0, "step": 0.001, "round": False},
                ),
                "tan_scaling": (
                    "FLOAT",
                    {"default": 1.6, "min": 1.0, "max": 2.0, "step": 0.01, "round": False},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "WaifuDiffusionV"

    def patch(
        self, model, sampling: str, schedule: str, sigma_min: float, sigma_max: float, tan_scaling: float
    ) -> tuple:
        m = model.clone()
        sigma_data = 1.0

        match schedule:
            case "Tan":
                use_tan = True
                model_class = ModelSamplingWaifuEDM
            case "XLTC":
                use_tan = True
                model_class = ModelSamplingCosXLTC
            case "XL10":
                use_tan = False
                model_class = ModelSamplingWaifuEDM
            case _:  # default
                logging.warning(f"Unknown schedule: {schedule}")
                use_tan = False
                model_class = ModelSamplingWaifuEDM

        match sampling:
            case ModelSamplingMode.EDM:
                sampling_mode = EDM
            case ModelSamplingMode.V_PRED:
                sampling_mode = V_PREDICTION
            case _:
                raise ValueError(f"Unknown sampling mode: {sampling}")

        class ModelSamplingWDV(model_class, sampling_mode):
            pass

        model_sampling = ModelSamplingWDV(model.model.model_config)
        model_sampling.set_parameters(sigma_min, sigma_max, sigma_data, use_tan, tan_scaling)
        m.add_object_patch("model_sampling", model_sampling)
        return (m,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ModelSamplingWaifuDiffusionV": ModelSamplingWaifuDiffusionV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelSamplingWaifuDiffusionV": "ModelSamplingWaifuDiffusionV",
}