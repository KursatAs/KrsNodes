import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from PIL import Image, ImageDraw
import random
import comfy.utils  # For common_upscale

class QwenImageEditLatentKrs:
    """
    A class for processing and snapping image dimensions to specific constraints,
    generating latent tensors, and optionally encoding the image into a VAE latent space.

    This class provides methods to:
    - Snap image dimensions to valid values based on user-defined constraints.
    - Generate latent tensors with specified properties.
    - Preprocess images for compatibility with downstream tasks.
    - Optionally encode images into a VAE latent space.

    The class is designed to work with PyTorch tensors and supports various customization
    options for snapping strategies, fitting modes, and latent tensor generation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The input image tensor to snap dimensions and process."}),
                "width_latent": ("INT", {"default": 1024, "min": 16, "max": 4096, "tooltip": "Desired output width in pixels. Snapped to valid values based on snap_mode."}),
                "height_latent": ("INT", {"default": 1024, "min": 16, "max": 4096, "tooltip": "Desired output height in pixels. Snapped to valid values based on snap_mode."}),
                "locked_axis": (["width", "height", "independent"], {"default": "width", "tooltip": "Which axis to lock when snapping dimensions. 'independent' allows both to snap separately."}),
                "snap_mode": (
                    ["multiple_of_16", "multiple_of_32", "multiple_of_64", "power_of_2_div_16"],
                    {"default": "multiple_of_16", "tooltip": "How to snap dimensions: multiples of 16, 32, 64, or powers of 2 divisible by 16."},
                ),
                "use_exact_input_dims": ("BOOLEAN", {"default": False, "tooltip": "If enabled, use exact input dimensions without snapping."}),
                "use_target_megapixels": ("BOOLEAN", {"default": False, "tooltip": "If enabled, snap dimensions to match the target megapixels instead of user width/height."}),
                "target_megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "tooltip": "Target total megapixels for the output image when use_target_megapixels is enabled."}),
                "fit_mode": (["pad_only", "allow_crop"], {"default": "pad_only", "tooltip": "How to fit the image: pad only (no cropping) or allow cropping to match snapped dimensions."}),
                "snap_strategy": (["up", "down", "nearest"], {"default": "nearest", "tooltip": "Strategy for snapping: round up, down, or to nearest valid value."}),
                "background_color": (["black", "gray", "white"], {"default": "gray", "tooltip": "Background color for padding areas: black, gray, or white."}),
                "latent_scale": ([8, 16], {"default": 8, "tooltip": "Downscale factor for latent space, typically 8 for SD1.5 or 16 for SDXL."}),
                "latent_channels": ("INT", {"default": 4, "min": 1, "max": 32, "tooltip": "Number of channels in the latent tensor. Auto-detected from model if provided."}),
                "latent_mode": (["zeros", "random"], {"default": "zeros", "tooltip": "How to initialize the custom latent: all zeros or random noise."}),
                "latent_seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "tooltip": "Seed for random latent generation. -1 for random seed."}),
                "upscale_method": (["lanczos", "bicubic", "area"], {"default": "lanczos", "tooltip": "Interpolation method for upscaling the image."}),
                "crop": (["disabled", "center"], {"default": "center", "tooltip": "Cropping method during upscaling: disabled or center crop."}),
                "force_5d": ("BOOLEAN", {"default": False, "tooltip": "Force 5D tensor for VAE encoding (for specific models like SVD)."}),
                "normalize_mode": (["minus1_1", "zero_1", "none"], {"default": "none", "tooltip": "Normalization mode for VAE input: [-1,1], [0,1], or none."}),
                "pad_plus_2": ("BOOLEAN", {"default": False, "tooltip": "Add 2 pixels padding to VAE input (for specific requirements)."}),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "Optional model to auto-detect latent channels from."}),
                "vae": ("VAE", {"tooltip": "Optional VAE for encoding the image into latent space. If provided, switches output to VAE latent."}),
            }
        }

    RETURN_TYPES = ("STRING", "LATENT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("preview", "latent", "viz_image", "padded_image")
    FUNCTION = "snap_dimensions"
    CATEGORY = "utils"

    def get_valid_values(self, snap_mode: str, latent_scale: int = 8):
        base_list = []
        if snap_mode == "power_of_2_div_16":
            base_list = [2 ** i for i in range(4, 13) if (2 ** i) % 16 == 0]
        elif snap_mode == "multiple_of_64":
            base_list = list(range(64, 4097, 64))
        elif snap_mode == "multiple_of_32":
            base_list = list(range(32, 4097, 32))
        elif snap_mode == "multiple_of_16":
            base_list = list(range(16, 4097, 16))
        else:
            raise ValueError(f"Invalid snap_mode: {snap_mode}")
        # Filter to ensure divisibility by latent_scale
        return [v for v in base_list if v % latent_scale == 0]

    def snap_value(self, value: int, valid_list, strategy: str) -> int:
        if not valid_list:
            return value
        if strategy == "up":
            candidates = [v for v in valid_list if v >= value]
            return min(candidates) if candidates else max(valid_list)
        elif strategy == "down":
            candidates = [v for v in valid_list if v <= value]
            return max(candidates) if candidates else min(valid_list)
        else:  # nearest
            return min(valid_list, key=lambda v: (abs(v - value), v))

    def validate_megapixel_target(self, target_mp: float, valid_list, ref_w: int, ref_h: int) -> Tuple[bool, str]:
        """
        Validate if target megapixels is feasible.
        Returns: (is_valid, warning_message)
        """
        if not valid_list or len(valid_list) == 0:
            return False, "No valid dimensions available for snapping"

        if target_mp <= 0:
            return False, f"Target megapixels must be > 0, got {target_mp}"

        if not math.isfinite(target_mp):
            return False, f"Target megapixels must be finite, got {target_mp}"

        if ref_w <= 0 or ref_h <= 0:
            return False, f"Input dimensions invalid: {ref_w}×{ref_h}"

        min_val = min(valid_list)
        max_val = max(valid_list)
        min_mp = (min_val * min_val) / 1_000_000
        max_mp = (max_val * max_val) / 1_000_000

        if target_mp < min_mp * 0.5:  # Allow 50% tolerance
            return False, f"Target {target_mp}MP too small. Min achievable: {min_mp:.2f}MP"

        if target_mp > max_mp * 1.5:  # Allow 50% tolerance
            return False, f"Target {target_mp}MP too large. Max achievable: {max_mp:.2f}MP"

        return True, ""

    def check_megapixel_accuracy(self, snapped_w: int, snapped_h: int, target_mp: float, valid_list) -> str:
        """
        Check how close we got to target and return warning if needed.
        """
        actual_mp = (snapped_w * snapped_h) / 1_000_000
        error_pct = abs(actual_mp - target_mp) / target_mp * 100

        warning = ""
        if error_pct > 20:
            warning = f"⚠️ Megapixel target missed by {error_pct:.1f}%: wanted {target_mp}MP, got {actual_mp:.2f}MP"

        # Check if clamped to boundary
        max_val = max(valid_list)
        min_val = min(valid_list)
        if snapped_w == max_val or snapped_h == max_val:
            warning += f" | Clamped to max dimension {max_val}"
        if snapped_w == min_val or snapped_h == min_val:
            warning += f" | Clamped to min dimension {min_val}"

        return warning

    def compute_from_megapixels(self, ref_w: int, ref_h: int, target_mp: float, valid_list, strategy: str) -> Tuple[int, int]:
        # PRE-VALIDATION
        is_valid, error_msg = self.validate_megapixel_target(target_mp, valid_list, ref_w, ref_h)
        if not is_valid:
            print(f"\x1b[31m✗ Megapixel validation failed: {error_msg}\x1b[0m")
            # Fallback to clamping to safe range
            target_mp = max(0.1, min(target_mp, 8.0))
            print(f"\x1b[33m→ Clamped target to {target_mp}MP\x1b[0m")

        input_size = ref_h * ref_w
        target_pixels = target_mp * 1_000_000
        scale_factor = math.sqrt(target_pixels / input_size)

        # Apply scale factor to both dimensions
        new_width = ref_w * scale_factor
        new_height = ref_h * scale_factor

        # Snap both dimensions to valid values
        snapped_w = self.snap_value(round(new_width), valid_list, strategy)
        snapped_h = self.snap_value(round(new_height), valid_list, strategy)

        # POST-VALIDATION
        warning = self.check_megapixel_accuracy(snapped_w, snapped_h, target_mp, valid_list)
        if warning:
            print(f"\x1b[33m{warning}\x1b[0m")

        return snapped_w, snapped_h

    def generate_preview_image(self, ref_w: int, ref_h: int, out_w: int, out_h: int) -> torch.Tensor:
        canvas_w, canvas_h = 512, 512
        scale_factor = min(canvas_w / max(1, ref_w, out_w), canvas_h / max(1, ref_h, out_h))
        out_scaled = (int(out_w * scale_factor), int(out_h * scale_factor))
        ref_scaled = (int(ref_w * scale_factor), int(ref_h * scale_factor))
        img = Image.new("RGB", (canvas_w, canvas_h), color=(30, 30, 30))
        draw = ImageDraw.Draw(img)

        out_x = (canvas_w - out_scaled[0]) // 2
        out_y = (canvas_h - out_scaled[1]) // 2
        draw.rectangle([out_x, out_y, out_x + out_scaled[0], out_y + out_scaled[1]], outline="red", width=2)
        draw.text((out_x + 4, out_y + 4), f"Out: {out_w}x{out_h}", fill="red")

        pad_x = (out_w - ref_w) // 2
        pad_y = (out_h - ref_h) // 2
        ref_x = out_x + int(pad_x * scale_factor)
        ref_y = out_y + int(pad_y * scale_factor)

        draw.rectangle([ref_x, ref_y, ref_x + ref_scaled[0], ref_y + ref_scaled[1]], outline="#66CCFF", width=2)
        draw.text((ref_x + 4, ref_y + 4), f"Ref: {ref_w}x{ref_h}", fill="#66CCFF")

        img_np = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0).contiguous()

    def _detect_model_latent_channels(self, model) -> Tuple[int, str]:
        try:
            fmt = getattr(model, 'latent_format', None)
            if fmt is None and hasattr(model, 'model'):
                fmt = getattr(model.model, 'latent_format', None)
            if fmt is None:
                return -1, 'manual'

            if hasattr(fmt, 'channels'):
                ch = int(getattr(fmt, 'channels'))
                if ch > 0:
                    return ch, 'model'

            for attr in ('latents_mean', 'latents_std'):
                if hasattr(fmt, attr):
                    t = getattr(fmt, attr)
                    if isinstance(t, torch.Tensor) and t.ndim >= 2:
                        ch = int(t.shape[1])
                        if ch > 0:
                            return ch, 'model'

            if hasattr(fmt, 'latent_channels'):
                ch = int(getattr(fmt, 'latent_channels'))
                if ch > 0:
                    return ch, 'model'
        except Exception:
            pass
        return -1, 'manual'

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """Ensure image is [1, height, width, 3] (HWC, RGB, float32 [0,1])."""
        print("\x1b[32m→ " + f"Input image shape: {image.shape}" + "\x1b[0m")

        # Reduce dimensionality in-place where possible
        if image.ndim > 4:
            while image.ndim > 4:
                image = image.squeeze(0)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.shape[0] > 1:
            image = image[:1]  # Slicing creates a view, not a copy

        if image.shape[-1] not in [1, 3, 4]:
            raise ValueError(f"Unsupported channels: {image.shape[-1]}. Expected 1, 3, or 4.")

        # Handle channel conversion efficiently
        needs_copy = False
        if image.shape[-1] == 4:
            image = image[..., :3]  # Slicing creates a view
        elif image.shape[-1] == 1:
            image = image.repeat(1, 1, 1, 3)  # This must copy
            needs_copy = True

        # Type conversion
        if image.dtype != torch.float32:
            image = image.float()  # This creates a copy
            needs_copy = True

        # Clamp in-place if possible
        if not needs_copy and image.is_contiguous():
            image = image.clamp_(0, 1)  # In-place clamp
        else:
            image = torch.clamp(image, 0, 1)

        print("\x1b[32m→ " + f"Preprocessed image shape: {image.shape}" + "\x1b[0m")
        return image.contiguous()

    def snap_dimensions(
        self,
        image: torch.Tensor,
        width_latent: int,
        height_latent: int,
        locked_axis: str,
        snap_mode: str,
        use_exact_input_dims: bool,
        use_target_megapixels: bool,
        target_megapixels: float,
        fit_mode: str,
        snap_strategy: str,
        background_color: str,
        latent_scale: int,
        latent_channels: int,
        latent_mode: str,
        latent_seed: int,
        upscale_method: str,
        crop: str,
        force_5d: bool,
        normalize_mode: str,
        pad_plus_2: bool,
        model=None,
        vae=None,
    ) -> Tuple[str, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        device = image.device

        # Preprocess to [1, H, W, 3]
        image = self.preprocess_image(image)
        ref_height, ref_width = image.shape[1:3]
        channels = 3
        ref_tensor = image

        # Store original dimensions for visualization
        original_ref_height, original_ref_width = ref_height, ref_width

        aspect_ratio = float(ref_height) / float(ref_width) if ref_width != 0 else 1.0

        valid_values = [] if use_exact_input_dims else self.get_valid_values(snap_mode, latent_scale)
        if use_exact_input_dims:
            snapped_width = ref_width
            snapped_height = ref_height
        else:
            valid_values = self.get_valid_values(snap_mode, latent_scale)

            if use_target_megapixels:
                snapped_width, snapped_height = self.compute_from_megapixels(ref_width, ref_height, target_megapixels, valid_values, snap_strategy)
            else:
                if locked_axis == "width":
                    snapped_width = self.snap_value(width_latent, valid_values, snap_strategy)
                    snapped_height = self.snap_value(round(snapped_width * aspect_ratio), valid_values, snap_strategy)
                elif locked_axis == "height":
                    snapped_height = self.snap_value(height_latent, valid_values, snap_strategy)
                    snapped_width = self.snap_value(round(snapped_height / aspect_ratio), valid_values, snap_strategy)
                else:
                    snapped_width = self.snap_value(width_latent, valid_values, snap_strategy)
                    snapped_height = self.snap_value(height_latent, valid_values, snap_strategy)

            # Ensure VAE-compatible dimensions (must be divisible by latent_scale)
            snapped_width = self.snap_value(snapped_width, valid_values, snap_strategy)
            snapped_height = self.snap_value(snapped_height, valid_values, snap_strategy)

        need_resize = (snapped_width != ref_width or snapped_height != ref_height) and fit_mode == "allow_crop"
        if fit_mode == "pad_only":
            if snapped_width < ref_width:
                snapped_width = self.snap_value(ref_width, valid_values, "up")
            if snapped_height < ref_height:
                snapped_height = self.snap_value(ref_height, valid_values, "up")

        detected_ch, source = self._detect_model_latent_channels(model)
        effective_channels = detected_ch if detected_ch > 0 else latent_channels
        channel_source = 'model' if detected_ch > 0 else 'manual'

        latent_divisible = (snapped_width % latent_scale == 0) and (snapped_height % latent_scale == 0)

        preview = (
            f"Ref: {ref_width}x{ref_height} | "
            f"Aspect: {aspect_ratio:.4f} → {snapped_height / snapped_width:.4f} | "
            f"Output: {snapped_width}x{snapped_height} | "
            f"Snap: {snap_mode}/{snap_strategy} | Fit: {fit_mode} | "
            f"Exact: {'ON' if use_exact_input_dims else 'OFF'} | "
            f"Target MP: {'ON' if use_target_megapixels else 'OFF'} ({target_megapixels}) | "
            f"Latent: C{effective_channels}@/{latent_scale} (divisible={latent_divisible}, src={channel_source}) | "
            f"Upscale: {upscale_method}, Crop: {crop} | "
            f"VAE Input: {'Yes' if vae is not None else 'No'} | Img: Scaled to {snapped_width}x{snapped_height}"
            f"{' (5D)' if force_5d else ''}, Norm: {normalize_mode}"
            f"{', Pad+2' if pad_plus_2 else ''}"
        )

        # Process image - optimized to reduce copies
        if need_resize:
            # Permute creates a view (no copy), upscale creates new tensor (unavoidable)
            ref_tensor_chw = ref_tensor.permute(0, 3, 1, 2)
            ref_tensor_chw = comfy.utils.common_upscale(ref_tensor_chw, snapped_width, snapped_height, upscale_method, crop)
            padded_tensor = ref_tensor_chw.permute(0, 2, 3, 1)
            ref_width, ref_height = snapped_width, snapped_height
        else:
            if ref_height == snapped_height and ref_width == snapped_width:
                # No need to clone if we're not modifying it later
                padded_tensor = ref_tensor
            else:
                bg_val = 0.0 if background_color == "black" else 1.0 if background_color == "white" else 0.5
                padded_tensor = torch.full((1, snapped_height, snapped_width, channels), bg_val, dtype=torch.float32, device=device)
                pix_offset_y = (snapped_height - ref_height) // 2
                pix_offset_x = (snapped_width - ref_width) // 2
                y0, x0 = 0, 0
                if pix_offset_y < 0:
                    y0 = -pix_offset_y
                    pix_offset_y = 0
                if pix_offset_x < 0:
                    x0 = -pix_offset_x
                    pix_offset_x = 0
                paste_h = min(ref_height - y0, snapped_height - pix_offset_y)
                paste_w = min(ref_width - x0, snapped_width - pix_offset_x)
                if paste_h > 0 and paste_w > 0:
                    # In-place assignment to pre-allocated tensor
                    padded_tensor[:, pix_offset_y:pix_offset_y + paste_h, pix_offset_x:pix_offset_x + paste_w, :] = \
                        ref_tensor[:, y0:y0 + paste_h, x0:x0 + paste_w, :]

        print("\x1b[32m→ " + f"Padded tensor shape: {padded_tensor.shape}" + "\x1b[0m")

        # Custom latent
        latent_h = max(1, snapped_height // latent_scale)
        latent_w = max(1, snapped_width // latent_scale)
        if latent_mode == "zeros":
            latent_tensor = torch.zeros((1, effective_channels, latent_h, latent_w), dtype=torch.float32, device=device)
        else:
            generator = torch.Generator(device=device)
            if latent_seed < 0:
                auto_seed = random.randint(0, 2147483647)
                generator.manual_seed(auto_seed)
            else:
                generator.manual_seed(latent_seed)
            latent_tensor = torch.randn((1, effective_channels, latent_h, latent_w), generator=generator, device=device)
            latent_tensor = (latent_tensor - torch.mean(latent_tensor)) / (torch.std(latent_tensor) + 1e-8)
        latent_output = {"samples": latent_tensor}

        # VAE encoding - optimized to reduce copies
        vae_latent_output = None
        if vae is not None:
            with torch.no_grad():
                expected_shape = (1, snapped_height, snapped_width, 3)
                if padded_tensor.shape != expected_shape:
                    raise ValueError(f"Padded tensor shape {padded_tensor.shape} does not match expected {expected_shape}")

                # Permute to CHW format (creates view, not copy)
                pixels = padded_tensor.permute(0, 3, 1, 2)  # [1,3,H,W]

                # Only upscale if dimensions changed (avoid redundant operation)
                if pixels.shape[2] != snapped_height or pixels.shape[3] != snapped_width:
                    pixels = comfy.utils.common_upscale(pixels, snapped_width, snapped_height, upscale_method, crop)
                else:
                    pixels = pixels.contiguous()  # Ensure contiguous only when needed

                # Apply padding if needed
                if pad_plus_2:
                    pixels = F.pad(pixels, (1, 1, 1, 1), mode='constant', value=0)
                    print("\x1b[32m→ " + f"Padded VAE input shape: {pixels.shape}" + "\x1b[0m")

                # Normalize in-place where possible
                if normalize_mode == "minus1_1":
                    pixels = pixels * 2.0 - 1.0
                    pixels.clamp_(-1.0, 1.0)  # In-place clamp
                elif normalize_mode == "zero_1":
                    pixels.clamp_(0.0, 1.0)  # In-place clamp
                # else: none - no operation needed

                # Convert to HWC (movedim creates view)
                pixels = pixels.movedim(1, -1)  # [1,H,W,3]

                # Move to VAE's device only if needed
                vae_device = next(vae.parameters()).device if hasattr(vae, 'parameters') else device
                if pixels.device != vae_device:
                    pixels = pixels.to(vae_device)

                print("\x1b[32m→ " + f"VAE input shape: {pixels.shape} (scaled to {snapped_width}x{snapped_height}{'+2' if pad_plus_2 else ''})" + "\x1b[0m")
                try:
                    if pixels.numel() > 0:
                        print("\x1b[32m→ " + f"VAE input stats: min={pixels.min().item():.4f}, max={pixels.max().item():.4f}, mean={pixels.mean().item():.4f}" + "\x1b[0m")
                except Exception:
                    pass
                print("\x1b[32m→ " + f"VAE input device: {pixels.device}, dtype: {pixels.dtype}" + "\x1b[0m")
                print("\x1b[32m→ " + f"VAE device: {vae_device}" + "\x1b[0m")

                if force_5d:
                    # Note: repeat() creates a copy - unavoidable for 5D
                    pixels_5d = pixels.unsqueeze(2).repeat(1, 1, 3, 1, 1)  # [1,H,3,W,3]
                    print("\x1b[32m→ " + f"Force 5D VAE input shape: {pixels_5d.shape}" + "\x1b[0m")
                    try:
                        # Slice is redundant since last dim is already 3, but keeping for compatibility
                        vae_output = vae.encode(pixels_5d[:, :, :, :, :3])
                        if isinstance(vae_output, torch.Tensor):
                            vae_latent = vae_output
                            if vae_latent.ndim > 4 and vae_latent.shape[2] == 1:
                                vae_latent = vae_latent.squeeze(2)
                        else:
                            vae_latent = vae_output.latent_dist.sample()
                        vae_latent_output = {"samples": vae_latent}
                        print("\x1b[32m→ " + f"VAE output shape: {vae_latent.shape}" + "\x1b[0m")
                    except Exception as e:
                        print("\x1b[32m→ " + f"VAE encoding (5D) failed: {str(e)}" + "\x1b[0m")
                        preview += f" | VAE Error (5D): {str(e)}"
                        vae_latent_output = latent_output
                else:
                    try:
                        # Slice to ensure 3 channels (creates view if already 3)
                        vae_output = vae.encode(pixels[:, :, :, :3])
                        if isinstance(vae_output, torch.Tensor):
                            vae_latent = vae_output
                            if vae_latent.ndim > 4 and vae_latent.shape[2] == 1:
                                vae_latent = vae_latent.squeeze(2)
                        else:
                            vae_latent = vae_output.latent_dist.sample()
                        vae_latent_output = {"samples": vae_latent}
                        print("\x1b[32m→ " + f"VAE output shape: {vae_latent.shape}" + "\x1b[0m")
                    except Exception as e:
                        print("\x1b[32m→ " + f"VAE encoding failed: {str(e)}" + "\x1b[0m")
                        preview += f" | VAE Error: {str(e)}"
                        vae_latent_output = latent_output

        common_latent = vae_latent_output if vae_latent_output is not None else latent_output

        return (
            preview,
            common_latent,
            self.generate_preview_image(original_ref_width, original_ref_height, snapped_width, snapped_height),
            padded_tensor,
        )

# Register the node
NODE_CLASS_MAPPINGS = {
    "QwenImageEditLatentKrs": QwenImageEditLatentKrs
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageEditLatentKrs": "Qwen Image Edit Latent Krs"
}
