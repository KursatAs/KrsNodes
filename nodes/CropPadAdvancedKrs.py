import torch
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


class CropPadAdvancedKrs:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The input image to crop and/or pad."}),
                "padding_x": ("INT", {"default": 256, "min": 1,
                                      "tooltip": "Horizontal padding amount per side when not using custom padding."}),
                "padding_y": ("INT", {"default": 256, "min": 1,
                                      "tooltip": "Vertical padding amount per side when not using custom padding."}),
                "use_custom_padding": ("BOOLEAN", {"default": False,
                                                   "tooltip": "Enable custom padding for each side independently. Allows negative values for cropping."}),
                "padding_top": ("INT", {"default": 0, "min": -4096,
                                        "tooltip": "Custom padding for the top side. Negative values crop from the top."}),
                "padding_bottom": ("INT", {"default": 0, "min": -4096,
                                           "tooltip": "Custom padding for the bottom side. Negative values crop from the bottom."}),
                "padding_left": ("INT", {"default": 0, "min": -4096,
                                         "tooltip": "Custom padding for the left side. Negative values crop from the left."}),
                "padding_right": ("INT", {"default": 0, "min": -4096,
                                          "tooltip": "Custom padding for the right side. Negative values crop from the right."}),
                "fit_to_target_resolution": ("BOOLEAN", {"default": False,
                                                         "tooltip": "Fit the padded image to a target resolution by centering it within the target canvas."}),
                "target_width": ("INT", {"default": 1024, "min": 1,
                                         "tooltip": "Target width for fitting the image when fit_to_target_resolution is enabled."}),
                "target_height": ("INT", {"default": 1024, "min": 1,
                                          "tooltip": "Target height for fitting the image when fit_to_target_resolution is enabled."}),
                "scale_down": ("BOOLEAN", {"default": False,
                                           "tooltip": "Scale down the image and padding if the total size exceeds the target resolution."}),
                "pad_color": (["white", "black", "gray", "red", "green", "blue", "yellow"], {"default": "white",
                                                                                             "tooltip": "Color for solid padding areas when transparent_padding is disabled."}),
                "transparent_padding": ("BOOLEAN", {"default": False,
                                                    "tooltip": "Use transparent padding instead of a solid color. Overrides pad_color."}),
                "feather_padding": ("BOOLEAN", {"default": False,
                                                "tooltip": "Apply feathering to the padding edges for smooth blending with the original image."}),
                "feather_radius": ("INT", {"default": 16, "min": 1, "max": 4096,
                                           "tooltip": "Radius of the feathering effect in pixels."}),
                "feather_type": (["linear", "gaussian"], {"default": "linear",
                                                          "tooltip": "Choose feathering method: linear ramp or gaussian blur."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "IMAGE_BLENDED")
    FUNCTION = "process_image"
    CATEGORY = "image/processing"

    def process_image(self, image, padding_x, padding_y, use_custom_padding,
                      padding_top, padding_bottom, padding_left, padding_right,
                      fit_to_target_resolution, target_width, target_height, scale_down,
                      pad_color, transparent_padding, feather_padding=False, feather_radius=16, feather_type="linear"):
        if not isinstance(image, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if image.dim() == 4:
            image = image[0]
        if image.dim() != 3:
            raise ValueError(f"Expected 3D tensor (H, W, C), got shape {image.shape}")

        h, w, c = image.shape
        if c not in (1, 3, 4):
            raise ValueError("Image must have 1 (L), 3 (RGB), or 4 (RGBA) channels")

        image_np = (image.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        pil_mode = "L" if c == 1 else "RGB" if c == 3 else "RGBA"
        img_pil = Image.fromarray(image_np, mode=pil_mode)

        if transparent_padding or pil_mode != "RGBA":
            img_pil = img_pil.convert("RGBA")

        if use_custom_padding:
            crop_l = max(0, -padding_left)
            crop_r = max(0, -padding_right)
            crop_t = max(0, -padding_top)
            crop_b = max(0, -padding_bottom)

            crop_box = (
                crop_l,
                crop_t,
                img_pil.width - crop_r,
                img_pil.height - crop_b
            )

            if crop_box[0] >= crop_box[2] or crop_box[1] >= crop_box[3]:
                raise ValueError("Cropping removed entire image due to negative padding")

            img_pil = img_pil.crop(crop_box)

            padding_left = max(0, padding_left)
            padding_right = max(0, padding_right)
            padding_top = max(0, padding_top)
            padding_bottom = max(0, padding_bottom)

            canvas_width = img_pil.width + padding_left + padding_right
            canvas_height = img_pil.height + padding_top + padding_bottom

            paste_x = padding_left
            paste_y = padding_top

        else:
            canvas_width = img_pil.width + (padding_x * 2)
            canvas_height = img_pil.height + (padding_y * 2)

            paste_x = padding_x
            paste_y = padding_y

        needs_final_centering = False
        temp_canvas_width = 0
        temp_canvas_height = 0
        final_paste_x = 0
        final_paste_y = 0

        if fit_to_target_resolution:
            total_width_needed = canvas_width
            total_height_needed = canvas_height

            if scale_down and (total_width_needed > target_width or total_height_needed > target_height):
                scale_factor = min(target_width / total_width_needed, target_height / total_height_needed)

                new_img_size = (int(img_pil.width * scale_factor), int(img_pil.height * scale_factor))
                img_pil = img_pil.resize(new_img_size, Image.Resampling.LANCZOS)

                paste_x = int(paste_x * scale_factor)
                paste_y = int(paste_y * scale_factor)

                canvas_width = int(canvas_width * scale_factor)
                canvas_height = int(canvas_height * scale_factor)

            final_paste_x = (target_width - canvas_width) // 2
            final_paste_y = (target_height - canvas_height) // 2

            needs_final_centering = True
            temp_canvas_width = canvas_width
            temp_canvas_height = canvas_height
            canvas_width = target_width
            canvas_height = target_height

        canvas_mode = "RGBA" if transparent_padding else "RGB"
        if transparent_padding:
            canvas_color = (0, 0, 0, 0)
        else:
            try:
                canvas_color = Image.new("RGB", (1, 1), pad_color).getpixel((0, 0))
            except:
                canvas_color = (0, 0, 0)

        actual_img_width = img_pil.width
        actual_img_height = img_pil.height

        img_x0 = paste_x
        img_y0 = paste_y

        if needs_final_centering:
            temp_canvas = Image.new(canvas_mode, (temp_canvas_width, temp_canvas_height), canvas_color)
            mask = img_pil.getchannel('A') if canvas_mode == "RGBA" else None
            temp_canvas.paste(img_pil, (paste_x, paste_y), mask)

            canvas = Image.new(canvas_mode, (canvas_width, canvas_height), canvas_color)
            canvas.paste(temp_canvas, (final_paste_x, final_paste_y))

            img_x0 = final_paste_x + paste_x
            img_y0 = final_paste_y + paste_y
        else:
            canvas = Image.new(canvas_mode, (canvas_width, canvas_height), canvas_color)
            mask = img_pil.getchannel('A') if canvas_mode == "RGBA" else None
            canvas.paste(img_pil, (paste_x, paste_y), mask)

        mask_img = np.zeros((canvas_height, canvas_width), dtype=np.float32)
        x0, y0 = img_x0, img_y0
        x1, y1 = img_x0 + actual_img_width, img_y0 + actual_img_height
        yy, xx = np.meshgrid(np.arange(canvas_height), np.arange(canvas_width), indexing='ij')
        outside_img = (xx < x0) | (xx >= x1) | (yy < y0) | (yy >= y1)
        mask_img[outside_img] = 255

        if feather_padding and feather_radius > 0:
            top_pad = img_y0
            bottom_pad = canvas_height - (img_y0 + actual_img_height)
            left_pad = img_x0
            right_pad = canvas_width - (img_x0 + actual_img_width)

            mask_img[:] = 0
            if top_pad > 0:
                mask_img[0:top_pad, :] = 255
            if bottom_pad > 0:
                mask_img[-bottom_pad:, :] = 255
            if left_pad > 0:
                mask_img[:, 0:left_pad] = 255
            if right_pad > 0:
                mask_img[:, -right_pad:] = 255

            if feather_type == "linear":
                inside_img = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)

                if top_pad > 0:
                    dist_top = yy - y0
                    feather_band = inside_img & (dist_top < feather_radius)
                    mask_img[feather_band] = np.maximum(mask_img[feather_band], 255 * (1 - dist_top[feather_band] / feather_radius))

                if bottom_pad > 0:
                    dist_bottom = y1 - 1 - yy
                    feather_band = inside_img & (dist_bottom < feather_radius)
                    mask_img[feather_band] = np.maximum(mask_img[feather_band], 255 * (1 - dist_bottom[feather_band] / feather_radius))

                if left_pad > 0:
                    dist_left = xx - x0
                    feather_band = inside_img & (dist_left < feather_radius)
                    mask_img[feather_band] = np.maximum(mask_img[feather_band], 255 * (1 - dist_left[feather_band] / feather_radius))

                if right_pad > 0:
                    dist_right = x1 - 1 - xx
                    feather_band = inside_img & (dist_right < feather_radius)
                    mask_img[feather_band] = np.maximum(mask_img[feather_band], 255 * (1 - dist_right[feather_band] / feather_radius))
            elif feather_type == "gaussian":
                sigma = feather_radius / 3.0
                mask_img = gaussian_filter(mask_img.astype(np.float32), sigma=sigma)
                mask_img = np.clip(mask_img, 0, 255)

        canvas_np = np.array(canvas).astype(np.float32) / 255.0
        pad_arr = np.array(canvas_color, dtype=np.float32) / 255.0
        if pad_arr.shape[0] != canvas_np.shape[2]:
            pad_arr = np.pad(pad_arr, (0, canvas_np.shape[2] - pad_arr.shape[0]), mode='constant')
        mask_norm = (mask_img / 255.0)[..., None]
        blended_np = canvas_np * (1 - mask_norm) + pad_arr * mask_norm

        blended_tensor = torch.from_numpy(blended_np).unsqueeze(0).float()
        mask_tensor = torch.from_numpy((mask_img / 255.0).astype(np.float32)).unsqueeze(0).float()
        canvas_tensor = torch.from_numpy(canvas_np).unsqueeze(0).float()
        return (canvas_tensor, mask_tensor, blended_tensor)


NODE_CLASS_MAPPINGS = {
    "CropPadAdvancedKrs": CropPadAdvancedKrs,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CropPadAdvancedKrs": "🧩 Crop Pad Advanced Krs",
}
