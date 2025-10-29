import torch
import numpy as np
from PIL import Image

class CropPadAdvancedKrs:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The input image to crop and/or pad."}),
                "padding_x": ("INT", {"default": 256, "min": 1, "tooltip": "Horizontal padding amount per side when not using custom padding."}),
                "padding_y": ("INT", {"default": 256, "min": 1, "tooltip": "Vertical padding amount per side when not using custom padding."}),
                "use_custom_padding": ("BOOLEAN", {"default": False, "tooltip": "Enable custom padding for each side independently. Allows negative values for cropping."}),
                "padding_top": ("INT", {"default": 0, "min": -4096, "tooltip": "Custom padding for the top side. Negative values crop from the top."}),
                "padding_bottom": ("INT", {"default": 0, "min": -4096, "tooltip": "Custom padding for the bottom side. Negative values crop from the bottom."}),
                "padding_left": ("INT", {"default": 0, "min": -4096, "tooltip": "Custom padding for the left side. Negative values crop from the left."}),
                "padding_right": ("INT", {"default": 0, "min": -4096, "tooltip": "Custom padding for the right side. Negative values crop from the right."}),
                "fit_to_target_resolution": ("BOOLEAN", {"default": False, "tooltip": "Fit the padded image to a target resolution by centering it within the target canvas."}),
                "target_width": ("INT", {"default": 1024, "min": 1, "tooltip": "Target width for fitting the image when fit_to_target_resolution is enabled."}),
                "target_height": ("INT", {"default": 1024, "min": 1, "tooltip": "Target height for fitting the image when fit_to_target_resolution is enabled."}),
                "scale_down": ("BOOLEAN", {"default": False, "tooltip": "Scale down the image and padding if the total size exceeds the target resolution."}),
                "pad_color": (["white", "black", "gray", "red", "green", "blue", "yellow"], {"default": "white", "tooltip": "Color for solid padding areas when transparent_padding is disabled."}),
                "transparent_padding": ("BOOLEAN", {"default": False, "tooltip": "Use transparent padding instead of a solid color. Overrides pad_color."}),
                "feather_padding": ("BOOLEAN", {"default": False, "tooltip": "Apply feathering to the padding edges for smooth blending with the original image."}),
                "feather_radius": ("INT", {"default": 16, "min": 1, "max": 4096, "tooltip": "Radius of the feathering effect in pixels."}),
                "feather_distance_type": (["manhattan", "smooth"], {"default": "smooth", "tooltip": "Distance calculation for feathering: 'manhattan' like other similar nodes does, 'smooth' for nicely smoothed rounded corners."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "IMAGE_BLENDED")
    FUNCTION = "process_image"
    CATEGORY = "image/processing"

    def process_image(self, image, padding_x, padding_y, use_custom_padding,
                      padding_top, padding_bottom, padding_left, padding_right,
                      fit_to_target_resolution, target_width, target_height, scale_down,
                      pad_color, transparent_padding, feather_padding=False, feather_radius=16, feather_distance_type="smooth"):
        # --- Validate image tensor ---
        if not isinstance(image, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if image.dim() == 4:
            image = image[0]
        if image.dim() != 3:
            raise ValueError(f"Expected 3D tensor (H, W, C), got shape {image.shape}")

        h, w, c = image.shape
        if c not in (1, 3, 4):
            raise ValueError("Image must have 1 (L), 3 (RGB), or 4 (RGBA) channels")

        # Clamp to [0, 1] and convert to uint8
        image_np = (image.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        pil_mode = "L" if c == 1 else "RGB" if c == 3 else "RGBA"
        img_pil = Image.fromarray(image_np, mode=pil_mode)

        if transparent_padding or pil_mode != "RGBA":
            img_pil = img_pil.convert("RGBA")

        # Map parameter names to internal variable names for compatibility
        canvas_width = padding_x
        canvas_height = padding_y

        # --- Handle cropping for negative padding ---
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
            # When not using custom padding, padding_x and padding_y represent padding amount per side
            # Calculate canvas size: image size + padding on both sides
            canvas_width = img_pil.width + (padding_x * 2)
            canvas_height = img_pil.height + (padding_y * 2)

            paste_x = padding_x  # Padding amount becomes the paste position
            paste_y = padding_y

        # --- Fit to resolution (works with padding) ---
        # Initialize variables to avoid warnings
        needs_final_centering = False
        temp_canvas_width = 0
        temp_canvas_height = 0
        final_paste_x = 0
        final_paste_y = 0

        if fit_to_target_resolution:
            # Calculate the total size needed including padding
            total_width_needed = canvas_width
            total_height_needed = canvas_height

            # If scale_down is enabled and the padded image exceeds target, scale down
            if scale_down and (total_width_needed > target_width or total_height_needed > target_height):
                scale_factor = min(target_width / total_width_needed, target_height / total_height_needed)

                # Scale down the image
                new_img_size = (int(img_pil.width * scale_factor), int(img_pil.height * scale_factor))
                img_pil = img_pil.resize(new_img_size, Image.Resampling.LANCZOS)

                # Scale down the padding as well
                paste_x = int(paste_x * scale_factor)
                paste_y = int(paste_y * scale_factor)

                # Recalculate canvas to fit within target
                canvas_width = int(canvas_width * scale_factor)
                canvas_height = int(canvas_height * scale_factor)

            # Now center the entire padded image within the target resolution
            # The canvas becomes the target size
            final_paste_x = (target_width - canvas_width) // 2
            final_paste_y = (target_height - canvas_height) // 2

            # We need to create the target-sized canvas first, then paste the padded image
            # So we'll handle this after creating the initial canvas
            # Store these values for later use
            needs_final_centering = True
            temp_canvas_width = canvas_width
            temp_canvas_height = canvas_height
            canvas_width = target_width
            canvas_height = target_height

        # --- Create canvas ---
        canvas_mode = "RGBA" if transparent_padding else "RGB"
        if transparent_padding:
            canvas_color = (0, 0, 0, 0)
        else:
            try:
                canvas_color = Image.new("RGB", (1, 1), pad_color).getpixel((0, 0))
            except:
                canvas_color = (0, 0, 0)

        # Store the actual image dimensions before any canvas operations
        actual_img_width = img_pil.width
        actual_img_height = img_pil.height

        if needs_final_centering:
            # Create intermediate canvas with padding
            temp_canvas = Image.new(canvas_mode, (temp_canvas_width, temp_canvas_height), canvas_color)
            mask = img_pil.getchannel('A') if canvas_mode == "RGBA" else None
            temp_canvas.paste(img_pil, (paste_x, paste_y), mask)

            # Create final target-sized canvas
            canvas = Image.new(canvas_mode, (canvas_width, canvas_height), canvas_color)
            # Paste the padded image centered on the target canvas
            canvas.paste(temp_canvas, (final_paste_x, final_paste_y))

            # Update paste positions for mask calculation
            paste_x = final_paste_x + paste_x
            paste_y = final_paste_y + paste_y
        else:
            canvas = Image.new(canvas_mode, (canvas_width, canvas_height), canvas_color)
            # --- Paste image onto canvas ---
            mask = img_pil.getchannel('A') if canvas_mode == "RGBA" else None
            canvas.paste(img_pil, (paste_x, paste_y), mask)

        # --- Create mask for padded/feathered areas ---
        mask_img = np.ones((canvas_height, canvas_width), dtype=np.float32) * 255
        x0, y0 = paste_x, paste_y
        x1, y1 = paste_x + actual_img_width, paste_y + actual_img_height
        yy, xx = np.meshgrid(np.arange(canvas_height), np.arange(canvas_width), indexing='ij')

        inside_img = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)

        if feather_padding:
            # Apply feathering gradient
            dist_left = xx - x0
            dist_right = x1 - xx - 1
            dist_top = yy - y0
            dist_bottom = y1 - yy - 1

            if feather_distance_type == "manhattan":
                dist_to_edge = np.minimum.reduce([dist_left, dist_right, dist_top, dist_bottom])
            else:  # smooth - smooth rounded corners via Gaussian filtering
                # Use Manhattan distance as base, then apply Gaussian smoothing
                # This creates smooth rounded corners without distance field discontinuities
                # Kursat 29.Oct.2025
                from scipy.ndimage import gaussian_filter

                # Start with Manhattan distance
                dist_to_edge = np.minimum.reduce([dist_left, dist_right, dist_top, dist_bottom])

                # Apply Gaussian smoothing to create rounded corners
                # Sigma is proportional to feather radius for smooth effect
                sigma = feather_radius * 0.15  # 15% of feather radius
                dist_to_edge = gaussian_filter(dist_to_edge.astype(np.float64), sigma=sigma, mode='nearest')

            feather_zone = (dist_to_edge >= 0) & (dist_to_edge < feather_radius)
            mask_img[dist_to_edge >= feather_radius] = 0
            mask_img[feather_zone] = 255 * (1 - dist_to_edge[feather_zone] / feather_radius)
        else:
            # No feathering - binary mask (sharp edges)
            # Inside image = 0 (no mask), outside = 255 (full mask)
            mask_img[inside_img] = 0

        # --- Blend image with padding using mask ---
        # Only perform blending if we did a direct paste (not using temp_canvas)
        if not needs_final_centering:
            canvas_np = np.array(canvas).astype(np.float32) / 255.0
            img_np = np.array(img_pil).astype(np.float32) / 255.0
            for c in range(canvas_np.shape[2]):
                region = canvas_np[y0:y1, x0:x1, c]
                img_region = img_np[:, :, c]
                mask_region = mask_img[y0:y1, x0:x1] / 255.0
                region = img_region * (1 - mask_region) + region * mask_region
                canvas_np[y0:y1, x0:x1, c] = region
        else:
            # Canvas already has the image pasted via temp_canvas
            canvas_np = np.array(canvas).astype(np.float32) / 255.0
        canvas_tensor = torch.from_numpy(canvas_np).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_img / 255.0).unsqueeze(0)
        # --- Third output: blended image using mask ---
        if transparent_padding:
            pad_arr = np.array([0, 0, 0, 0], dtype=np.float32) / 255.0
        else:
            pad_arr = np.array(canvas_color, dtype=np.float32) / 255.0
            if pad_arr.shape[0] != canvas_np.shape[2]:
                pad_arr = np.pad(pad_arr, (0, canvas_np.shape[2] - pad_arr.shape[0]), mode='constant')
        pad_img = np.ones_like(canvas_np) * pad_arr
        mask_norm = (mask_img / 255.0)[..., None]
        blended_np = canvas_np * (1 - mask_norm) + pad_img * mask_norm
        blended_tensor = torch.from_numpy(blended_np).unsqueeze(0)
        return (canvas_tensor, mask_tensor, blended_tensor)

# Register
NODE_CLASS_MAPPINGS = {
    "CropPadAdvancedKrs": CropPadAdvancedKrs,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CropPadAdvancedKrs": "🧩 Crop Pad Advanced Krs",
}
