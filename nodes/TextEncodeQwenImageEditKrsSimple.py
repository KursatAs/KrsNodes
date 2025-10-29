import node_helpers
import math
from comfy import utils


class TextEncodeQwenImageEditKrsSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "alignment": ([8, 16, 24, 32, 56, 64], {"default": 32}),
                "vl_resolution": (
                    "INT",
                    {"default": 384, "min": 256, "max": 2048, "step": 64},
                ),
                "system_prompt": (
                    ["default", "minimal", "custom"],
                    {"default": "default"}
                ),
            },
            "optional": {
                "vae": ("VAE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "custom_system_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = (
        "CONDITIONING",
        "IMAGE",
    )
    RETURN_NAMES = (
        "CONDITIONING",
        "output_image",
    )
    FUNCTION = "encode"

    CATEGORY = "utils/conditioning"

    def encode(self, clip, prompt, alignment=32, vl_resolution=384, system_prompt="default", vae=None, image1=None, image2=None, image3=None, custom_system_prompt=""):
        ref_latents = []
        images = [image1, image2, image3]
        images_vl = []
        output_image = None

        # Select system prompt based on user choice
        if system_prompt == "minimal":
            llama_template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        elif system_prompt == "custom" and custom_system_prompt:
            llama_template = f"<|im_start|>system\n{custom_system_prompt}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n"
        else:  # default
            llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                cur_height, cur_width = image.shape[1:3]
                samples = image.movedim(-1, 1)

                # For vision input (configurable resolution)
                res = vl_resolution * vl_resolution
                input_size = cur_height * cur_width
                scale_factor = math.sqrt(res / input_size)

                new_height = cur_height * scale_factor
                new_width = cur_width * scale_factor
                height = round(new_height)
                width = round(new_width)

                s = utils.common_upscale(samples, width, height, "lanczos", "center")
                # Adjust image tensor slicing to retain RGB channels
                images_vl.append(s.movedim(1, -1)[:, :, :, :3])

                # For VAE encoding (1024x1024 target resolution with alignment)
                if vae is not None:
                    res = 1024 * 1024
                    scale_factor = math.sqrt(res / input_size)

                    new_height = cur_height * scale_factor
                    new_width = cur_width * scale_factor
                    height = round(new_height / alignment) * alignment
                    width = round(new_width / alignment) * alignment

                    s = utils.common_upscale(samples, width, height, "lanczos", "center")
                    resized_vae_image = s.movedim(1, -1)[:, :, :, :3]
                    ref_latents.append(vae.encode(resized_vae_image))

                    # Store the VAE-resized image only for image1
                    if i == 0 and image1 is not None:
                        output_image = resized_vae_image

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        return (conditioning, output_image)


# Register the node
NODE_CLASS_MAPPINGS = {
    "TextEncodeQwenImageEditKrsSimple": TextEncodeQwenImageEditKrsSimple
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeQwenImageEditKrsSimple": "Text Encode Qwen Image Edit Simple Krs"
}
