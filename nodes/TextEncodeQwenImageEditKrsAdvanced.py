import node_helpers
import math
import torch
from comfy import utils


class TextEncodeQwenImageEditKrsAdvanced:
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
                "upscale_method": (["lanczos", "bicubic", "area", "nearest"], {"default": "bicubic"}),
            },
        }

    RETURN_TYPES = (
        "CONDITIONING",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "LATENT",
    )
    RETURN_NAMES = (
        "CONDITIONING",
        "output_image1",
        "output_image2",
        "output_image3",
        "latent",
    )
    FUNCTION = "encode"

    CATEGORY = "utils/conditioning"

    def encode(self, clip, prompt, alignment=32, vl_resolution=384, system_prompt="default", vae=None, image1=None, image2=None, image3=None, custom_system_prompt="", upscale_method="bicubic"):
        ref_latents = []
        images = [image1, image2, image3]
        images_vl = []
        resized_images = [None, None, None]

        template_prefix = "<|im_start|>system\n"
        template_suffix = "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        instruction_content = ""
        if system_prompt == "minimal":
            instruction_content = "You are a helpful assistant."
        elif system_prompt == "custom" and custom_system_prompt:
            # for handling mis use of instruction
            if template_prefix in custom_system_prompt:
                # remove prefix from instruction
                custom_system_prompt = custom_system_prompt.split(template_prefix)[1]
            if template_suffix in custom_system_prompt:
                # remove suffix from instruction
                custom_system_prompt = custom_system_prompt.split(template_suffix)[0]
            if "{}" in custom_system_prompt:
                # remove {} from instruction
                custom_system_prompt = custom_system_prompt.replace("{}", "")
            instruction_content = custom_system_prompt
        else:  # default
            instruction_content = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
        llama_template = template_prefix + instruction_content + template_suffix

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

                s = utils.common_upscale(samples, width, height, upscale_method, "center")
                images_vl.append(s.movedim(1, -1))

                # For VAE encoding (1024x1024 target resolution with alignment)
                if vae is not None:
                    res = 1024 * 1024
                    scale_factor = math.sqrt(res / input_size)

                    new_height = cur_height * scale_factor
                    new_width = cur_width * scale_factor
                    height = round(new_height / alignment) * alignment
                    width = round(new_width / alignment) * alignment

                    s = utils.common_upscale(samples, width, height, upscale_method, "center")
                    resized_vae_image = s.movedim(1, -1)[:, :, :, :3]
                    ref_latents.append(vae.encode(resized_vae_image))

                    resized_images[i] = resized_vae_image

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        # Return latent of first image if available, otherwise return empty latent
        if image1 is not None and len(ref_latents) > 0:
            samples = ref_latents[0]
        else:
            samples = torch.zeros(1, 4, 128, 128)
        latent_out = {"samples": samples}
        return (conditioning, resized_images[0], resized_images[1], resized_images[2], latent_out)


# Register the node
NODE_CLASS_MAPPINGS = {
    "TextEncodeQwenImageEditKrsAdvanced": TextEncodeQwenImageEditKrsAdvanced
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeQwenImageEditKrsAdvanced": "Text Encode Qwen Image Edit Advanced Krs"
}
