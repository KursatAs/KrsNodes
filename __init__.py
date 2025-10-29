from .nodes.TextEncodeQwenImageEditKrsSimple import NODE_CLASS_MAPPINGS as simple_mappings, NODE_DISPLAY_NAME_MAPPINGS as simple_display
from .nodes.TextEncodeQwenImageEditKrsAdvanced import NODE_CLASS_MAPPINGS as advanced_mappings, NODE_DISPLAY_NAME_MAPPINGS as advanced_display
from .nodes.QwenImageEditLatentKrs import NODE_CLASS_MAPPINGS as latent_mappings, NODE_DISPLAY_NAME_MAPPINGS as latent_display
from .nodes.CropPadAdvancedKrs import NODE_CLASS_MAPPINGS as crop_mappings, NODE_DISPLAY_NAME_MAPPINGS as crop_display

NODE_CLASS_MAPPINGS = {**simple_mappings, **advanced_mappings, **latent_mappings, **crop_mappings}
NODE_DISPLAY_NAME_MAPPINGS = {**simple_display, **advanced_display, **latent_display, **crop_display}
