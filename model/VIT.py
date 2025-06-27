import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers import ViTConfig, ViTModel


#model = timm.create_model('vit_base_patch16_224', pretrained=True)
#model.eval()



# Initializing a ViT vit-base-patch16-224 style configuration
configuration = ViTConfig()

# Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
model = ViTModel(configuration)

# Accessing the model configuration
configuration = model.config

x = torch.randn((1,1,224,224))
y = model(x)
print(y)