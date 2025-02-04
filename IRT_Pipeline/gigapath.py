
#%%
import subprocess
import sys
import importlib
import timm
import huggingface_hub
import os
print(f'timm version: {timm.__version__}, huggingface_hub version: {huggingface_hub.__version__}')
# Set Hugging Face token

from huggingface_hub import hf_hub_download

# Replace 'your-token' with your actual Hugging Face token
token = "your_token"

# model_path = hf_hub_download(repo_id="prov-gigapath/prov-gigapath", filename="model.safetensors", use_auth_token=token)

os.environ['HF_TOKEN'] = 'your_token'
#%%
# Load the model
def get_model():
    return timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)

# %%
try:
    tile_model = get_model()
    print("Model loaded successfully.")
except RuntimeError as e:
    print(f"RuntimeError: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
# %%


