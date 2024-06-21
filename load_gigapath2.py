
import timm
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=f'/home/xuelonan/secrier_lab/persistence/HistoMIL/API.env')
hf_api_key = os.getenv('HF_READ_KEY')
login(token=hf_api_key)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download the model file
model_file = hf_hub_download("prov-gigapath/prov-gigapath", "pytorch_model.bin")

# Load the model architecture without weights
model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False).to(device)

# Load the state dict
state_dict = torch.load(model_file, map_location=device)

# Load weights in chunks
for key, value in state_dict.items():
    if key in model.state_dict():
        model.state_dict()[key].copy_(value.to(device))
    elif key.startswith('model.'):
        stripped_key = key[6:]  # remove 'model.' prefix
        if stripped_key in model.state_dict():
            model.state_dict()[stripped_key].copy_(value.to(device))
    del value  # Free up memory

model.eval()

print('model loaded succesfully in parts')
