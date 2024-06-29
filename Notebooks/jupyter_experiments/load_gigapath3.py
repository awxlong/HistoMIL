import timm
import torch
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=f'/home/xuelonan/secrier_lab/persistence/HistoMIL/API.env')
hf_api_key = os.getenv('HF_READ_KEY')
login(token=hf_api_key)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with torch.cuda.stream(torch.cuda.Stream()):
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).to(device)

print('model loaded succesfully with cuda stream')