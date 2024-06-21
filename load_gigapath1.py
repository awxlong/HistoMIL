
import timm
import torch
from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv(dotenv_path=f'{args.api_dir}API.env')
hf_api_key = os.getenv('HF_READ_KEY')
login(token=hf_api_key)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model in half-precision
with torch.cuda.amp.autocast():
    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).to(device).half()
model.eval()

print('model loaded successfully for autocast and eval')

