"""
config local machine and user with all files of a cohort
"""
# changing to HistoMIL's directory
import os
import sys
sys.path.append(os.getcwd())
# HistoMIL relevant packages
from HistoMIL.DATA.Cohort.location import Locations
from HistoMIL.EXP.workspace.env import Machine
from HistoMIL.EXP.workspace.env import Person
import pickle
import pdb
from args import get_args_machine_config

# Login and starting experiment-related API calls, e.g., HugginFace and WandB
from dotenv import load_dotenv
# from huggingface_hub import login

def setup_config(args):
        localcohort_name = args.cohort_name # e.g. "BRCA" for breast cancer
        data_dir = f"{args.data_dir}{localcohort_name}/" # e.g. /SAN/ugi/WSI_Trans/DATA/BRCA/

        
        #--------------------------> init machine and person
        data_locs = Locations(root = data_dir,
                              sub_dirs = {
                                        "slide":f"TCGA-{localcohort_name}/",
                                        "tissue":"Tissue/",
                                        "patch":"Patch/",
                                        "patch_img":"Patch_Image/",
                                        "feature":"Feature/",
                                        },
                              is_build = True)
        
        exp_dir = f"{args.exp_dir}"                    # e.g. /home/xuelonan/g0_arrest/
        exp_locs = Locations(root=exp_dir,
                             sub_dirs={
                                        "src":"HistoMIL/",
                                        "idx":"Data/",
                                        "saved_models":"SavedModels/",
                                        "out_files":"OutFiles/",
                                        "temp":"Temp/",
                                        "user":"User/",
                                        },
                             is_build = True) 


        # pdb.set_trace()
        load_dotenv(dotenv_path=f"{args.api_dir}API.env")
        
        machine = Machine(data_locs, exp_locs)
        user = Person(id=args.id)
        user.name = args.username
        user.wandb_api_key = os.getenv("WANDB_API_KEY")
        # hf_api_key = os.getenv("HF_READ_KEY")
        # login(token=hf_api_key) # no need to initalize this now 
        # pdb.set_trace()
        # save as pickle
        loc = exp_locs.abs_loc("user")
        with open(f"/{loc}/{localcohort_name}_machine_config.pkl", "wb") as f:
                pickle.dump([data_locs, exp_locs, machine, user], f)

        print(f"User configuration settings stored in {loc}{localcohort_name}_machine_config.pkl!")
def main():
        args = get_args_machine_config()
        setup_config(args)

if __name__ == "__main__":
    main()
    print("Data and experiment directories setup!")