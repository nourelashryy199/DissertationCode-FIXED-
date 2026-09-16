#this script is used to download/cache the 14B Qwen model before running its generation job on Stanage.
#it should be run once somewhere with internet access so that the later GPU job can load the model from the existing Hugging Face cache instead of downloading it during generation.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_TO_DOWNLOAD = "Qwen/Qwen2.5-14B-Instruct"  #the 14B model that will be downloaded and stored in the Hugging Face cache


def predownload_model(model_name: str):
    print(f"Pre-downloading/caching model: {model_name} ...")
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForCausalLM.from_pretrained(model_name)
    print(f"  {model_name} cached successfully.")


if __name__ == "__main__":
    predownload_model(MODEL_TO_DOWNLOAD)