

import torch
#PyTorch, the deep learning library everything is built on top of, is imported here to allow for device management (CPU vs GPU) and tensor operations.
from transformers import AutoModelForCausalLM, AutoTokenizer
#from Hugging Face's transformers library. "Auto" means these classes automatically select the appropriate model/tokenizer architecture based on the model name provided, so the same code works for any Hugging Face model without us writing model-specific loading logic. 

import config


class LegalPromptModel: #this defines a class; a reusable blueprint for a "loaded language model that can generate text". __init__ runs when we create one.  
    def __init__(self, model_name: str):
        # model_name is now REQUIRED, not optional-with-a-default —
        # forces every caller to be explicit about which model in
        # the progression it's running, rather than silently falling
        # back to a single hardcoded default as Phase 0 did.
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._loaded = False

    def load(self, device_map: str = "auto", dtype: torch.dtype = torch.bfloat16): #loading the model
        print(f"Loading {self.model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )
        self._loaded = True

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            print(f"Loaded {self.model_name} — GPU memory allocated: {allocated:.2f} GB")
        else:
            print(f"WARNING: Loaded {self.model_name} but no CUDA device detected.")

    def generate(self, prompt_text: str, max_new_tokens: int = None) -> str: #generating text
        if not self._loaded:
            raise RuntimeError("Call .load() before .generate().")

        messages = [{"role": "user", "content": prompt_text}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        input_length = inputs["input_ids"].shape[-1]

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or config.MAX_NEW_TOKENS,
            do_sample=True,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        decoded = self.tokenizer.decode(
            output[0][input_length:], skip_special_tokens=True
        )
        return decoded

    def generate_and_parse(self, prompt_text: str, max_new_tokens: int = None): #for generating and parsing
        raw_output = self.generate(prompt_text, max_new_tokens=max_new_tokens)
        parsed = config.extract_final_answer(raw_output)
        return raw_output, parsed

    def unload(self): #for freeing memory 
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False
        print(f"Unloaded {self.model_name}, GPU memory freed.")