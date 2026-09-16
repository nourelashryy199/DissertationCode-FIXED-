import torch
#PyTorch, the deep learning library used to load and run the model. It is also used here to check whether a GPU is available, keep track of GPU memory, and clear GPU memory when the model is unloaded.

from transformers import AutoModelForCausalLM, AutoTokenizer
#from Hugging Face's transformers library. AutoTokenizer loads the tokenizer associated with whichever model name is provided, while AutoModelForCausalLM loads the corresponding causal language model. Using the Auto classes means the same code can be used for the different Qwen model sizes without writing model-specific loading code.

import Phase01HPC.ThesisWork.config as config
#imports the shared experiment settings from config.py, including the token limit, temperature, top_p, and the function used to extract the final answer from the model output.


class LegalPromptModel: #this defines a reusable class for loading one of the language models, generating responses from prompts, parsing its final answers, and unloading it when it is no longer needed.
    def __init__(self, model_name: str):
        #model_name is required here because the 7B, 14B and 32B models are run separately.
        #this makes the caller explicitly specify which model is being used instead of silently falling back to one default model.
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._loaded = False

    def load(self, device_map: str = "auto", dtype: torch.dtype = torch.bfloat16): #loading the model and its tokenizer
        print(f"Loading {self.model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            #some causal language model tokenizers do not have a separate padding token, so the end-of-sequence token is used as the padding token if needed.

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )
        #loads the model in bfloat16. device_map="auto" lets Hugging Face decide how to place the model on the available device(s), which is especially useful for the 32B model that uses two GPUs.
        self._loaded = True

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            print(f"Loaded {self.model_name} — GPU memory allocated: {allocated:.2f} GB")
            #prints the amount of GPU memory allocated after loading, mainly as a quick check that the model has loaded onto CUDA successfully.
        else:
            print(f"WARNING: Loaded {self.model_name} but no CUDA device detected.")

    def generate(self, prompt_text: str, max_new_tokens: int = None) -> str: #generating a response from one completed prompt
        if not self._loaded:
            raise RuntimeError("Call .load() before .generate().")
            #prevents generation from being attempted before the model and tokenizer have actually been loaded.

        messages = [{"role": "user", "content": prompt_text}]
        #the completed experimental prompt is passed to the model as a single user message.

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)
        #applies the chat template expected by the instruct model, converts the result to tensors, and moves the inputs onto the same device as the model.

        input_length = inputs["input_ids"].shape[-1]
        #stores the number of input tokens so that the original prompt can be removed when the generated sequence is decoded.

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or config.MAX_NEW_TOKENS,
            do_sample=True,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        #generates the model response using stochastic sampling.
        #temperature and top_p come from config.py and are shared across the experiment, while max_new_tokens defaults to the experiment-wide token budget unless another value is explicitly passed to this function.

        decoded = self.tokenizer.decode(
            output[0][input_length:], skip_special_tokens=True
        )
        #only the newly generated tokens are decoded, rather than decoding the original prompt together with the response.
        return decoded

    def generate_and_parse(self, prompt_text: str, max_new_tokens: int = None): #generating the raw response and then extracting its final answer
        raw_output = self.generate(prompt_text, max_new_tokens=max_new_tokens)
        parsed = config.extract_final_answer(raw_output)
        #extract_final_answer() looks for the required "Final Answer:" line in the generated response. If it cannot find one, parsed will be None and this is treated as a parsing failure later.
        return raw_output, parsed

    def unload(self): #freeing the model and GPU memory after it is no longer needed
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            #releases unused cached GPU memory after deleting the model.
        self._loaded = False
        print(f"Unloaded {self.model_name}, GPU memory freed.")