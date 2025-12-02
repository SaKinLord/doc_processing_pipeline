# Patch for src/vlm_manager.py
# Apply these changes to fix T4 GPU compatibility

"""
INSTRUCTIONS:
Find the _load_qwen_vl method in vlm_manager.py and replace it with the version below.
The key changes are:
1. Use torch.float16 instead of torch.bfloat16 (T4 doesn't support bfloat16 well)
2. Better error handling for quantization failures
3. Detect T4 GPU and skip bitsandbytes quantization (which has triton issues)
"""

# Replace the existing _load_qwen_vl method with this:

def _load_qwen_vl(self):
    """Load Qwen-VL-Chat model with T4-compatible fallbacks."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    print("    - Initializing Qwen-VL-Chat model...")
    
    model_id = "Qwen/Qwen-VL-Chat"
    
    # 🆕 BUGFIX: Detect T4 GPU - doesn't support BFloat16 well
    device_name = torch.cuda.get_device_name(0).lower()
    is_older_gpu = any(gpu in device_name for gpu in ['t4', 'p100', 'k80', 'p4', 'v100'])
    
    if is_older_gpu:
        print(f"    - Detected older GPU ({device_name}), using float16")
    
    # Try quantization first (if not on problematic GPU)
    if self.use_4bit and not is_older_gpu:
        try:
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,  # 🆕 Use float16, not bfloat16
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=quantization_config
            )
            print("    - ✅ Qwen-VL-Chat loaded with 4-bit quantization.")
            
        except Exception as e:
            print(f"    - ⚠️ 4-bit quantization failed: {e}")
            print("    - Falling back to float16...")
            self._load_qwen_float16(model_id)
    else:
        # Directly use float16 for T4 and older GPUs
        self._load_qwen_float16(model_id)
    
    try:
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"    - ⚠️ Tokenizer loading failed: {e}")
        self.model = None
        return


def _load_qwen_float16(self, model_id):
    """
    Load Qwen with float16 - T4 compatible fallback.
    
    🆕 NEW METHOD: Added to handle T4 GPU compatibility.
    """
    import torch
    from transformers import AutoModelForCausalLM
    
    print("    - Loading with float16 (T4-compatible)...")
    try:
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,  # 🆕 Use float16 instead of bfloat16
            low_cpu_mem_usage=True  # 🆕 Reduce memory usage during loading
        )
        print("    - ✅ Qwen-VL-Chat loaded with float16.")
    except Exception as e:
        print(f"    - ⚠️ Float16 loading failed: {e}")
        print("    - VLM will be disabled.")
        self.model = None


# Also update _load_llava similarly:

def _load_llava(self):
    """Load LLaVA-1.5 model with T4-compatible fallbacks."""
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    import torch
    
    print("    - Initializing LLaVA-1.5 model...")
    
    model_id = "llava-hf/llava-1.5-7b-hf"
    
    # 🆕 BUGFIX: Detect older GPU
    device_name = torch.cuda.get_device_name(0).lower()
    is_older_gpu = any(gpu in device_name for gpu in ['t4', 'p100', 'k80', 'p4', 'v100'])
    
    # Try quantization first (if not on problematic GPU)
    if self.use_4bit and not is_older_gpu:
        try:
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,  # 🆕 Use float16
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=quantization_config
            )
            print("    - ✅ LLaVA-1.5 loaded with 4-bit quantization.")
            
        except Exception as e:
            print(f"    - ⚠️ 4-bit quantization failed: {e}")
            print("    - Falling back to float16...")
            
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16  # 🆕 Use float16
            )
            print("    - ✅ LLaVA-1.5 loaded with float16.")
    else:
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16  # 🆕 Use float16 for T4
        )
        print("    - ✅ LLaVA-1.5 loaded with float16.")
    
    self.processor = AutoProcessor.from_pretrained(model_id)
