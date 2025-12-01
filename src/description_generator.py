# src/description_generator.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
from difflib import SequenceMatcher

class DescriptionGenerator:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DescriptionGenerator, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Load model and tokenizer only on first initialization
        if hasattr(self, 'pipe'):
            return

        print("  - Initializing Natural Language Description Generator (loading Phi-3 model for GPU)...")
        print("    (This may take a few minutes and requires an internet connection on first run)")
        
        if not torch.cuda.is_available():
            print("  - [WARNING] CUDA not available. Description generation will be very slow on CPU.")
            self.pipe = None
            return

        torch.random.manual_seed(0)
        model_id = "microsoft/Phi-3-mini-4k-instruct"
        
        # Try loading with quantization first, fall back to float16 if it fails
        model = self._load_model_with_fallback(model_id)
        
        if model is None:
            self.pipe = None
            return
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
            print("  - ✅ Description Generator initialized successfully on GPU.")
        except Exception as e:
            print(f"  - [ERROR] Failed to create pipeline: {e}")
            print("  - Natural language descriptions will be disabled.")
            self.pipe = None

    def _load_model_with_fallback(self, model_id: str):
        """
        Load model with quantization fallback.
        
        Tries 4-bit quantization first (saves VRAM), falls back to float16 if 
        bitsandbytes/triton compatibility issues occur.
        """
        # Attempt 1: Try 4-bit quantization (requires compatible triton + bitsandbytes)
        try:
            from transformers import BitsAndBytesConfig
            
            print("  - Attempting 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
                quantization_config=quantization_config
            )
            print("  - ✅ Loaded with 4-bit quantization (memory efficient)")
            return model
            
        except Exception as e:
            print(f"  - ⚠️ 4-bit quantization failed: {e}")
            print("  - Falling back to float16 (uses more VRAM but compatible)...")
        
        # Attempt 2: Fall back to float16 (no quantization)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="cuda",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            print("  - ✅ Loaded with float16 (no quantization)")
            return model
            
        except Exception as e:
            print(f"  - [ERROR] float16 loading also failed: {e}")
        
        # Attempt 3: Try bfloat16 as last resort
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="cuda",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            print("  - ✅ Loaded with bfloat16")
            return model
            
        except Exception as e:
            print(f"  - [ERROR] All loading methods failed: {e}")
            print("  - Natural language descriptions will be disabled.")
            return None

    def _generate_text(self, prompt, max_length=100):
        if not self.pipe:
            return ""

        generation_args = {
            "max_new_tokens": max_length,
            "return_full_text": False,
            "temperature": 0.1,
            "do_sample": False,
        }
        
        messages = [{"role": "user", "content": prompt}]
        try:
            output = self.pipe(messages, **generation_args)
            if output and output[0]['generated_text']:
                return output[0]['generated_text'].strip()
        except Exception as e:
            print(f"Generation error: {e}")
            
        return ""

    def correct_ocr_mistakes(self, noisy_text):
        """
        SAFE CORRECTION: Uses LLM to fix obvious OCR typos but prevents hallucinations.
        Strictly checks similarity to ensure the model doesn't invent content.
        """
        if not self.pipe or not noisy_text or len(noisy_text) < 4:
            return noisy_text

        # 1. Strict Prompting
        prompt = f"""You are a text correction tool. Your ONLY job is to fix broken English words in the text below caused by OCR errors.
        
        Rules:
        1. Do NOT add new information.
        2. Do NOT explain anything.
        3. Output ONLY the corrected text.
        4. If the text is mostly numbers or codes, return it exactly as is.
        
        Broken Text: "{noisy_text}"
        Corrected Text:"""

        correction = self._generate_text(prompt, max_length=len(noisy_text) + 20)
        
        # Cleanup: remove quotes if model added them
        correction = correction.replace('"', '').strip()
        
        # 2. HALLUCINATION GUARDRAILS
        
        # A. Length Check: Correction shouldn't be drastically different in length
        len_ratio = len(correction) / (len(noisy_text) + 1)
        if len_ratio < 0.5 or len_ratio > 1.5:
            return noisy_text

        # B. Similarity Check: Uses SequenceMatcher to ensure structural similarity
        # We look for a similarity > 0.4 (allows for fixing typos but bans total rewrites)
        similarity = SequenceMatcher(None, noisy_text.lower(), correction.lower()).ratio()
        
        if similarity < 0.4:
            return noisy_text

        # If it passes checks, return the improved text
        return correction

    def generate_for_table(self, table_df: pd.DataFrame) -> str:
        if table_df.empty:
            return "The table is empty."
        
        num_rows, num_cols = table_df.shape
        column_names = table_df.columns.tolist()
        
        head_sample = table_df.head(2).to_string(index=False, max_cols=10)
        tail_sample = table_df.tail(2).to_string(index=False, max_cols=10) if num_rows > 4 else ""
        
        prompt = f"""
You are a data analysis assistant. Based on the following table information, write a single, concise, descriptive sentence summarizing what the table is about. Do not list the data values, just describe the table's purpose and content type.

Table Shape: {num_rows} rows × {num_cols} columns
Column Headers: {column_names}

Sample Data (First 2 rows):
---
{head_sample}
---
"""
        if tail_sample:
            prompt += f"""
Sample Data (Last 2 rows):
---
{tail_sample}
---
"""
        prompt += "\nOne-sentence summary:"
        return self._generate_text(prompt, max_length=60)

    def generate_for_figure(self, kind: str, caption: str) -> str:
        if not caption and not kind:
            return "No information available to describe the figure."

        prompt = f"""
You are a document analysis assistant. Based on the following information, write a one-sentence natural language description for the figure.

- Figure Type: '{kind}'
- Provided Caption: '{caption if caption else 'None'}'

One-sentence description:
"""
        return self._generate_text(prompt, max_length=50)
    
    def _verify_value_in_text(self, value, text_source, threshold=0.85):
        value = str(value).lower().strip()
        text_source = text_source.lower()
        if len(value) < 4:
            return value in text_source.split()
        if value in text_source:
            return True
        v_words = value.split()
        if not v_words: return False
        try:
            start_index = text_source.find(v_words[0])
            if start_index == -1: return False
            end_index = text_source.find(v_words[-1], start_index)
            if end_index == -1: return False
            if (end_index - start_index) > (len(value) + 20):
                return False
            return True
        except:
            return False

    def extract_fields_from_text(self, text_blocks_summary: str, target_fields: list = None) -> dict:
        if not self.pipe:
            return {}
        if target_fields is None:
            target_fields = ['date', 'to', 'from', 'subject', 'fax', 'phone', 'pages']

        fields_str = ', '.join(target_fields)
        prompt = f"""You are a strict data extraction assistant. Your job is to find specific information in the provided OCR text.

OCR Text:
---
{text_blocks_summary[:3500]}
---

Task: Extract values for these fields: {fields_str}

RULES:
1. EXTRACT ONLY what is explicitly written in the text.
2. IF A FIELD IS NOT FOUND, DO NOT INVENT DATA.
3. If not found, output: field_name: null
4. Do not convert dates.
5. Output format must be exactly: key: value

Extracted Data:"""

        try:
            result_text = self._generate_text(prompt, max_length=150)
            extracted = {}
            for line in result_text.strip().split('\n'):
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip().lower()
                        value = parts[1].strip()
                        if value.lower() in ['null', 'not found', 'n/a', 'none', 'unknown']: continue
                        if key not in [f.lower() for f in target_fields]: continue
                        if self._verify_value_in_text(value, text_blocks_summary):
                            extracted[key] = value
            return extracted
        except Exception as e:
            print(f"    - Error during LLM field extraction: {e}")
            return {}

    def extract_fields_structured(self, text_blocks: list, target_fields: list = None) -> dict:
        if not self.pipe:
            return {}
        if target_fields is None:
            target_fields = ['date', 'to', 'from', 'subject', 'fax', 'phone', 'pages']
        block_summaries = []
        for i, block in enumerate(text_blocks[:50]):
            content = block.get('content', '').strip()
            if content:
                block_summaries.append(f"[Block {i+1}] {content}")
        text_summary = '\n'.join(block_summaries)
        return self.extract_fields_from_text(text_summary, target_fields)
