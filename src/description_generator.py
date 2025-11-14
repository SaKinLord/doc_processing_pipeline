# src/description_generator.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import pandas as pd

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

        try:
            torch.random.manual_seed(0)
            model_id = "microsoft/Phi-3-mini-4k-instruct"

            # 4-bit Quantization Settings
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            # Load model to GPU in 4-bit
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
                quantization_config=quantization_config
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            self.pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
            print("  - ✅ Description Generator initialized successfully on GPU.")
        except Exception as e:
            print(f"  - [ERROR] Failed to initialize Description Generator: {e}")
            print("  - Natural language descriptions will be disabled.")
            self.pipe = None

    def _generate_text(self, prompt, max_length=100):
        if not self.pipe:
            return "Description generation is disabled."

        generation_args = {
            "max_new_tokens": max_length,
            "return_full_text": False,
            "temperature": 0.3,
            "do_sample": True,
        }
        
        # Create a message list in the format expected by Phi-3
        messages = [
            {"role": "user", "content": prompt},
        ]

        # Run the model
        output = self.pipe(messages, **generation_args)
        
        # Clean the output
        if output and output[0]['generated_text']:
            return output[0]['generated_text'].strip()
        return "Could not generate a description."

    def generate_for_table(self, table_df: pd.DataFrame) -> str:
        """
        ENHANCED: Generates a one-sentence summary from a pandas DataFrame.
        Now provides much more context: shape, column headers, and head/tail samples.
        """
        if table_df.empty:
            return "The table is empty."
        
        num_rows, num_cols = table_df.shape
        column_names = table_df.columns.tolist()
        
        # Get sample data from head and tail
        head_sample = table_df.head(2).to_string(index=False, max_cols=10)
        tail_sample = table_df.tail(2).to_string(index=False, max_cols=10) if num_rows > 4 else ""
        
        # Build comprehensive context
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
        """Generates a one-sentence description based on a figure's type and caption."""
        if not caption and not kind:
            return "No information available to describe the figure."
            
        prompt = f"""
You are a document analysis assistant. Based on the following information, write a one-sentence natural language description for the figure.

- Figure Type: '{kind}'
- Provided Caption: '{caption if caption else 'None'}'

One-sentence description:
"""
        return self._generate_text(prompt, max_length=50)