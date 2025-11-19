import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
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
            return ""

        generation_args = {
            "max_new_tokens": max_length,
            "return_full_text": False,
            "temperature": 0.1,
            "do_sample": False,
        }
        
        messages = [{"role": "user", "content": prompt}]
        output = self.pipe(messages, **generation_args)
        
        if output and output[0]['generated_text']:
            return output[0]['generated_text'].strip()
        return ""

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
    
    def _verify_value_in_text(self, value, text_source, threshold=0.85):
        """
        MORE ROBUST VERIFICATION:
        The value found by the LLM must appear as a 'whole' within the text block.
        We are searching for phrases/segments, not just individual words.
        """
        value = str(value).lower().strip()
        text_source = text_source.lower()
        
        # 1. If the value is very short (e.g., "10", "ok"), strictly look for an exact match
        if len(value) < 4:
            return value in text_source.split() # Accept only if present as a whole word

        # 2. Quick check: Does it appear directly?
        if value in text_source:
            return True
            
        # 3. Fuzzy Search - The Most Robust Part
        # Checks the best match ratio of the value within the text.
        # If similarity is less than 85%, it is considered a hallucination.
        
        # SequenceMatcher can be slow on large texts, so instead of sliding windows,
        # we use simple 'token overlap' + 'sequence preservation'.
        
        # Simple but effective method: Are the words SIDE BY SIDE or CLOSE in the text?
        v_words = value.split()
        if not v_words: return False
        
        try:
            # Find the first word
            start_index = text_source.find(v_words[0])
            if start_index == -1: return False # Reject if even the first word is missing
            
            # Search for the last word after the first word
            end_index = text_source.find(v_words[-1], start_index)
            if end_index == -1: return False # Reject if the last word is missing
            
            # If the distance is too large (e.g., "John" at top of page, "Smith" at bottom), reject.
            # Allow a tolerance of word count * 10 characters.
            if (end_index - start_index) > (len(value) + 20):
                return False
                
            return True
        except:
            return False

    def extract_fields_from_text(self, text_blocks_summary: str, target_fields: list = None) -> dict:
        """
        Uses LLM to extract fields but verifies them against the source text.
        """
        if not self.pipe:
            return {}

        if target_fields is None:
            target_fields = ['date', 'to', 'from', 'subject', 'fax', 'phone', 'pages']

        fields_str = ', '.join(target_fields)

        # NEW PROMPT: Much stricter rules
        prompt = f"""You are a strict data extraction assistant. Your job is to find specific information in the provided OCR text.

OCR Text:
---
{text_blocks_summary[:3500]}  # Truncate if text is too long
---

Task: Extract values for these fields: {fields_str}

RULES:
1. EXTRACT ONLY what is explicitly written in the text.
2. IF A FIELD IS NOT FOUND, DO NOT INVENT DATA. DO NOT WRITE "John Smith" or "ACME".
3. If not found, output: field_name: null
4. Do not convert dates to "today". Use the date found in text.
5. Output format must be exactly: key: value

Extracted Data:"""

        try:
            result_text = self._generate_text(prompt, max_length=150)
            print(f"    - LLM Raw Output: {result_text.replace(chr(10), ' | ')}") # For debugging

            extracted = {}
            for line in result_text.strip().split('\n'):
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip().lower()
                        value = parts[1].strip()
                        
                        # Cleanup
                        if value.lower() in ['null', 'not found', 'n/a', 'none', 'unknown']:
                            continue
                            
                        # FILTER: Is it one of the requested fields?
                        if key not in [f.lower() for f in target_fields]:
                            continue

                        # CRITICAL STEP: Verification
                        # Does the value found by the LLM exist in the original text?
                        if self._verify_value_in_text(value, text_blocks_summary):
                            extracted[key] = value
                        else:
                            print(f"    - [HALLUCINATION BLOCKED] LLM suggested '{value}' for '{key}', but it was not in text.")

            return extracted
        except Exception as e:
            print(f"    - Error during LLM field extraction: {e}")
            return {}

    def extract_fields_structured(self, text_blocks: list, target_fields: list = None) -> dict:
        """
        Enhanced field extraction using structured text block information.

        Args:
            text_blocks: List of text block dictionaries with 'content' and 'bounding_box'
            target_fields: List of field names to extract

        Returns:
            Dictionary of extracted fields.
        """
        if not self.pipe:
            return {}

        if target_fields is None:
            target_fields = ['date', 'to', 'from', 'subject', 'fax', 'phone', 'pages']

        # Create a structured summary of text blocks
        block_summaries = []
        for i, block in enumerate(text_blocks[:50]):  # Limit to first 50 blocks
            content = block.get('content', '').strip()
            if content:
                block_summaries.append(f"[Block {i+1}] {content}")

        text_summary = '\n'.join(block_summaries)

        return self.extract_fields_from_text(text_summary, target_fields)