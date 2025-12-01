# src/vlm_manager.py
"""
🆕 Vision-Language Model (VLM) Manager (Feature 3)

This module provides VLM capabilities for complex form understanding and OCR confirmation.
Supports Qwen-VL-Chat and Llava-1.5 models with optional 4-bit quantization.

Usage:
    from src.vlm_manager import VLMManager
    
    vlm = VLMManager()
    result = vlm.query_image(image, "Extract the sender name and date in JSON format")
    
WARNING: This feature requires significant GPU memory (8GB+ VRAM).
Enable only when needed for complex form extraction.
"""

import json
import re
import cv2
import numpy as np
from typing import Optional, Dict, Any, List, Union
from PIL import Image

# Import config
try:
    from . import config
except ImportError:
    import config


class VLMManager:
    """
    Vision-Language Model Manager for document understanding.
    
    Provides image-based question answering capabilities for:
    - Complex form field extraction
    - OCR result confirmation/correction
    - Document classification
    - Handwriting interpretation
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to avoid loading model multiple times."""
        if not cls._instance:
            cls._instance = super(VLMManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = None, use_4bit: bool = None):
        """
        Initialize the VLM Manager.
        
        Args:
            model_name: 'qwen-vl-chat' or 'llava-1.5' (default from config)
            use_4bit: Enable 4-bit quantization for reduced memory (default from config)
        """
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.model_name = model_name or getattr(config, 'VLM_MODEL', 'qwen-vl-chat')
        self.use_4bit = use_4bit if use_4bit is not None else getattr(config, 'VLM_USE_4BIT', True)
        self.max_tokens = getattr(config, 'VLM_MAX_TOKENS', 256)
        self.temperature = getattr(config, 'VLM_TEMPERATURE', 0.1)
        
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._initialized = False
        self._available = False
        
        # Check if VLM is enabled
        if not getattr(config, 'USE_VLM', False):
            print("    - VLM is disabled in config. Enable USE_VLM to use this feature.")
            return
        
        self._load_model()
    
    def _load_model(self):
        """Load the VLM model based on configuration."""
        import torch
        
        if not torch.cuda.is_available():
            print("    - ⚠️ VLM requires CUDA. GPU not available.")
            return
        
        try:
            if 'qwen' in self.model_name.lower():
                self._load_qwen_vl()
            elif 'llava' in self.model_name.lower():
                self._load_llava()
            else:
                print(f"    - ⚠️ Unknown VLM model: {self.model_name}")
                return
            
            self._initialized = True
            self._available = True
            
        except Exception as e:
            print(f"    - ⚠️ Failed to initialize VLM: {e}")
            print("    - VLM features will be disabled.")
    
    def _load_qwen_vl(self):
        """Load Qwen-VL-Chat model with quantization fallback."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print("    - Initializing Qwen-VL-Chat model...")
        
        model_id = "Qwen/Qwen-VL-Chat"
        
        # Try quantization first, fall back to float16 if it fails
        if self.use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
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
                print("    - Falling back to bfloat16...")
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16
                )
                print("    - ✅ Qwen-VL-Chat loaded with bfloat16.")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            print("    - ✅ Qwen-VL-Chat loaded with bfloat16.")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    def _load_llava(self):
        """Load LLaVA-1.5 model with quantization fallback."""
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        import torch
        
        print("    - Initializing LLaVA-1.5 model...")
        
        model_id = "llava-hf/llava-1.5-7b-hf"
        
        # Try quantization first, fall back to float16 if it fails
        if self.use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
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
                print("    - Falling back to bfloat16...")
                
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_id,
                    device_map="auto",
                    torch_dtype=torch.bfloat16
                )
                print("    - ✅ LLaVA-1.5 loaded with bfloat16.")
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            print("    - ✅ LLaVA-1.5 loaded with bfloat16.")
        
        self.processor = AutoProcessor.from_pretrained(model_id)
    
    @property
    def is_available(self) -> bool:
        """Check if VLM is available for use."""
        return self._available and self.model is not None
    
    def query_image(
        self,
        image: Union[np.ndarray, Image.Image, str],
        prompt: str,
        max_tokens: int = None,
        temperature: float = None
    ) -> str:
        """
        Query the VLM with an image and a text prompt.
        
        Args:
            image: BGR numpy array, PIL Image, or path to image file
            prompt: Question or instruction about the image
            max_tokens: Maximum response tokens (default from config)
            temperature: Sampling temperature (default from config)
        
        Returns:
            str: Model's text response
        """
        if not self.is_available:
            print("    - VLM is not available. Returning empty response.")
            return ""
        
        # Convert image to PIL
        pil_image = self._to_pil_image(image)
        if pil_image is None:
            return ""
        
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        
        try:
            if 'qwen' in self.model_name.lower():
                return self._query_qwen(pil_image, prompt, max_tokens, temperature)
            elif 'llava' in self.model_name.lower():
                return self._query_llava(pil_image, prompt, max_tokens, temperature)
            else:
                return ""
        except Exception as e:
            print(f"    - VLM query failed: {e}")
            return ""
    
    def _to_pil_image(self, image: Union[np.ndarray, Image.Image, str]) -> Optional[Image.Image]:
        """Convert various image formats to PIL Image."""
        try:
            if isinstance(image, str):
                return Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return Image.fromarray(image)
            elif isinstance(image, Image.Image):
                return image.convert('RGB')
            else:
                print(f"    - Unsupported image type: {type(image)}")
                return None
        except Exception as e:
            print(f"    - Image conversion failed: {e}")
            return None
    
    def _query_qwen(self, image: Image.Image, prompt: str, max_tokens: int, temperature: float) -> str:
        """Query Qwen-VL-Chat model."""
        import tempfile
        import os
        
        # Save image temporarily (Qwen-VL expects file path)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            image.save(temp_path)
        
        try:
            query = self.tokenizer.from_list_format([
                {'image': temp_path},
                {'text': prompt},
            ])
            
            response, _ = self.model.chat(
                self.tokenizer,
                query=query,
                history=None,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0
            )
            
            return response.strip()
        finally:
            os.unlink(temp_path)
    
    def _query_llava(self, image: Image.Image, prompt: str, max_tokens: int, temperature: float) -> str:
        """Query LLaVA-1.5 model."""
        import torch
        
        # Format prompt for LLaVA
        formatted_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        
        inputs = self.processor(
            text=formatted_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant's response
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        
        return response
    
    def extract_fields_json(
        self,
        image: Union[np.ndarray, Image.Image],
        field_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Extract structured fields from a document image.
        
        Args:
            image: Document image
            field_names: List of fields to extract (e.g., ['sender', 'date', 'subject'])
                        If None, extracts common form fields.
        
        Returns:
            dict: Extracted field values
        """
        if field_names is None:
            field_names = ['sender', 'recipient', 'date', 'subject', 'reference_number']
        
        fields_str = ', '.join(field_names)
        
        prompt = f"""Analyze this document image and extract the following fields: {fields_str}.

Return your response as a valid JSON object with the field names as keys.
If a field is not found, use null as the value.
Only include the JSON object in your response, no additional text.

Example format:
{{"sender": "John Smith", "date": "2024-01-15", "subject": null}}"""
        
        response = self.query_image(image, prompt)
        
        # Parse JSON from response
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {}
        except json.JSONDecodeError:
            print(f"    - Failed to parse VLM JSON response: {response[:100]}...")
            return {}
    
    def confirm_ocr_text(
        self,
        image: Union[np.ndarray, Image.Image],
        ocr_text: str,
        region_description: str = "text region"
    ) -> str:
        """
        Use VLM to confirm or correct OCR output.
        
        This is the "Confirmation Mechanism" for handling OCR errors on complex forms.
        
        Args:
            image: Image of the text region
            ocr_text: Original OCR output to verify
            region_description: Description of what the region contains
        
        Returns:
            str: Confirmed or corrected text
        """
        prompt = f"""I ran OCR on this {region_description} and got: "{ocr_text}"

Please look at the image and tell me:
1. Is the OCR result correct?
2. If not, what is the correct text?

Respond with ONLY the correct text, nothing else. If the OCR is correct, repeat it exactly."""
        
        response = self.query_image(image, prompt, max_tokens=100)
        
        # Clean up response
        response = response.strip()
        response = re.sub(r'^["\']|["\']$', '', response)  # Remove quotes
        
        return response if response else ocr_text
    
    def classify_document(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Classify the document type using VLM.
        
        Args:
            image: Full document image
        
        Returns:
            dict: Classification result with type and confidence
        """
        prompt = """Classify this document. What type is it?

Options:
- fax_cover_sheet
- letter
- invoice
- form
- report
- memo
- contract
- receipt
- other

Respond with ONLY the document type from the options above, nothing else."""
        
        response = self.query_image(image, prompt, max_tokens=50)
        
        # Normalize response
        doc_type = response.lower().strip().replace(' ', '_')
        
        valid_types = ['fax_cover_sheet', 'letter', 'invoice', 'form', 'report', 
                       'memo', 'contract', 'receipt', 'other']
        
        if doc_type not in valid_types:
            doc_type = 'other'
        
        return {
            'document_type': doc_type,
            'raw_response': response
        }
    
    def describe_handwriting(self, image: Union[np.ndarray, Image.Image]) -> str:
        """
        Get a detailed description/transcription of handwritten content.
        
        Args:
            image: Image containing handwriting
        
        Returns:
            str: Transcribed/described handwriting content
        """
        prompt = """This image contains handwritten text. Please:
1. Transcribe what you can read
2. Note any parts that are unclear

Provide the transcription directly without any formatting or explanation."""
        
        return self.query_image(image, prompt, max_tokens=200)


# =============================================================================
#  CONVENIENCE FUNCTIONS
# =============================================================================

_vlm_manager_instance = None


def get_vlm_manager() -> VLMManager:
    """Get the singleton VLM Manager instance."""
    global _vlm_manager_instance
    if _vlm_manager_instance is None:
        _vlm_manager_instance = VLMManager()
    return _vlm_manager_instance


def query_image_vlm(image, prompt: str) -> str:
    """Convenience function to query VLM without managing the instance."""
    return get_vlm_manager().query_image(image, prompt)


def extract_fields_vlm(image, field_names: List[str] = None) -> Dict[str, Any]:
    """Convenience function to extract fields using VLM."""
    return get_vlm_manager().extract_fields_json(image, field_names)


def confirm_ocr_vlm(image, ocr_text: str, region_description: str = "text region") -> str:
    """Convenience function to confirm OCR using VLM."""
    return get_vlm_manager().confirm_ocr_text(image, ocr_text, region_description)
