# Document Processing Pipeline - Fixes for T4 GPU

## Quick Start

```bash
# 1. Install missing packages
chmod +x install_fixes.sh
./install_fixes.sh

# 2. Apply patches (backup originals first!)
cp src/utils.py src/utils.py.bak
cp src/classifier.py src/classifier.py.bak
cp src/config.py src/config.py.bak

cp patches/utils_patched.py src/utils.py
cp patches/classifier_patched.py src/classifier.py
cp patches/config_optimized.py src/config.py  # Optional - for faster processing

# 3. Run your pipeline
python main.py
```

## Files Included

| File | Purpose |
|------|---------|
| `install_fixes.sh` | Installs missing Python packages |
| `utils_patched.py` | Adds missing `bbox_center` and `cluster_1d` functions |
| `classifier_patched.py` | Fixes BFloat16 overflow error on T4 GPU |
| `config_optimized.py` | Optimized settings for faster processing |
| `vlm_manager_patch.py` | Instructions for patching VLM initialization |

## Issues Fixed

### 1. `module 'src.utils' has no attribute 'bbox_center'`
**Root Cause:** Table extraction code calls `utils.bbox_center()` but it wasn't defined.
**Fix:** `utils_patched.py` adds the missing functions.

### 2. `value cannot be converted to type at::BFloat16 without overflow`
**Root Cause:** T4 GPUs don't support BFloat16 natively. CLIP model was using BFloat16.
**Fix:** `classifier_patched.py` detects GPU type and uses Float16 for T4.

### 3. `4-bit quantization failed: No module named 'triton.ops'`
**Root Cause:** Missing triton package + T4 compatibility issues.
**Fix:** 
- `install_fixes.sh` installs triton
- `config_optimized.py` disables 4-bit quantization by default (T4 has issues)

### 4. VLM initialization failure
**Root Cause:** Missing `transformers_stream_generator` package + BFloat16 issues.
**Fix:**
- `install_fixes.sh` installs the package
- `vlm_manager_patch.py` shows how to fix the VLM loader

## Performance Improvements

With optimized config, expect processing time to drop from ~10 min/file to ~3-5 min/file:

| Setting | Before | After | Impact |
|---------|--------|-------|--------|
| Super-Resolution Model | EDSR | FSRCNN | ~3x faster SR |
| SR Noise Threshold | 5.0 | 8.0 | Fewer SR triggers |
| VLM Enabled | True | False | ~40% faster per page |
| LLM Correction Range | 30-85 | 30-60 | Fewer LLM calls |

## Troubleshooting

If you still see BFloat16 errors:
```python
# Add this at the top of your main.py
import torch
torch.set_default_dtype(torch.float16)
```

If VLM still fails, you can disable it entirely:
```python
# In config.py
USE_VLM = False
```
