#!/usr/bin/env python
"""Download CosyVoice3 model"""
import os

# Try modelscope first, fallback to huggingface if needed
try:
    from modelscope import snapshot_download
    print("Downloading Fun-CosyVoice3-0.5B-2512 model from ModelScope...")
    snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', 
                     local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
    print("Model downloaded successfully!")
except Exception as e:
    print(f"ModelScope download failed: {e}")
    print("Trying HuggingFace...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', 
                         local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
        print("Model downloaded successfully from HuggingFace!")
    except Exception as e2:
        print(f"HuggingFace download also failed: {e2}")
        print("Please download manually")
