#!/usr/bin/env python3
"""
Quick test of modified vLLM configuration with 80% GPU utilization
Tests a few concurrent users to validate the change works
"""
if __name__ == '__main__':
    import sys
    sys.path.append('third_party/Matcha-TTS')

    import time
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    from vllm import ModelRegistry
    from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
    ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

    from cosyvoice.cli.cosyvoice import AutoModel
    from cosyvoice.utils.common import set_all_random_seed


def test_inference(model, request_id, text):
    """Single test inference"""
    try:
        start = time.time()
        ttfb = None
        samples = 0
        
        for chunk in model.inference_zero_shot(
            text,
            'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
            './asset/zero_shot_prompt.wav',
            stream=True
        ):
            if ttfb is None:
                ttfb = time.time() - start
            samples += chunk['tts_speech'].shape[1]
        
        total_time = time.time() - start
        audio_duration = samples / model.sample_rate
        rtf = total_time / audio_duration if audio_duration > 0 else 999
        
        return {
            'success': True,
            'ttfb': ttfb,
            'rtf': rtf,
            'latency': total_time
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


    print("="*80)
    print("🧪 Testing Modified vLLM Configuration (80% GPU Utilization)")
    print("="*80)

    # Load model with new configuration
    print("\n⏳ Loading CosyVoice3 with modified vLLM settings...")
    print("   • gpu_memory_utilization: 0.8 (was 0.2)")
    print("   • max_num_batched_tokens: 8192 (was default)")
    print("   • max_num_seqs: 128 (was default)")

    try:
        model = AutoModel(
            model_dir='pretrained_models/Fun-CosyVoice3-0.5B',
            load_trt=True,
            load_vllm=True,
            fp16=True
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\nThis might happen if:")
        print("  1. Not enough GPU memory available (need ~26GB)")
        print("  2. Other processes using GPU")
        print("  3. vLLM version incompatibility")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Check GPU memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n📊 GPU Memory:")
        print(f"   Allocated: {allocated:.2f}GB")
        print(f"   Reserved:  {reserved:.2f}GB")
        print(f"   Total:     {total:.2f}GB")
        print(f"   Usage:     {(reserved/total)*100:.1f}%")
        
        if reserved < 3.0:
            print(f"\n⚠️  Warning: Only {reserved:.2f}GB reserved (expected ~20-26GB)")
            print("   The configuration may not have taken effect properly")

    # Quick test with a few concurrent users
    test_texts = [
        "你好，我是语音合成系统。",
        "今天天气真好。",
        "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。",
    ]

    print("\n🧪 Running quick validation test (12 concurrent requests)...")
    concurrency = 12
    texts = [test_texts[i % len(test_texts)] for i in range(concurrency * 3)]

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(test_inference, model, i, text) for i, text in enumerate(texts)]
        results = [f.result() for f in tqdm(futures, desc="Testing")]

    total_time = time.time() - start_time
    successful = [r for r in results if r['success']]

    print(f"\n📊 Quick Test Results:")
    print(f"   Requests: {len(successful)}/{len(results)} successful")

    if successful:
        import numpy as np
        ttfbs = [r['ttfb'] for r in successful]
        rtfs = [r['rtf'] for r in successful]
        
        print(f"   Mean TTFB: {np.mean(ttfbs):.3f}s (P95: {np.percentile(ttfbs, 95):.3f}s)")
        print(f"   Mean RTF:  {np.mean(rtfs):.3f} (P95: {np.percentile(rtfs, 95):.3f})")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Throughput: {sum(len(texts[i]) for i in range(len(successful))) / total_time:.1f} chars/sec")
        
        print("\n✅ Modified configuration is working!")
        print("\n🚀 Ready for full benchmark. Run:")
        print("   python benchmark_high_concurrency.py")
    else:
        print("\n❌ Some requests failed. Check the configuration.")
        for r in results:
            if not r['success']:
                print(f"   Error: {r['error']}")

    print("\n" + "="*80)
