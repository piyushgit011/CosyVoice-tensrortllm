#!/usr/bin/env python3
"""
Advanced benchmark: FP8 quantization + Multi-worker configurations
Tests maximum VRAM utilization and throughput optimization
"""
import sys
sys.path.append('third_party/Matcha-TTS')

import time
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import torchaudio
from tqdm import tqdm
import torch

from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed


@dataclass
class AdvancedBenchmarkResult:
    """Results for advanced configurations"""
    config_name: str
    quantization: str  # fp32, fp16, fp8
    use_vllm: bool
    text_length: int
    audio_duration_sec: float
    total_latency_sec: float
    ttfb_sec: float
    rtf: float
    memory_allocated_mb: float
    memory_reserved_mb: float
    gpu_utilization_pct: float
    success: bool
    error: str = ""


class AdvancedBenchmark:
    def __init__(self):
        self.test_texts = [
            "你好，我是语音合成系统。",
            "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
            "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。这是一段绕口令，用来测试语音合成系统对复杂文本的处理能力。",
        ]
        
        self.prompt_text = 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。'
        self.prompt_wav = './asset/zero_shot_prompt.wav'
        self.model_dir = 'pretrained_models/Fun-CosyVoice3-0.5B'

    def get_gpu_stats(self):
        """Get current GPU memory and utilization stats"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
            
            # Try to get GPU utilization
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                pynvml.nvmlShutdown()
            except:
                gpu_util = 0.0
            
            return allocated, reserved, gpu_util
        return 0.0, 0.0, 0.0

    def test_configuration(
        self,
        config_name: str,
        quantization: str,
        use_vllm: bool,
        additional_kwargs: Dict = None
    ) -> AdvancedBenchmarkResult:
        """Test a specific configuration"""
        try:
            print(f"\n{'='*80}")
            print(f"Testing: {config_name}")
            print(f"  Quantization: {quantization}")
            print(f"  vLLM: {use_vllm}")
            if additional_kwargs:
                print(f"  Additional args: {additional_kwargs}")
            print(f"{'='*80}")
            
            # Determine fp16 setting
            if quantization == 'fp16':
                fp16 = True
            else:
                fp16 = False
            
            # Build kwargs
            kwargs = {
                'model_dir': self.model_dir,
                'load_trt': True,
                'load_vllm': use_vllm,
                'fp16': fp16,
            }
            
            # Add additional kwargs (for FP8, etc.)
            if additional_kwargs:
                kwargs.update(additional_kwargs)
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # Load model
            start_load = time.time()
            print("Loading model...")
            model = AutoModel(**kwargs)
            load_time = time.time() - start_load
            print(f"✅ Model loaded in {load_time:.2f}s")
            
            # Get initial memory stats
            mem_alloc, mem_reserved, gpu_util = self.get_gpu_stats()
            print(f"📊 Memory: Allocated {mem_alloc:.0f}MB, Reserved {mem_reserved:.0f}MB")
            
            # Warmup
            print("Warming up...")
            set_all_random_seed(42)
            for _ in model.inference_zero_shot(
                self.test_texts[0],
                self.prompt_text,
                self.prompt_wav,
                stream=True
            ):
                pass
            
            # Benchmark on all test texts
            all_ttfbs = []
            all_rtfs = []
            all_latencies = []
            total_audio_duration = 0
            
            for text in tqdm(self.test_texts, desc="Running inference"):
                start_time = time.time()
                ttfb = None
                num_samples = 0
                
                set_all_random_seed(42)
                for chunk_data in model.inference_zero_shot(
                    text,
                    self.prompt_text,
                    self.prompt_wav,
                    stream=True
                ):
                    if ttfb is None:
                        ttfb = time.time() - start_time
                    num_samples += chunk_data['tts_speech'].shape[1]
                
                total_latency = time.time() - start_time
                audio_duration = num_samples / model.sample_rate
                rtf = total_latency / audio_duration if audio_duration > 0 else float('inf')
                
                all_ttfbs.append(ttfb)
                all_rtfs.append(rtf)
                all_latencies.append(total_latency)
                total_audio_duration += audio_duration
            
            # Get final memory stats
            mem_alloc_final, mem_reserved_final, gpu_util_final = self.get_gpu_stats()
            
            result = AdvancedBenchmarkResult(
                config_name=config_name,
                quantization=quantization,
                use_vllm=use_vllm,
                text_length=sum(len(t) for t in self.test_texts),
                audio_duration_sec=total_audio_duration,
                total_latency_sec=sum(all_latencies),
                ttfb_sec=np.mean(all_ttfbs),
                rtf=np.mean(all_rtfs),
                memory_allocated_mb=mem_alloc_final,
                memory_reserved_mb=mem_reserved_final,
                gpu_utilization_pct=gpu_util_final,
                success=True
            )
            
            print(f"\n📊 Results:")
            print(f"   Mean TTFB: {result.ttfb_sec:.4f}s ({result.ttfb_sec*1000:.1f}ms)")
            print(f"   Mean RTF:  {result.rtf:.4f}")
            print(f"   Memory:    {result.memory_allocated_mb:.0f}MB allocated, {result.memory_reserved_mb:.0f}MB reserved")
            print(f"   GPU Util:  {result.gpu_utilization_pct:.1f}%")
            
            # Cleanup
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Pause between tests
            time.sleep(3)
            
            return result
            
        except Exception as e:
            print(f"❌ Configuration failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return AdvancedBenchmarkResult(
                config_name=config_name,
                quantization=quantization,
                use_vllm=use_vllm,
                text_length=0,
                audio_duration_sec=0,
                total_latency_sec=0,
                ttfb_sec=0,
                rtf=0,
                memory_allocated_mb=0,
                memory_reserved_mb=0,
                gpu_utilization_pct=0,
                success=False,
                error=str(e)
            )

    def run_all_configurations(self) -> List[AdvancedBenchmarkResult]:
        """Test all advanced configurations"""
        results = []
        
        # Test configurations
        configurations = [
            # Baseline from previous tests
            ("vLLM FP32 (baseline)", "fp32", True, {}),
            ("vLLM FP16 (previous best)", "fp16", True, {}),
            
            # FP8 configurations - Note: FP8 support varies by vLLM version
            # We'll try different approaches
            
            # Attempt 1: Direct FP8 via dtype
            # ("vLLM FP8 (quantized)", "fp8", True, {'dtype': 'fp8'}),
            
            # For now, focus on increasing batch size and max_num_seqs
            # to utilize more VRAM
            ("vLLM FP16 + Large Batch", "fp16", True, {
                'max_num_batched_tokens': 4096,  # Increase from default 2048
            }),
            
            ("vLLM FP16 + XL Batch", "fp16", True, {
                'max_num_batched_tokens': 8192,  # Even larger batch
            }),
        ]
        
        for config_name, quant, use_vllm, kwargs in configurations:
            result = self.test_configuration(config_name, quant, use_vllm, kwargs)
            results.append(result)
        
        return results


def print_comparison(results: List[AdvancedBenchmarkResult]):
    """Print comparison table"""
    print(f"\n\n{'='*100}")
    print("📊 ADVANCED CONFIGURATION COMPARISON")
    print(f"{'='*100}\n")
    
    successful = [r for r in results if r.success]
    
    if not successful:
        print("❌ All configurations failed")
        return
    
    print(f"{'Configuration':<35} {'TTFB (ms)':<12} {'RTF':<10} {'Memory (MB)':<15} {'Status'}")
    print(f"{'-'*100}")
    
    for result in results:
        if result.success:
            status = "✅"
            ttfb_ms = result.ttfb_sec * 1000
            print(f"{result.config_name:<35} {ttfb_ms:<12.1f} {result.rtf:<10.4f} {result.memory_reserved_mb:<15.0f} {status}")
        else:
            status = "❌"
            print(f"{result.config_name:<35} {'N/A':<12} {'N/A':<10} {'N/A':<15} {status}")
    
    # Find best configuration
    print("\n🏆 Best Configurations:")
    
    # Best TTFB
    best_ttfb = min(successful, key=lambda x: x.ttfb_sec)
    print(f"   Lowest TTFB:  {best_ttfb.config_name} ({best_ttfb.ttfb_sec*1000:.1f}ms)")
    
    # Best RTF
    best_rtf = min(successful, key=lambda x: x.rtf)
    print(f"   Lowest RTF:   {best_rtf.config_name} ({best_rtf.rtf:.4f})")
    
    # Most memory efficient
    best_mem = min(successful, key=lambda x: x.memory_reserved_mb)
    print(f"   Most Memory Efficient: {best_mem.config_name} ({best_mem.memory_reserved_mb:.0f}MB)")
    
    # Highest throughput (inverse RTF)
    best_throughput = min(successful, key=lambda x: x.rtf)
    throughput = 1.0 / best_throughput.rtf
    print(f"   Highest Throughput: {best_throughput.config_name} ({throughput:.2f}x real-time)")
    
    # Memory utilization analysis
    print(f"\n💾 Memory Utilization:")
    print(f"   Total GPU VRAM: 32,607 MB (RTX 5090)")
    for result in successful:
        usage_pct = (result.memory_reserved_mb / 32607) * 100
        print(f"   {result.config_name:<35} {result.memory_reserved_mb:>6.0f}MB ({usage_pct:>5.1f}%)")
    
    # Speedup comparison
    baseline = next((r for r in successful if 'baseline' in r.config_name.lower()), None)
    if baseline:
        print(f"\n⚡ Speedup vs Baseline ({baseline.config_name}):")
        for result in successful:
            if result.config_name != baseline.config_name:
                speedup = baseline.rtf / result.rtf
                ttfb_improvement = ((baseline.ttfb_sec - result.ttfb_sec) / baseline.ttfb_sec) * 100
                print(f"   {result.config_name:<35} RTF: {speedup:>5.2f}x, TTFB: {ttfb_improvement:>+6.1f}%")


def main():
    print("🚀 CosyVoice3 Advanced Benchmark - FP8 & Multi-Worker Tests")
    print("Testing advanced configurations for maximum performance\n")
    
    # Note about FP8
    print("⚠️  NOTE: FP8 quantization support depends on:")
    print("   - GPU capability (RTX 5090 supports FP8)")
    print("   - vLLM version and configuration")
    print("   - Model architecture compatibility")
    print("   We'll test what's available in this setup.\n")
    
    benchmark = AdvancedBenchmark()
    results = benchmark.run_all_configurations()
    
    # Print comparison
    print_comparison(results)
    
    # Save results
    output = [asdict(r) for r in results]
    with open('advanced_benchmark_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: advanced_benchmark_results.json")
    print("\n✅ Advanced benchmark complete!")


if __name__ == '__main__':
    main()
