#!/usr/bin/env python3
"""
Benchmark CosyVoice3 with quantization for lower latency
Tests different quantization methods: FP16, INT8, etc.
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

from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed


@dataclass
class QuantBenchmarkResult:
    """Results for quantization benchmark"""
    config_name: str
    fp16: bool
    load_vllm: bool
    text_length: int
    audio_duration_sec: float
    total_latency_sec: float
    ttfb_sec: float
    rtf: float
    memory_allocated_mb: float
    success: bool
    error: str = ""


class QuantizationBenchmark:
    def __init__(self):
        self.test_texts = [
            "你好，我是语音合成系统。",
            "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
            "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。这是一段绕口令，用来测试语音合成系统对复杂文本的处理能力。",
        ]
        
        self.prompt_text = 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。'
        self.prompt_wav = './asset/zero_shot_prompt.wav'
        self.model_dir = 'pretrained_models/Fun-CosyVoice3-0.5B'

    def test_configuration(
        self,
        config_name: str,
        load_vllm: bool,
        fp16: bool
    ) -> QuantBenchmarkResult:
        """Test a specific model configuration"""
        import torch
        
        try:
            print(f"\n{'='*80}")
            print(f"Testing configuration: {config_name}")
            print(f"  vLLM: {load_vllm}, FP16: {fp16}")
            print(f"{'='*80}")
            
            # Load model
            start_load = time.time()
            print("Loading model...")
            model = AutoModel(
                model_dir=self.model_dir,
                load_trt=True,
                load_vllm=load_vllm,
                fp16=fp16
            )
            load_time = time.time() - start_load
            print(f"✅ Model loaded in {load_time:.2f}s")
            
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
            
            # Get memory usage
            memory_allocated = 0
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
                torch.cuda.reset_peak_memory_stats()
            
            result = QuantBenchmarkResult(
                config_name=config_name,
                fp16=fp16,
                load_vllm=load_vllm,
                text_length=sum(len(t) for t in self.test_texts),
                audio_duration_sec=total_audio_duration,
                total_latency_sec=sum(all_latencies),
                ttfb_sec=np.mean(all_ttfbs),
                rtf=np.mean(all_rtfs),
                memory_allocated_mb=memory_allocated,
                success=True
            )
            
            print(f"\n📊 Results:")
            print(f"   Mean TTFB: {result.ttfb_sec:.4f}s")
            print(f"   Mean RTF:  {result.rtf:.4f}")
            print(f"   Memory:    {result.memory_allocated_mb:.2f} MB")
            
            # Cleanup
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            print(f"❌ Configuration failed: {str(e)}")
            return QuantBenchmarkResult(
                config_name=config_name,
                fp16=fp16,
                load_vllm=load_vllm,
                text_length=0,
                audio_duration_sec=0,
                total_latency_sec=0,
                ttfb_sec=0,
                rtf=0,
                memory_allocated_mb=0,
                success=False,
                error=str(e)
            )

    def run_all_configurations(self) -> List[QuantBenchmarkResult]:
        """Test all quantization configurations"""
        configurations = [
            # vLLM configurations
            ("vLLM + FP32", True, False),
            ("vLLM + FP16", True, True),
            
            # Non-vLLM configurations for comparison
            ("PyTorch + FP32", False, False),
            ("PyTorch + FP16", False, True),
        ]
        
        results = []
        for config_name, load_vllm, fp16 in configurations:
            result = self.test_configuration(config_name, load_vllm, fp16)
            results.append(result)
            
            # Pause between tests
            time.sleep(3)
        
        return results


def print_comparison(results: List[QuantBenchmarkResult]):
    """Print comparison table"""
    print(f"\n\n{'='*80}")
    print("📊 QUANTIZATION COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    
    successful = [r for r in results if r.success]
    
    if not successful:
        print("❌ All configurations failed")
        return
    
    print(f"{'Configuration':<25} {'TTFB (ms)':<12} {'RTF':<10} {'Memory (MB)':<15} {'Status'}")
    print(f"{'-'*85}")
    
    for result in results:
        if result.success:
            status = "✅"
            ttfb_ms = result.ttfb_sec * 1000
            print(f"{result.config_name:<25} {ttfb_ms:<12.2f} {result.rtf:<10.4f} {result.memory_allocated_mb:<15.2f} {status}")
        else:
            status = "❌"
            print(f"{result.config_name:<25} {'N/A':<12} {'N/A':<10} {'N/A':<15} {status}")
    
    # Find best configuration
    print("\n🏆 Best Configurations:")
    
    # Best TTFB
    best_ttfb = min(successful, key=lambda x: x.ttfb_sec)
    print(f"   Lowest TTFB:  {best_ttfb.config_name} ({best_ttfb.ttfb_sec*1000:.2f}ms)")
    
    # Best RTF
    best_rtf = min(successful, key=lambda x: x.rtf)
    print(f"   Lowest RTF:   {best_rtf.config_name} ({best_rtf.rtf:.4f})")
    
    # Lowest memory
    best_mem = min(successful, key=lambda x: x.memory_allocated_mb)
    print(f"   Lowest Memory: {best_mem.config_name} ({best_mem.memory_allocated_mb:.2f} MB)")
    
    # Speedup comparison
    if any(not r.load_vllm for r in successful):
        baseline = next((r for r in successful if not r.load_vllm and not r.fp16), None)
        if baseline:
            print(f"\n⚡ Speedup vs PyTorch FP32 baseline:")
            for result in successful:
                if result.config_name != baseline.config_name:
                    speedup = baseline.rtf / result.rtf
                    print(f"   {result.config_name}: {speedup:.2f}x faster")


def main():
    print("🚀 CosyVoice3 Quantization Benchmark")
    print("Testing different configurations for optimal latency\n")
    
    benchmark = QuantizationBenchmark()
    results = benchmark.run_all_configurations()
    
    # Print comparison
    print_comparison(results)
    
    # Save results
    output = [asdict(r) for r in results]
    with open('quantization_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: quantization_results.json")
    print("\n✅ Benchmark complete!")


if __name__ == '__main__':
    main()
