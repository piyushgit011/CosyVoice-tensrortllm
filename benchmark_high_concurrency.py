#!/usr/bin/env python3
"""
High Concurrency Benchmark - Push RTX 5090 to its limits
Tests very high concurrency (20, 32, 48, 64) to maximize GPU utilization
"""
import sys
sys.path.append('third_party/Matcha-TTS')

import time
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from tqdm import tqdm

from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed


@dataclass
class HighConcurrencyResult:
    """Results from high concurrency test"""
    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    mean_ttfb_sec: float
    p50_ttfb_sec: float
    p95_ttfb_sec: float
    p99_ttfb_sec: float
    max_ttfb_sec: float
    mean_rtf: float
    p95_rtf: float
    max_rtf: float
    aggregate_throughput_chars_per_sec: float
    total_duration_sec: float
    peak_memory_mb: float
    avg_memory_mb: float


class HighConcurrencyBenchmark:
    def __init__(self, model_dir: str):
        """Initialize model once for all tests"""
        print(f"🚀 Loading CosyVoice3 with vLLM + FP16...")
        self.cosyvoice = AutoModel(
            model_dir=model_dir,
            load_trt=True,
            load_vllm=True,
            fp16=True  # Best configuration from previous tests
        )
        self.sample_rate = self.cosyvoice.sample_rate
        print(f"✅ Model loaded! Sample rate: {self.sample_rate}")
        
        # Diverse test sentences
        self.test_texts = [
            "你好，我是语音合成系统。",
            "今天天气真好。",
            "欢迎使用CosyVoice。",
            "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
            "人工智能技术正在快速发展，语音合成是其中一个重要的应用方向。",
            "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。这是一段绕口令，用来测试语音合成系统对复杂文本的处理能力。",
            "在科技飞速发展的今天，深度学习和大语言模型正在改变我们的生活方式，语音合成技术也取得了突破性的进展，能够生成更加自然流畅的人声。",
        ]
        
        self.prompt_text = 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。'
        self.prompt_wav = './asset/zero_shot_prompt.wav'

    def run_single_request(self, request_id: int, text: str):
        """Run a single TTS request"""
        try:
            start_time = time.time()
            ttfb = None
            num_samples = 0
            
            for chunk_data in self.cosyvoice.inference_zero_shot(
                text,
                self.prompt_text,
                self.prompt_wav,
                stream=True
            ):
                if ttfb is None:
                    ttfb = time.time() - start_time
                num_samples += chunk_data['tts_speech'].shape[1]
            
            total_latency = time.time() - start_time
            audio_duration = num_samples / self.sample_rate
            rtf = total_latency / audio_duration if audio_duration > 0 else float('inf')
            
            return {
                'success': True,
                'ttfb': ttfb,
                'rtf': rtf,
                'latency': total_latency,
                'text_length': len(text)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def test_concurrency(self, concurrency: int, rounds: int = 3) -> HighConcurrencyResult:
        """Test a specific concurrency level"""
        print(f"\n{'='*80}")
        print(f"🧪 Testing Concurrency: {concurrency} ({rounds} rounds)")
        print(f"{'='*80}")
        
        # Prepare requests
        num_requests = concurrency * rounds
        texts = [self.test_texts[i % len(self.test_texts)] for i in range(num_requests)]
        
        # Track memory before test
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        
        # Run concurrent requests
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(self.run_single_request, i, text)
                for i, text in enumerate(texts)
            ]
            
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc=f"Concurrency {concurrency}"):
                results.append(future.result())
        
        total_duration = time.time() - start_time
        
        # Track memory after test
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            avg_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        else:
            peak_memory = avg_memory = 0
        
        # Analyze results
        successful = [r for r in results if r['success']]
        failed = len(results) - len(successful)
        
        if not successful:
            return HighConcurrencyResult(
                concurrency=concurrency,
                total_requests=num_requests,
                successful_requests=0,
                failed_requests=failed,
                mean_ttfb_sec=0, p50_ttfb_sec=0, p95_ttfb_sec=0, p99_ttfb_sec=0, max_ttfb_sec=0,
                mean_rtf=0, p95_rtf=0, max_rtf=0,
                aggregate_throughput_chars_per_sec=0,
                total_duration_sec=total_duration,
                peak_memory_mb=peak_memory,
                avg_memory_mb=avg_memory
            )
        
        ttfbs = [r['ttfb'] for r in successful]
        rtfs = [r['rtf'] for r in successful]
        total_chars = sum(r['text_length'] for r in successful)
        
        return HighConcurrencyResult(
            concurrency=concurrency,
            total_requests=num_requests,
            successful_requests=len(successful),
            failed_requests=failed,
            mean_ttfb_sec=np.mean(ttfbs),
            p50_ttfb_sec=np.percentile(ttfbs, 50),
            p95_ttfb_sec=np.percentile(ttfbs, 95),
            p99_ttfb_sec=np.percentile(ttfbs, 99),
            max_ttfb_sec=np.max(ttfbs),
            mean_rtf=np.mean(rtfs),
            p95_rtf=np.percentile(rtfs, 95),
            max_rtf=np.max(rtfs),
            aggregate_throughput_chars_per_sec=total_chars / total_duration,
            total_duration_sec=total_duration,
            peak_memory_mb=peak_memory,
            avg_memory_mb=avg_memory
        )


def print_results(results: List[HighConcurrencyResult]):
    """Print comprehensive results"""
    print(f"\n\n{'='*100}")
    print("📊 HIGH CONCURRENCY BENCHMARK RESULTS")
    print(f"{'='*100}\n")
    
    print(f"{'Conc':<6} {'Total':<7} {'Success':<8} {'Mean TTFB':<12} {'P95 TTFB':<12} {'P99 TTFB':<12} {'Mean RTF':<10} {'P95 RTF':<10}")
    print(f"{'-'*100}")
    
    for r in results:
        if r.successful_requests > 0:
            print(f"{r.concurrency:<6} {r.total_requests:<7} {r.successful_requests:<8} "
                  f"{r.mean_ttfb_sec:<12.3f} {r.p95_ttfb_sec:<12.3f} {r.p99_ttfb_sec:<12.3f} "
                  f"{r.mean_rtf:<10.3f} {r.p95_rtf:<10.3f}")
        else:
            print(f"{r.concurrency:<6} {r.total_requests:<7} {r.successful_requests:<8} {'FAILED':<12}")
    
    print(f"\n{'='*100}")
    print("📈 THROUGHPUT & MEMORY ANALYSIS")
    print(f"{'='*100}\n")
    
    print(f"{'Concurrency':<12} {'Throughput (chars/s)':<25} {'Duration (s)':<15} {'Peak Memory (MB)':<20}")
    print(f"{'-'*100}")
    
    for r in results:
        if r.successful_requests > 0:
            print(f"{r.concurrency:<12} {r.aggregate_throughput_chars_per_sec:<25.2f} "
                  f"{r.total_duration_sec:<15.2f} {r.peak_memory_mb:<20.1f}")
    
    # Find optimal concurrency
    print(f"\n{'='*100}")
    print("🎯 OPTIMAL CONCURRENCY ANALYSIS")
    print(f"{'='*100}\n")
    
    successful_results = [r for r in results if r.successful_requests > 0]
    
    # Best throughput
    best_throughput = max(successful_results, key=lambda x: x.aggregate_throughput_chars_per_sec)
    print(f"🚀 Highest Throughput: Concurrency {best_throughput.concurrency}")
    print(f"   {best_throughput.aggregate_throughput_chars_per_sec:.2f} chars/sec")
    print(f"   {best_throughput.peak_memory_mb:.0f}MB peak memory")
    
    # Best for interactive (TTFB < 2s, RTF < 1.5)
    interactive = [r for r in successful_results if r.p95_ttfb_sec < 2.0 and r.p95_rtf < 1.5]
    if interactive:
        best_interactive = max(interactive, key=lambda x: x.concurrency)
        print(f"\n💬 Best for Interactive Apps: Concurrency {best_interactive.concurrency}")
        print(f"   P95 TTFB: {best_interactive.p95_ttfb_sec:.3f}s, P95 RTF: {best_interactive.p95_rtf:.3f}")
    
    # Saturation point (where RTF > 2.0)
    saturated = [r for r in successful_results if r.mean_rtf > 2.0]
    if saturated:
        saturation_point = min(saturated, key=lambda x: x.concurrency)
        print(f"\n⚠️  GPU Saturation Point: Concurrency {saturation_point.concurrency}")
        print(f"   Mean RTF exceeds 2.0 (non-real-time)")
    
    # Memory utilization
    print(f"\n💾 Memory Utilization:")
    max_memory = max(r.peak_memory_mb for r in successful_results)
    print(f"   Peak usage: {max_memory:.0f}MB / 32,607MB ({max_memory/32607*100:.1f}%)")
    print(f"   Remaining:  {32607-max_memory:.0f}MB available")


def main():
    print("="*100)
    print("🔥 HIGH CONCURRENCY STRESS TEST - RTX 5090")
    print("Testing extreme concurrency to maximize GPU utilization")
    print("="*100)
    
    model_dir = 'pretrained_models/Fun-CosyVoice3-0.5B'
    
    # Initialize benchmark (loads model once)
    benchmark = HighConcurrencyBenchmark(model_dir)
    
    # Warmup
    print("\n🔥 Warming up...")
    _ = benchmark.test_concurrency(concurrency=2, rounds=1)
    print("✅ Warmup complete!\n")
    
    # Test progressively higher concurrency
    concurrency_levels = [4, 8, 12, 16, 20, 24, 32, 40, 48]
    
    results = []
    for concurrency in concurrency_levels:
        result = benchmark.test_concurrency(concurrency, rounds=3)
        results.append(result)
        
        # Print quick summary
        print(f"\n📊 Quick Summary:")
        print(f"   Success Rate: {result.successful_requests}/{result.total_requests}")
        if result.successful_requests > 0:
            print(f"   P95 TTFB: {result.p95_ttfb_sec:.3f}s, P95 RTF: {result.p95_rtf:.3f}")
            print(f"   Throughput: {result.aggregate_throughput_chars_per_sec:.2f} chars/sec")
            print(f"   Peak Memory: {result.peak_memory_mb:.0f}MB")
        
        # Brief pause
        time.sleep(2)
        
        # Stop if we're seeing high failure rates
        if result.failed_requests > result.total_requests * 0.2:
            print("\n⚠️  High failure rate detected, stopping concurrency tests")
            break
    
    # Print comprehensive results
    print_results(results)
    
    # Save results
    output = [asdict(r) for r in results]
    with open('high_concurrency_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: high_concurrency_results.json")
    print("\n✅ High concurrency benchmark complete!")


if __name__ == '__main__':
    main()
