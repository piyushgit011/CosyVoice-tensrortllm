#!/usr/bin/env python3
"""
Comprehensive streaming TTS benchmark for CosyVoice3 with vLLM
Tests TTFB (Time To First Byte), RTF (Real-Time Factor), and concurrency
"""
import sys
sys.path.append('third_party/Matcha-TTS')

import time
import asyncio
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import torchaudio
from tqdm import tqdm

from vllm import ModelRegistry
from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed


@dataclass
class BenchmarkResult:
    """Results from a single TTS request"""
    request_id: int
    text_length: int
    audio_duration_sec: float
    total_latency_sec: float
    ttfb_sec: float  # Time to first audio chunk
    rtf: float  # Real-Time Factor
    throughput_chars_per_sec: float
    num_chunks: int
    success: bool
    error: str = ""


class StreamingBenchmark:
    def __init__(self, model_dir: str, load_vllm: bool = True, fp16: bool = False):
        """Initialize CosyVoice3 model with vLLM"""
        print(f"Loading model from {model_dir}...")
        self.cosyvoice = AutoModel(
            model_dir=model_dir,
            load_trt=True,
            load_vllm=load_vllm,
            fp16=fp16
        )
        self.sample_rate = self.cosyvoice.sample_rate
        print(f"Model loaded successfully! Sample rate: {self.sample_rate}")

    def run_single_inference(
        self,
        request_id: int,
        text: str,
        prompt_text: str,
        prompt_wav: str,
        stream: bool = True,
        save_audio: bool = False
    ) -> BenchmarkResult:
        """Run a single streaming TTS inference and measure metrics"""
        try:
            start_time = time.time()
            ttfb = None
            num_chunks = 0
            total_audio_samples = 0
            audio_chunks = []

            # Stream inference
            for chunk_data in self.cosyvoice.inference_zero_shot(
                text,
                prompt_text,
                prompt_wav,
                stream=stream
            ):
                if ttfb is None:
                    ttfb = time.time() - start_time
                
                num_chunks += 1
                audio_tensor = chunk_data['tts_speech']
                total_audio_samples += audio_tensor.shape[1]
                
                if save_audio:
                    audio_chunks.append(audio_tensor)

            total_latency = time.time() - start_time
            audio_duration = total_audio_samples / self.sample_rate
            rtf = total_latency / audio_duration if audio_duration > 0 else float('inf')

            # Save audio if requested
            if save_audio and audio_chunks:
                import torch
                full_audio = torch.cat(audio_chunks, dim=1)
                torchaudio.save(
                    f'benchmark_output_{request_id}.wav',
                    full_audio,
                    self.sample_rate
                )

            return BenchmarkResult(
                request_id=request_id,
                text_length=len(text),
                audio_duration_sec=audio_duration,
                total_latency_sec=total_latency,
                ttfb_sec=ttfb,
                rtf=rtf,
                throughput_chars_per_sec=len(text) / total_latency,
                num_chunks=num_chunks,
                success=True
            )

        except Exception as e:
            return BenchmarkResult(
                request_id=request_id,
                text_length=len(text),
                audio_duration_sec=0,
                total_latency_sec=0,
                ttfb_sec=0,
                rtf=0,
                throughput_chars_per_sec=0,
                num_chunks=0,
                success=False,
                error=str(e)
            )

    def run_concurrent_benchmark(
        self,
        texts: List[str],
        prompt_text: str,
        prompt_wav: str,
        concurrency: int,
        stream: bool = True,
        save_audio: bool = False
    ) -> List[BenchmarkResult]:
        """Run multiple concurrent requests"""
        results = []
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for i, text in enumerate(texts):
                future = executor.submit(
                    self.run_single_inference,
                    i,
                    text,
                    prompt_text,
                    prompt_wav,
                    stream,
                    save_audio and i == 0  # Only save first audio
                )
                futures.append(future)
            
            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Concurrency {concurrency}"):
                results.append(future.result())
        
        return results


def print_statistics(results: List[BenchmarkResult], concurrency: int):
    """Print comprehensive statistics"""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    if not successful:
        print(f"❌ All requests failed for concurrency {concurrency}")
        return
    
    ttfbs = [r.ttfb_sec for r in successful]
    rtfs = [r.rtf for r in successful]
    latencies = [r.total_latency_sec for r in successful]
    throughputs = [r.throughput_chars_per_sec for r in successful]
    
    print(f"\n{'='*80}")
    print(f"📊 BENCHMARK RESULTS - Concurrency: {concurrency}")
    print(f"{'='*80}")
    print(f"Total Requests: {len(results)} | Successful: {len(successful)} | Failed: {len(failed)}")
    print(f"\n⏱️  TTFB (Time To First Byte) - Lower is better:")
    print(f"   Min:    {min(ttfbs):.4f}s")
    print(f"   Mean:   {np.mean(ttfbs):.4f}s")
    print(f"   Median: {np.median(ttfbs):.4f}s")
    print(f"   P95:    {np.percentile(ttfbs, 95):.4f}s")
    print(f"   Max:    {max(ttfbs):.4f}s")
    
    print(f"\n🎵 RTF (Real-Time Factor) - Lower is better:")
    print(f"   Min:    {min(rtfs):.4f}")
    print(f"   Mean:   {np.mean(rtfs):.4f}")
    print(f"   Median: {np.median(rtfs):.4f}")
    print(f"   P95:    {np.percentile(rtfs, 95):.4f}")
    print(f"   Max:    {max(rtfs):.4f}")
    
    print(f"\n⚡ Total Latency:")
    print(f"   Min:    {min(latencies):.4f}s")
    print(f"   Mean:   {np.mean(latencies):.4f}s")
    print(f"   Median: {np.median(latencies):.4f}s")
    print(f"   P95:    {np.percentile(latencies, 95):.4f}s")
    print(f"   Max:    {max(latencies):.4f}s")
    
    print(f"\n📈 Throughput (chars/sec):")
    print(f"   Min:    {min(throughputs):.2f}")
    print(f"   Mean:   {np.mean(throughputs):.2f}")
    print(f"   Max:    {max(throughputs):.2f}")
    
    # Calculate aggregate throughput
    total_chars = sum(r.text_length for r in successful)
    max_latency = max(latencies)
    aggregate_throughput = total_chars / max_latency
    print(f"\n🚀 Aggregate System Throughput: {aggregate_throughput:.2f} chars/sec")
    
    if failed:
        print(f"\n⚠️  Failed Requests: {len(failed)}")
        for r in failed[:5]:  # Show first 5 errors
            print(f"   Request {r.request_id}: {r.error}")


def save_results_json(all_results: Dict[int, List[BenchmarkResult]], filename: str):
    """Save all results to JSON"""
    output = {}
    for concurrency, results in all_results.items():
        output[f"concurrency_{concurrency}"] = [asdict(r) for r in results]
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Full results saved to: {filename}")


def main():
    # Test sentences of varying lengths
    test_texts = [
        # Short sentences
        "你好，我是语音合成系统。",
        "今天天气真好。",
        "欢迎使用CosyVoice。",
        
        # Medium sentences
        "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
        "人工智能技术正在快速发展，语音合成是其中一个重要的应用方向。",
        
        # Long sentences
        "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。这是一段绕口令，用来测试语音合成系统对复杂文本的处理能力。",
        "在科技飞速发展的今天，深度学习和大语言模型正在改变我们的生活方式，语音合成技术也取得了突破性的进展，能够生成更加自然流畅的人声。",
    ]
    
    # Configuration
    model_dir = 'pretrained_models/Fun-CosyVoice3-0.5B'
    prompt_text = 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。'
    prompt_wav = './asset/zero_shot_prompt.wav'
    
    # Concurrency levels to test
    concurrency_levels = [1, 2, 4, 8, 16]
    
    # Initialize benchmark
    print("🚀 Starting CosyVoice3 Streaming Benchmark with vLLM")
    print(f"Model: {model_dir}")
    print(f"Test texts: {len(test_texts)}")
    print(f"Concurrency levels: {concurrency_levels}")
    print(f"\n⏳ Loading model...")
    
    benchmark = StreamingBenchmark(model_dir=model_dir, load_vllm=True, fp16=False)
    
    # Warmup
    print("\n🔥 Warming up model...")
    set_all_random_seed(42)
    benchmark.run_single_inference(
        0,
        test_texts[0],
        prompt_text,
        prompt_wav,
        stream=True,
        save_audio=False
    )
    print("✅ Warmup complete!\n")
    
    # Run benchmarks
    all_results = {}
    
    for concurrency in concurrency_levels:
        print(f"\n{'='*80}")
        print(f"🧪 Testing with concurrency: {concurrency}")
        print(f"{'='*80}")
        
        # Prepare requests (cycle through test texts to have enough requests)
        num_requests = concurrency * 3  # 3 rounds per concurrency level
        texts = [test_texts[i % len(test_texts)] for i in range(num_requests)]
        
        # Run benchmark
        set_all_random_seed(42)
        results = benchmark.run_concurrent_benchmark(
            texts=texts,
            prompt_text=prompt_text,
            prompt_wav=prompt_wav,
            concurrency=concurrency,
            stream=True,
            save_audio=(concurrency == 1)  # Save audio only for single request
        )
        
        all_results[concurrency] = results
        print_statistics(results, concurrency)
        
        # Brief pause between tests
        time.sleep(2)
    
    # Save all results
    save_results_json(all_results, 'benchmark_results.json')
    
    # Final summary
    print(f"\n\n{'='*80}")
    print("📊 SUMMARY ACROSS ALL CONCURRENCY LEVELS")
    print(f"{'='*80}")
    
    print(f"\n{'Concurrency':<15} {'Mean TTFB (s)':<15} {'P95 TTFB (s)':<15} {'Mean RTF':<15} {'P95 RTF':<15}")
    print(f"{'-'*75}")
    
    for concurrency in concurrency_levels:
        results = [r for r in all_results[concurrency] if r.success]
        if results:
            ttfbs = [r.ttfb_sec for r in results]
            rtfs = [r.rtf for r in results]
            print(f"{concurrency:<15} {np.mean(ttfbs):<15.4f} {np.percentile(ttfbs, 95):<15.4f} {np.mean(rtfs):<15.4f} {np.percentile(rtfs, 95):<15.4f}")
    
    print("\n✅ Benchmark complete!")


if __name__ == '__main__':
    main()
