#!/usr/bin/env python3
"""
TensorRT-LLM Benchmark for CosyVoice
Expects Triton server to be running (bash run.sh 3 3)
Tests maximum concurrency while maintaining RTF < 2.0 and low TTFB
"""
import tritonclient.grpc as grpcclient
import numpy as np
import time
import json
from dataclasses import dataclass, asdict
from typing import List
import concurrent.futures
from tqdm import tqdm
import soundfile as sf


@dataclass
class TRTLLMBenchmarkResult:
    """Results for TensorRT-LLM benchmark"""
    concurrency: int
    total_requests: int
    successful_requests: int
    mean_ttfb_sec: float
    p50_ttfb_sec: float
    p95_ttfb_sec: float
    p99_ttfb_sec: float
    mean_rtf: float
    p95_rtf: float
    p99_rtf: float
    aggregate_throughput_chars_per_sec: float
    total_duration_sec: float


class TensorRTLLMBenchmark:
    def __init__(self, server_url="localhost:8001", model_name="cosyvoice2"):
        """Initialize Triton client"""
        self.client = grpcclient.InferenceServerClient(url=server_url)
        self.model_name = model_name
        
        # Check if server is running
        if not self.client.is_server_live():
            raise ConnectionError("Triton server is not running! Start with: bash run.sh 3 3")
        
        print(f"✅ Connected to Triton server at {server_url}")
        print(f"   Model: {model_name}")
        
        # Test texts - diverse lengths
        self.test_texts = [
            "你好，我是语音合成系统。",
            "今天天气真好。",
            "欢迎使用CosyVoice。",
            "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。",
            "人工智能技术正在快速发展，语音合成是其中一个重要的应用方向。",
            "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。",
            "在科技飞速发展的今天，深度学习和大语言模型正在改变我们的生活方式，语音合成技术也取得了突破性的进展。",
        ]
        
        # Reference audio for zero-shot voice cloning
        self.reference_audio_path = "../../asset/zero_shot_prompt.wav"
        self.reference_text = "希望你以后能够做的比我还好呦。"

    def run_single_inference(self, request_id: int, text: str, streaming: bool = True):
        """Run a single TTS inference request"""
        try:
            start_time = time.time()
            ttfb = None
            audio_chunks = []
            
            # Prepare inputs
            text_input = grpcclient.InferInput("INPUT_TEXT", [1], "BYTES")
            text_input.set_data_from_numpy(np.array([text.encode('utf-8')], dtype=object))
            
            ref_text_input = grpcclient.InferInput("REFERENCE_TEXT", [1], "BYTES")
            ref_text_input.set_data_from_numpy(np.array([self.reference_text.encode('utf-8')], dtype=object))
            
            # Load reference audio
            ref_audio, sr = sf.read(self.reference_audio_path)
            ref_audio_input = grpcclient.InferInput("REFERENCE_AUDIO", ref_audio.shape, "FP32")
            ref_audio_input.set_data_from_numpy(ref_audio.astype(np.float32))
            
            # Output
            output = grpcclient.InferRequestedOutput("OUTPUT_AUDIO")
            
            inputs = [text_input, ref_text_input, ref_audio_input]
            outputs = [output]
            
            if streaming:
                # Streaming mode - measure TTFB
                def callback(result, error):
                    nonlocal ttfb, audio_chunks
                    if error:
                        return
                    if ttfb is None:
                        ttfb = time.time() - start_time
                    audio = result.as_numpy("OUTPUT_AUDIO")
                    audio_chunks.append(audio)
                
                self.client.async_stream_infer(
                    model_name=self.model_name,
                    inputs=inputs,
                    outputs=outputs,
                    request_id=str(request_id),
                    callback=callback
                )
                
                # Wait for completion
                time.sleep(0.1)  # Allow some processing
                while self.client.is_server_ready():
                    if ttfb is not None and len(audio_chunks) > 0:
                        break
                    time.sleep(0.01)
            else:
                # Offline mode
                result = self.client.infer(
                    model_name=self.model_name,
                    inputs=inputs,
                    outputs=outputs
                )
                ttfb = time.time() - start_time
                audio_chunks = [result.as_numpy("OUTPUT_AUDIO")]
            
            total_latency = time.time() - start_time
            
            # Calculate RTF
            total_samples = sum(chunk.size for chunk in audio_chunks)
            audio_duration = total_samples / 24000  # CosyVoice2 sample rate
            rtf = total_latency / audio_duration if audio_duration > 0 else float('inf')
            
            return {
                'success': True,
                'ttfb': ttfb or total_latency,
                'rtf': rtf,
                'latency': total_latency,
                'text_length': len(text),
                'audio_duration': audio_duration
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'ttfb': 0,
                'rtf': 0,
                'latency': 0,
                'text_length': len(text)
            }

    def test_concurrency(self, concurrency: int, rounds: int = 3) -> TRTLLMBenchmarkResult:
        """Test specific concurrency level"""
        print(f"\n{'='*80}")
        print(f"🧪 Testing TensorRT-LLM Concurrency: {concurrency} users ({rounds} rounds)")
        print(f"{'='*80}")
        
        num_requests = concurrency * rounds
        texts = [self.test_texts[i % len(self.test_texts)] for i in range(num_requests)]
        
        results = []
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(self.run_single_inference, i, text, streaming=True)
                for i, text in enumerate(texts)
            ]
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures), desc=f"Concurrency {concurrency}"):
                results.append(future.result())
        
        total_duration = time.time() - start_time
        
        # Analyze results
        successful = [r for r in results if r['success']]
        
        if not successful:
            print(f"❌ All requests failed!")
            return None
        
        ttfbs = [r['ttfb'] for r in successful]
        rtfs = [r['rtf'] for r in successful]
        total_chars = sum(r['text_length'] for r in successful)
        
        result = TRTLLMBenchmarkResult(
            concurrency=concurrency,
            total_requests=num_requests,
            successful_requests=len(successful),
            mean_ttfb_sec=np.mean(ttfbs),
            p50_ttfb_sec=np.percentile(ttfbs, 50),
            p95_ttfb_sec=np.percentile(ttfbs, 95),
            p99_ttfb_sec=np.percentile(ttfbs, 99),
            mean_rtf=np.mean(rtfs),
            p95_rtf=np.percentile(rtfs, 95),
            p99_rtf=np.percentile(rtfs, 99),
            aggregate_throughput_chars_per_sec=total_chars / total_duration,
            total_duration_sec=total_duration
        )
        
        print(f"\n📊 Results:")
        print(f"   Success: {len(successful)}/{num_requests}")
        print(f"   Mean TTFB: {result.mean_ttfb_sec*1000:.1f}ms (P95: {result.p95_ttfb_sec*1000:.1f}ms)")
        print(f"   Mean RTF: {result.mean_rtf:.4f} (P95: {result.p95_rtf:.4f})")
        print(f"   Throughput: {result.aggregate_throughput_chars_per_sec:.2f} chars/sec")
        
        return result


def find_max_concurrency_with_rtf_limit(benchmark, max_rtf=2.0, max_concurrency=200):
    """Binary search to find maximum concurrency while maintaining RTF < max_rtf"""
    print(f"\n{'='*80}")
    print(f"🔍 Finding Maximum Concurrency with RTF < {max_rtf}")
    print(f"{'='*80}\n")
    
    low, high = 1, max_concurrency
    best_concurrency = 1
    best_result = None
    
    tested = {}
    
    while low <= high:
        mid = (low + high) // 2
        print(f"\n🧪 Testing concurrency: {mid}")
        
        result = benchmark.test_concurrency(mid, rounds=2)
        tested[mid] = result
        
        if result and result.p95_rtf < max_rtf:
            # Can handle this concurrency
            best_concurrency = mid
            best_result = result
            low = mid + 1
            print(f"✅ P95 RTF {result.p95_rtf:.3f} < {max_rtf} - trying higher")
        else:
            # Too high
            high = mid - 1
            if result:
                print(f"❌ P95 RTF {result.p95_rtf:.3f} >= {max_rtf} - trying lower")
    
    return best_concurrency, best_result, tested


def main():
    print("="*100)
    print("🚀 TensorRT-LLM MAXIMUM CONCURRENCY BENCHMARK")
    print("   Finding max concurrency while maintaining RTF < 2.0")
    print("   Expected: 4x faster than vLLM, 100+ concurrent users possible")
    print("="*100)
    
    try:
        benchmark = TensorRTLLMBenchmark()
    except ConnectionError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo start Triton server:")
        print("  cd runtime/triton_trtllm")
        print("  bash run.sh 0 3  # Setup and start server")
        return
    
    # Warmup
    print("\n🔥 Warming up...")
    benchmark.test_concurrency(2, rounds=1)
    
    # Find maximum concurrency
    max_conc, best_result, all_results = find_max_concurrency_with_rtf_limit(
        benchmark, 
        max_rtf=2.0,
        max_concurrency=200
    )
    
    # Print summary
    print(f"\n\n{'='*100}")
    print("📊 TENSORRT-LLM BENCHMARK RESULTS")
    print(f"{'='*100}\n")
    
    print(f"🏆 Maximum Concurrency with RTF < 2.0: {max_conc} users")
    if best_result:
        print(f"   P95 TTFB: {best_result.p95_ttfb_sec*1000:.1f}ms")
        print(f"   P95 RTF: {best_result.p95_rtf:.4f}")
        print(f"   Throughput: {best_result.aggregate_throughput_chars_per_sec:.2f} chars/sec")
    
    # Compare with vLLM results
    print(f"\n📈 Comparison vs vLLM (from previous benchmarks):")
    print(f"   Configuration         Max Concurrency  P95 TTFB    P95 RTF    Throughput")
    print(f"   {'-'*85}")
    print(f"   vLLM + FP16                   8         1.94s       0.94      51.8 chars/s")
    print(f"   TensorRT-LLM             {max_conc:>5}     {best_result.p95_ttfb_sec:>8.3f}s   {best_result.p95_rtf:>8.3f}   {best_result.aggregate_throughput_chars_per_sec:>7.1f} chars/s")
    
    if max_conc > 8:
        improvement = max_conc / 8
        print(f"\n   🚀 TensorRT-LLM supports {improvement:.1f}x more concurrent users!")
    
    # Save results
    output = {
        'max_concurrency': max_conc,
        'best_result': asdict(best_result) if best_result else None,
        'all_tested': {k: asdict(v) if v else None for k, v in all_results.items()}
    }
    
    with open('tensorrtllm_benchmark_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: tensorrtllm_benchmark_results.json")
    print("\n✅ TensorRT-LLM benchmark complete!")


if __name__ == '__main__':
    main()
