#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai
from openvino import get_version
import os
import psutil
import hashlib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('models_path')
    parser.add_argument('adapter_path')
    parser.add_argument('lora_mode')
    parser.add_argument('device')
    args = parser.parse_args()

    device = args.device

    modes = {
        "auto": openvino_genai.AdapterConfig.Mode.MODE_AUTO,
        "fuse": openvino_genai.AdapterConfig.Mode.MODE_FUSE,
        "dynamic": openvino_genai.AdapterConfig.Mode.MODE_DYNAMIC,
        "static": openvino_genai.AdapterConfig.Mode.MODE_STATIC,
        "static_rank": openvino_genai.AdapterConfig.Mode.MODE_DYNAMIC
    }
    lora_mode = modes[args.lora_mode]
    print(f"LoRA adapters loading mode: {lora_mode}")

    adapter = openvino_genai.Adapter(args.adapter_path)
    adapter_config = openvino_genai.AdapterConfig(adapter, mode=lora_mode)
    config = {}
    if device == 'NPU':
        config["NPUW_DEVICES"] = "NPU"
        config["NPUW_LLM_MAX_LORA_RANK"] = "32"
        #config["NPUW_DUMP_SUBS"] = "YES"

    prompt = ['Help me concentrate']

    print(f'openvino runtime version: {get_version()}')

    rss_usage_gb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
    print(f"rss_usage:{rss_usage_gb:.3f} GB")

    pipe = openvino_genai.LLMPipeline(args.models_path, device, config, adapters=adapter_config)  # register all required adapters here

    print("Generate with LoRA adapter and alpha set to 0.75:")
    res1 = pipe.generate(prompt, max_new_tokens=100, adapters=openvino_genai.AdapterConfig(adapter, 0.75))
    print(res1.texts[0])
    results_md5 = (hashlib.new("md5", res1.texts[0].encode(), usedforsecurity=False).hexdigest())
    print(results_md5)
    perf_metrics = res1.perf_metrics
    print(f"Output token size: {perf_metrics.get_num_generated_tokens()}")
    print(f"Load time: {perf_metrics.get_load_time():.2f} ms")
    print(f"Generate time: {perf_metrics.get_generate_duration().mean:.2f} ± {perf_metrics.get_generate_duration().std:.2f} ms")
    print(f"Tokenization time: {perf_metrics.get_tokenization_duration().mean:.2f} ± {perf_metrics.get_tokenization_duration().std:.2f} ms")
    print(f"Detokenization time: {perf_metrics.get_detokenization_duration().mean:.2f} ± {perf_metrics.get_detokenization_duration().std:.2f} ms")
    print(f"TTFT: {perf_metrics.get_ttft().mean:.2f} ± {perf_metrics.get_ttft().std:.2f} ms")
    print(f"TPOT: {perf_metrics.get_tpot().mean:.2f} ± {perf_metrics.get_tpot().std:.2f} ms")
    print(f"Throughput : {perf_metrics.get_throughput().mean:.2f} ± {perf_metrics.get_throughput().std:.2f} tokens/s")
    rss_usage_gb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
    print(f"rss_usage:{rss_usage_gb:.3f} GB")

    print("\n-----------------------------")
    print("Generate without LoRA adapter:")
    res2 = pipe.generate(prompt, max_new_tokens=100, adapters=openvino_genai.AdapterConfig())
    print(res2.texts[0])
    results_md5 = (hashlib.new("md5", res2.texts[0].encode(), usedforsecurity=False).hexdigest())
    print(results_md5)
    perf_metrics = res2.perf_metrics
    print(f"Output token size: {perf_metrics.get_num_generated_tokens()}")
    print(f"Load time: {perf_metrics.get_load_time():.2f} ms")
    print(f"Generate time: {perf_metrics.get_generate_duration().mean:.2f} ± {perf_metrics.get_generate_duration().std:.2f} ms")
    print(f"Tokenization time: {perf_metrics.get_tokenization_duration().mean:.2f} ± {perf_metrics.get_tokenization_duration().std:.2f} ms")
    print(f"Detokenization time: {perf_metrics.get_detokenization_duration().mean:.2f} ± {perf_metrics.get_detokenization_duration().std:.2f} ms")
    print(f"TTFT: {perf_metrics.get_ttft().mean:.2f} ± {perf_metrics.get_ttft().std:.2f} ms")
    print(f"TPOT: {perf_metrics.get_tpot().mean:.2f} ± {perf_metrics.get_tpot().std:.2f} ms")
    print(f"Throughput : {perf_metrics.get_throughput().mean:.2f} ± {perf_metrics.get_throughput().std:.2f} tokens/s")
    rss_usage_gb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
    print(f"rss_usage:{rss_usage_gb:.3f} GB")

if '__main__' == __name__:
    main()
