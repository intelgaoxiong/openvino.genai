#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai
from openvino import get_version

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('models_path')
    parser.add_argument('adapter_path')
    parser.add_argument('prompt_file')
    parser.add_argument('device')
    args = parser.parse_args()

    device = args.device
    adapter = openvino_genai.Adapter(args.adapter_path)
    adapter_config = openvino_genai.AdapterConfig(adapter)
    config = {}
    if device == 'NPU':
        config["NPUW_DEVICES"] = "NPU"

    if args.prompt_file is not None:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompt = [f.read()]
    else:
        prompt = ['The Sky is blue because'] if args.prompt is None else [args.prompt]
    if len(prompt) == 0:
        raise RuntimeError(f'Prompt is empty!')

    print(f'openvino runtime version: {get_version()}')

    pipe = openvino_genai.LLMPipeline(args.models_path, device, config, adapters=adapter_config)  # register all required adapters here

    print("Generate with LoRA adapter and alpha set to 0.75:")
    res1 = pipe.generate(prompt, max_new_tokens=100, adapters=openvino_genai.AdapterConfig(adapter, 0.75))
    print(res1.texts[0])
    perf_metrics = res1.perf_metrics
    print(f"Output token size: {perf_metrics.get_num_generated_tokens()}")
    print(f"Load time: {perf_metrics.get_load_time():.2f} ms")
    print(f"Generate time: {perf_metrics.get_generate_duration().mean:.2f} ± {perf_metrics.get_generate_duration().std:.2f} ms")
    print(f"Tokenization time: {perf_metrics.get_tokenization_duration().mean:.2f} ± {perf_metrics.get_tokenization_duration().std:.2f} ms")
    print(f"Detokenization time: {perf_metrics.get_detokenization_duration().mean:.2f} ± {perf_metrics.get_detokenization_duration().std:.2f} ms")
    print(f"TTFT: {perf_metrics.get_ttft().mean:.2f} ± {perf_metrics.get_ttft().std:.2f} ms")
    print(f"TPOT: {perf_metrics.get_tpot().mean:.2f} ± {perf_metrics.get_tpot().std:.2f} ms")
    print(f"Throughput : {perf_metrics.get_throughput().mean:.2f} ± {perf_metrics.get_throughput().std:.2f} tokens/s")

    print("\n-----------------------------")
    print("Generate without LoRA adapter:")
    res2 = pipe.generate(prompt, max_new_tokens=100, adapters=openvino_genai.AdapterConfig())
    print(res2.texts[0])
    perf_metrics = res2.perf_metrics
    print(f"Output token size: {perf_metrics.get_num_generated_tokens()}")
    print(f"Load time: {perf_metrics.get_load_time():.2f} ms")
    print(f"Generate time: {perf_metrics.get_generate_duration().mean:.2f} ± {perf_metrics.get_generate_duration().std:.2f} ms")
    print(f"Tokenization time: {perf_metrics.get_tokenization_duration().mean:.2f} ± {perf_metrics.get_tokenization_duration().std:.2f} ms")
    print(f"Detokenization time: {perf_metrics.get_detokenization_duration().mean:.2f} ± {perf_metrics.get_detokenization_duration().std:.2f} ms")
    print(f"TTFT: {perf_metrics.get_ttft().mean:.2f} ± {perf_metrics.get_ttft().std:.2f} ms")
    print(f"TPOT: {perf_metrics.get_tpot().mean:.2f} ± {perf_metrics.get_tpot().std:.2f} ms")
    print(f"Throughput : {perf_metrics.get_throughput().mean:.2f} ± {perf_metrics.get_throughput().std:.2f} tokens/s")

if '__main__' == __name__:
    main()
