// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vision_encoder.hpp"
#include "utils.hpp"


#include "visual_language/qwen2vl/classes.hpp"
#include "visual_language/phi3_vision/classes.hpp"
#include "visual_language/minicpm/classes.hpp"
#include "visual_language/llava/classes.hpp"
#include "visual_language/llava_next/classes.hpp"
#include "visual_language/internvl_chat/classes.hpp"

#include <fstream>

namespace ov::genai {

void convertToStaticShape(std::shared_ptr<ov::Model>& model, size_t patch_size) {
    int batch_size = 1;
    int patch_len = 1376; // should a align to 16 for better infer performance on NPU

    std::map<ov::Output<ov::Node>, ov::PartialShape> shapes;

    for (const auto& input : model->inputs()) {
        auto input_shape = input.get_partial_shape();
        std::string input_name = input.get_any_name();

        std::cout << "input_name: " << input_name << std::endl;

        if (input_name.find("pixel_values") == 0) {
            input_shape[0] = batch_size;
            input_shape[1] = 3;
            input_shape[2] = patch_size;
            input_shape[3] = patch_size * patch_len;
        } else if (input_name.find("patch_attention_mask") == 0) {
            input_shape[0] = batch_size;
            input_shape[1] = 1;
            input_shape[2] = patch_len;
        } else if (input_name.find("position_ids") == 0) {
            input_shape[0] = batch_size;
            input_shape[1] = patch_len;
        }

        shapes[input] = input_shape;
    }

    // Reshape the model
    model->reshape(shapes);
}

VisionEncoder::VisionEncoder(const std::filesystem::path& model_dir, const std::string& device, const ov::AnyMap properties) {
    std::cout << "Create VisionEncoder on device " << device << " with model " << model_dir / "openvino_vision_embeddings_model.xml" << std::endl;
    m_processor_config = utils::from_config_json_if_exists<ProcessorConfig>(model_dir, "preprocessor_config.json");
    auto model = utils::singleton_core().read_model(model_dir / "openvino_vision_embeddings_model.xml");
    convertToStaticShape(model, m_processor_config.patch_size);
    std::filesystem::path blob_path("./vit.blob");
    ov::CompiledModel compiled_model;
    if (std::filesystem::exists(blob_path)) {
        std::ifstream fin(blob_path, std::ios::in | std::ios::binary);
        if (!fin.is_open()) {
            OPENVINO_THROW("Blob file can't be opened");
        }
        std::cout << "Compile VisionEncoder importing compiled model" << std::endl;
        compiled_model = ov::genai::utils::singleton_core().import_model(fin, "NPU", properties);
    } else {
        compiled_model = utils::singleton_core().compile_model(model, device, properties);
        std::ofstream fout(blob_path, std::ios::out | std::ios::binary);
        if (!fout.is_open()) {
            OPENVINO_THROW("Blob file can't be exported");
        }
        compiled_model.export_model(fout);
    }
    std::cout << "Compile VisionEncoder model done!" << std::endl;

    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM vision embeddings model");
    m_ireq_queue_vision_encoder = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
}

VisionEncoder::VisionEncoder(
    const std::string& model,
    const ov::Tensor& weights,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) {
    auto compiled_model = utils::singleton_core().compile_model(model, weights, device, device_config);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM vision embeddings model");
    m_ireq_queue_vision_encoder = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
    m_processor_config = utils::from_config_json_if_exists<ProcessorConfig>(config_dir_path, "preprocessor_config.json");
}

ProcessorConfig VisionEncoder::get_processor_config() const {
    return m_processor_config;
}

VisionEncoder::Ptr VisionEncoder::create(const std::filesystem::path& model_dir, const VLMModelType model_type, const std::string& device, const ov::AnyMap properties) {
    if (model_type == VLMModelType::MINICPM) {
        return std::make_shared<VisionEncoderMiniCPM>(model_dir, device, properties);
    } else if (model_type == VLMModelType::LLAVA) {
        return std::make_shared<VisionEncoderLLaVA>(model_dir, device, properties);
    } else if (model_type == VLMModelType::LLAVA_NEXT) {
        return std::make_shared<VisionEncoderLLaVANext>(model_dir, device, properties);
    } else if (model_type == VLMModelType::INTERNVL_CHAT) {
        return std::make_shared<VisionEncoderInternVLChat>(model_dir, device, properties);
    } else if (model_type == VLMModelType::PHI3_V) {
        return std::make_shared<VisionEncoderPhi3V>(model_dir, device, properties);
    } else if (model_type == VLMModelType::QWEN2_VL) {
        return std::make_shared<VisionEncoderQwen2VL>(model_dir, device, properties);
    } else {
        OPENVINO_THROW("Unsupported model type in VLM VisionEncoder class. Please, create feature request on new model support");
    }
}

VisionEncoder::Ptr VisionEncoder::create(
    const std::string& model,
    const ov::Tensor& weights,
    const std::filesystem::path& config_dir_path,
    const VLMModelType model_type,
    const std::string& device,
    const ov::AnyMap device_config) {
    if (model_type == VLMModelType::MINICPM) {
        return std::make_shared<VisionEncoderMiniCPM>(model, weights, config_dir_path, device, device_config);
    } else if (model_type == VLMModelType::LLAVA) {
        return std::make_shared<VisionEncoderLLaVA>(model, weights, config_dir_path, device, device_config);
    } else if (model_type == VLMModelType::LLAVA_NEXT) {
        return std::make_shared<VisionEncoderLLaVANext>(model, weights, config_dir_path, device, device_config);
    } else if (model_type == VLMModelType::INTERNVL_CHAT) {
        return std::make_shared<VisionEncoderInternVLChat>(model, weights, config_dir_path, device, device_config);
    } else if (model_type == VLMModelType::PHI3_V) {
        return std::make_shared<VisionEncoderPhi3V>(model, weights, config_dir_path, device, device_config);
    } else if (model_type == VLMModelType::QWEN2_VL) {
        return std::make_shared<VisionEncoderQwen2VL>(model, weights, config_dir_path, device, device_config);
    } else {
        OPENVINO_THROW("Unsupported model type in VLM VisionEncoder class. Please, create feature request on new model support");
    }
}

} // namespace ov::genai
