import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import os
import openvino.runtime as ov
from nncf import quantize
from nncf.quantization import QuantizationPreset
from torch.utils.data import Dataset, DataLoader

# Initialize model and tokenizer
model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# Create dummy input
dummy_input = tokenizer("Hello, how are you?", return_tensors="pt")
print("Input shape:", dummy_input['input_ids'].shape)

# Specify output directory
output_dir = "FINAL"
os.makedirs(output_dir, exist_ok=True)

# Specify the output file path for ONNX
onnx_path = os.path.join(output_dir, "tinyllama.onnx")

# Export the model to ONNX format
torch.onnx.export(
    model,
    (dummy_input['input_ids'],),
    onnx_path,
    opset_version=14,
    input_names=["input_ids"],
    output_names=["output"]
)

# Load the ONNX model and convert to OpenVINO IR
core = ov.Core()
onnx_model = core.read_model(onnx_path)

# Specify the output paths for the OpenVINO IR format files
ir_xml_path = os.path.join(output_dir, "tinyllama_ir.xml")
ir_bin_path = os.path.join(output_dir, "tinyllama_ir.bin")

# Convert the ONNX model to OpenVINO IR format
ov.serialize(onnx_model, ir_xml_path, ir_bin_path)

print("Model has been successfully converted to OpenVINO IR format and saved to", output_dir)

# Create a dummy calibration dataset
class CalibrationDataset(Dataset):
    def __init__(self, num_samples, batch_size=1):
        self.num_samples = num_samples
        self.batch_size = batch_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        # Generate a tensor with the correct shape [1, 7]
        input_ids = torch.randint(0, 100, (1, 7), dtype=torch.int32)  # Ensure the shape is [1, 7]
        return {"input_ids": input_ids}

    def get_batch_size(self):
        return self.batch_size

    def get_length(self):
        return self.__len__()

    def get_inference_data(self):
        dataloader = DataLoader(self, batch_size=self.batch_size)
        for batch in dataloader:
            input_ids = batch["input_ids"].squeeze(0)
            yield {"input_ids": input_ids}

calibration_dataset = CalibrationDataset(num_samples=256, batch_size=1)

nncf_config = {
    "input_info": {
        "sample_size": [1, 7]  # Adjust according to your model's input shape
    },
    "compression": {
        "algorithm": "quantization",
        "preset": "performance"  # Using the performance preset for a balance between speed and accuracy
    },
    "quantizer": {
        "precision_bits": {
            "weights": 8,          # Quantize weights to 8 bits
            "activations": 8       # Quantize activations to 8 bits
        },
        "mode": "asymmetric"        # Use asymmetric quantization mode
    }
}

# Load the OpenVINO model for quantization
ov_model = core.read_model(ir_xml_path, ir_bin_path)

# Quantize the OpenVINO model
quantized_model = quantize(
    ov_model,
    calibration_dataset,
    preset=QuantizationPreset.PERFORMANCE
)

# Specify the output paths for the quantized model files
quantized_model_xml_path = os.path.join(output_dir, "tinyllama_quantized.xml")
quantized_model_bin_path = os.path.join(output_dir, "tinyllama_quantized.bin")

# Save the quantized model
ov.serialize(quantized_model, quantized_model_xml_path, quantized_model_bin_path)

print("Quantized OpenVINO model saved as tinyllama_quantized.xml and tinyllama_quantized.bin in the directory:", output_dir)

# Save the model and tokenizer configuration files
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
