#  Imports and Setup
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import os
import time
import psutil
import openvino.runtime as ov
from openvino.runtime import Core
import numpy as np
import matplotlib.pyplot as plt

#  Initialize model and tokenizer
model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

#  Load quantized model
output_dir = "C:\\Users\\HP\\Desktop\\Intel_LLM\\.venv\\Pipeline\\FINAL"
quantized_ir_path = os.path.join(output_dir, "tinyllama_quantized.xml")
core = Core() 
quantized_model = core.read_model(quantized_ir_path)
quantized_compiled_model = core.compile_model(quantized_model, "CPU")

#  Define utility functions
def get_file_size(file_path):
    return os.path.getsize(file_path) / (1024 ** 2)  # Size in MB

def benchmark_model(model, input_text, tokenizer, num_runs=10):
    inputs = tokenizer(input_text, return_tensors="pt")
    start_time = time.time()
    for _ in range(num_runs):
        outputs = model(**inputs)
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    return avg_time

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # Convert to MB

def benchmark_quantized_model(compiled_model, input_text, tokenizer, num_runs=10):
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].detach().cpu().numpy()
    
    # Get the expected input shape from the model
    input_shape = compiled_model.input(0).shape
    
    # Pad or truncate the input to match the expected shape
    if input_ids.shape[1] < input_shape[1]:
        pad_length = input_shape[1] - input_ids.shape[1]
        input_ids = np.pad(input_ids, ((0, 0), (0, pad_length)), mode='constant', constant_values=tokenizer.pad_token_id)
    elif input_ids.shape[1] > input_shape[1]:
        input_ids = input_ids[:, :input_shape[1]]

    start_time = time.time()
    for _ in range(num_runs):
        infer_request = compiled_model.create_infer_request()
        infer_request.infer(inputs={"input_ids": input_ids})
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    return avg_time

#  Calculate model sizes
original_model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
quantized_model_size = get_file_size(quantized_ir_path) + get_file_size(quantized_ir_path.replace('.xml', '.bin'))

print(f"Original model size: {original_model_size:.2f} MB")
print(f"Quantized model size: {quantized_model_size:.2f} MB")

# Cell 6: Benchmark models
input_text = "India is Known for"
original_model_time = benchmark_model(model, input_text, tokenizer)
original_model_memory = get_memory_usage()

# Get the expected input shape from the quantized model
input_shape = quantized_compiled_model.input(0).shape
print(f"Expected input shape for quantized model: {input_shape}")

quantized_model_time = benchmark_quantized_model(quantized_compiled_model, input_text, tokenizer)
quantized_model_memory = get_memory_usage()

#  Print comparison results
print(f"Original model - Average Inference Time: {original_model_time:.4f} seconds")
print(f"Quantized model - Average Inference Time: {quantized_model_time:.4f} seconds")
print(f"Original model - Peak Memory Usage: {original_model_memory:.2f} MB")
print(f"Quantized model - Peak Memory Usage: {quantized_model_memory:.2f} MB")
print(f"Original model size: {original_model_size:.2f} MB")
print(f"Quantized model size: {quantized_model_size:.2f} MB")


# Sample data
models = ['Original', 'Quantized']
inference_times = [original_model_time, quantized_model_time]
model_sizes = [original_model_size, quantized_model_size]

# Plot Inference Time
fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.bar(models, inference_times, color=['blue', 'orange'])
ax1.set_ylabel('Inference Time (seconds)')
ax1.set_title('Inference Time Comparison')
for i, v in enumerate(inference_times):
    ax1.text(i, v, f'{v:.4f}', ha='center', va='bottom')

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('inference_FINAL.png')
plt.close(fig)  # Close the figure to avoid overlap

# Plot Model Size
fig, ax2 = plt.subplots(figsize=(8, 6))
ax2.bar(models, model_sizes, color=['blue', 'orange'])
ax2.set_ylabel('Model Size (MB)')
ax2.set_title('Model Size Comparison')
for i, v in enumerate(model_sizes):
    ax2.text(i, v, f'{v:.2f}', ha='center', va='bottom')

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('model_size_FINAL.png')
plt.close(fig)  # Close the figure to avoid overlap

print("Graphs saved successfully.")
