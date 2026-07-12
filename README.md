# Gemini 3 Flash Student Distillation

This repository contains the architecture, training pipeline, and deployment artifacts for a 13.1M parameter student model distilled from Google's Gemini 3 Flash.

## Architecture
- **Type**: Transformer Decoder
- **Parameters**: 13.1 Million
- **Features**: RoPE (Rotary Positional Embeddings), SwiGLU Activation.

## Deployment Instructions

### 1. FastAPI Inference API
Run the production server using `uvicorn`:
```bash
pip install fastapi uvicorn torch
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 2. ONNX Runtime
For high-performance CPU/GPU inference without PyTorch:
```python
import onnxruntime as ort
session = ort.InferenceSession('student_model.onnx')
# Run inference using session.run()
```

### 3. Quantized Model (INT8)
The model has been optimized using dynamic quantization, reducing latency by ~1.5x on CPU. Use `student_model_int8.pth` for resource-constrained environments.

## Files
- `train.py`: Model definition and training logic.
- `app.py`: FastAPI implementation.
- `program.md`: Research objectives and constraints.
