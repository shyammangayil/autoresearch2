from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from train import StudentModel

app = FastAPI(title="Student Model Inference API")

# Load model globally for persistence
device = 'cpu'
model = StudentModel()
model.load_state_dict(torch.load('student_model.pth', map_location=device))
model.eval()

class InferenceRequest(BaseModel):
    token_ids: list[int]
    top_k: int = 5

@app.post("/predict")
async def predict(request: InferenceRequest):
    try:
        input_tensor = torch.tensor([request.token_ids])
        with torch.no_grad():
            logits, _ = model(input_tensor)
            # Get last token probabilities
            probs = F.softmax(logits[:, -1, :], dim=-1)
            top_probs, top_indices = torch.topk(probs, request.top_k)
            
        return {
            "top_predictions": [
                {"token_id": int(idx), "probability": float(p)} 
                for p, idx in zip(top_probs[0], top_indices[0])
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

print("FastAPI script written to app.py")
