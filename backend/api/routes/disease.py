import io
import torch
import os
from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from pydantic import BaseModel
from PIL import Image
from loguru import logger
from openai import OpenAI
from backend.trms_model import PLANT_VILLAGE_CLASSES, get_transform

router = APIRouter()

@router.post("/disease-predict")
async def predict_disease(request: Request, file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Access model from app state
    model = getattr(request.app.state, "disease_model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Disease model not loaded")

    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        transform = get_transform()
        input_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)

        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            
        predicted_class = PLANT_VILLAGE_CLASSES[predicted_idx.item()]
        
        # Simple advice mapping
        advice = "Please consult an agricultural expert for specific treatment recommendations."
        if "healthy" in predicted_class:
            advice = "Your plant looks healthy! Continue regular maintenance and monitoring."
        elif "Blight" in predicted_class or "blight" in predicted_class:
            advice = "Apply appropriate fungicide (e.g., Copper-based or Mancozeb). Remove infected parts and improve air circulation."
        elif "Rust" in predicted_class or "rust" in predicted_class:
            advice = "Use sulfur-based fungicides or appropriate rust-control sprays. Ensure proper crop rotation."
        elif "Aphids" in predicted_class or "Aphid" in predicted_class or "Mite" in predicted_class:
            advice = "Use neem oil or appropriate insecticidal soaps. Check for predators like ladybugs."
        elif "Blast" in predicted_class:
            advice = "Avoid excessive nitrogen fertilizers. Apply Tricyclazole or similar blast-control fungicides."
        elif "Smut" in predicted_class:
            advice = "Treat seeds with fungicides before planting. Remove and destroy infected heads immediately."
        elif "Bacterial" in predicted_class:
            advice = "Use copper-based bactericides. Avoid overhead irrigation and work in fields only when plants are dry."
        elif "virus" in predicted_class or "Virus" in predicted_class or "Tungro" in predicted_class:
            advice = "Control vector insects (like whiteflies or leafhoppers). Remove and destroy infected plants immediately."


        return {
            "class": predicted_class,
            "confidence": round(confidence.item() * 100, 2),
            "advice": advice
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

class AdvisoryRequest(BaseModel):
    disease_name: str

@router.post("/generate-advisory")
def generate_advisory(request: AdvisoryRequest):
    api_key = os.getenv("NVIDIA_API_KEY")
    base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    model_name = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")

    logger.info(f"Generating advisory for: {request.disease_name} using {model_name}")

    if not api_key:
        logger.error("NVIDIA API key not configured")
        raise HTTPException(status_code=500, detail="NVIDIA API key not configured")

    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        
        prompt = f"""
    You are a professional agricultural plant pathology expert.
    
    The detected disease is: {request.disease_name}
    
    I want you to provide a complete, detailed advisory in HINDI language.
    
    At the very top, give the Disease Name in both English and Hindi like this:
    Disease: [English Name] (हिंदी नाम)
    
    Then provide the following sections in Hindi:
    1. रोग का कारण (Cause of Disease)
    2. यह कैसे फैलता है (How It Spreads)
    3. रासायनिक उपचार (Chemical Treatment)
    4. जैविक उपचार (Organic Treatment)
    5. बचाव के उपाय (Preventive Measures)
    6. किसानों के लिए व्यावहारिक सलाह (Farmer Practical Advice)

    IMPORTANT RULES:
    - Language: Pure Hindi (farmer-friendly).
    - DO NOT use markdown characters like "###", "##", or "**" in your response.
    - Use clear, plain text headers for each section.
    - Keep it practical and actionable for an Indian farmer.
    """

        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
        )
        
        advisory_content = completion.choices[0].message.content
        logger.success("Advisory generated successfully")
        return {"advisory": advisory_content}
        
    except Exception as e:
        logger.error(f"LLM generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

