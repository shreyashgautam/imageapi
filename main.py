import os
import io
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# ✅ Set huggingface model cache location
os.environ["HF_HOME"] = "/tmp/huggingface"  # recommended over TRANSFORMERS_CACHE

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ CORS config (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with ["http://your-app.com"] in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load model and supporting components
try:
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

# ✅ Health check route
@app.get("/")
def home():
    return {"message": "🖼️ Image Captioning API is running!"}

# ✅ Captioning endpoint
@app.post("/caption/")
async def caption_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess and generate caption
        pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)
        output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        return {"success": True, "caption": caption}

    except Exception as e:
        return {"success": False, "error": str(e)}
