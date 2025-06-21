# image_caption_api.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch
import io

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for your Android app (adjust origin as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with ["http://localhost:3000"] if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and supporting components
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.get("/")
def home():
    return {"message": "🖼️ Image Captioning API is running!"}

@app.post("/caption/")
async def caption_image(file: UploadFile = File(...)):
    try:
        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Feature extraction and inference
        pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)
        output_ids = model.generate(pixel_values, max_length=16, num_beams=4)

        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        return {"success": True, "caption": caption}
    except Exception as e:
        return {"success": False, "error": str(e)}
