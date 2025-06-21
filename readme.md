# 🖼️ Image Captioning API

A FastAPI-based image captioning server using the ViT-GPT2 transformer model.

## 🚀 Features
- Upload an image
- Get a caption generated using `nlpconnect/vit-gpt2-image-captioning`

## 🧪 Local Development

```bash
pip install -r requirements.txt
uvicorn image_caption_api:app --reload
