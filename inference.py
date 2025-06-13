# inference.py
import clip
import torch
import numpy as np
from transformers import GPT2Tokenizer
import PIL.Image as Image
from model import ClipCaptionModel, generate2
import os

# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path = "./sample.jpg"  # 테스트할 이미지 경로
weights_path = "./checkpoints/foot_caption_prefix-009.pt"  # 학습된 모델 가중치
prefix_length = 32

# 모델 로딩
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

model = ClipCaptionModel(prefix_length)
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model = model.eval().to(device)

# 이미지 전처리 및 feature 추출
image = Image.open(image_path).convert("RGB")
image_tensor = preprocess(image).unsqueeze(0).to(device)

with torch.no_grad():
    prefix = clip_model.encode_image(image_tensor).to(device, dtype=torch.float32)
    prefix = prefix / prefix.norm(2, -1, keepdim=True)  # 정규화
    prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)

# 캡션 생성
generated_caption = generate2(model, tokenizer, embed=prefix_embed)
print(f"\nGenerated caption: {generated_caption}")
