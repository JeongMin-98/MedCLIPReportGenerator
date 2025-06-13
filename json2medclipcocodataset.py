import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
import re
import unicodedata
import pickle
from collections import defaultdict
from sklearn.model_selection import train_test_split

# 텍스트 정제 함수
def _clean_report(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # 비ASCII 문자 제거
    text = re.sub(r'([.!?]){2,}', r'\1', text)  # 중복된 마침표 제거
    text = re.sub(r'\[\s*finding\s*\]', '[FINDING]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\s*conclusion\s*\]', '[CONCLUSION]', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\s*diagnosis\s*\]', '[DIAGNOSIS]', text, flags=re.IGNORECASE)
    parts = re.split(r'\[\s*recommend(?:ation)?\s*\]', text, flags=re.IGNORECASE)
    text = parts[0]
    text = text.replace('_x000D_', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    cleaned = re.sub(r'\[\s*(FINDING|DIAGNOSIS|CONCLUSION)\s*\]', '', text, flags=re.IGNORECASE).strip()
    sentences = [s.strip() for s in re.split(r'\.\s*', cleaned) if s.strip()]
    if len(sentences) % 2 == 0 and all(sentences[i].lower() == sentences[i + len(sentences)//2].lower() for i in range(len(sentences)//2)):
        sentences = sentences[:len(sentences)//2]
    final_text = '. '.join(sentences) + '.'
    return final_text.strip()

# 모델 로딩
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
medclip_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
medclip_model.from_pretrained()
medclip_model.to(device)
medclip_processor = MedCLIPProcessor()
# JSON 데이터 로딩
with open("data/json/final_samples_both_only_v2.json", "r") as f:
    data = json.load(f)

medclip_embedding_dict = {}
captions = []

# 배치 처리 설정
BATCH_SIZE = 8
temp_batch = []
temp_items = []

for item in tqdm(data, desc="Embedding"):
    image_path = item.get("merged_image_path")
    if not image_path or not os.path.exists(image_path):
        continue

    try:
        image = Image.open(image_path).convert("RGB").resize((224,224))
        temp_batch.append(image)
        temp_items.append(item)

        if len(temp_batch) == BATCH_SIZE:
            # 배치 임베딩
            inputs = medclip_processor(images=temp_batch, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                features = medclip_model.encode_image(pixel_values=inputs["pixel_values"])
                features = features.cpu().numpy()

            for i, item in enumerate(temp_items):
                patient_id = item["patient_id"]
                diagnosis = _clean_report(item["diagnosis"])  # _clean_report로 텍스트 정제
                medclip_embedding_dict[patient_id] = features[i]
                captions.append({
                    "image_id": patient_id,
                    "caption": diagnosis,
                    "class": item.get("class", "")
                })

            temp_batch.clear()
            temp_items.clear()

    except Exception as e:
        print(f"[ERROR] {item.get('merged_image_path', 'Unknown path')}: {e}")
        continue

# 마지막 배치 처리
if temp_batch:
    inputs = medclip_processor(images=temp_batch, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = medclip_model.encode_image(pixel_values=inputs["pixel_values"]).cpu().numpy()


    for i, item in enumerate(temp_items):
        patient_id = item["patient_id"]
        diagnosis = _clean_report(item["diagnosis"])  # _clean_report로 텍스트 정제
        medclip_embedding_dict[patient_id] = features[i]
        captions.append({
            "image_id": patient_id,
            "caption": diagnosis,
            "class": item.get("class", "")
        })

# 클래스별 데이터 그룹화
class_to_items = defaultdict(list)
for cap in captions:
    class_to_items[cap["class"]].append(cap)

# 클래스별로 10% 데이터를 테스트 세트로 추출
train_captions, test_captions = [], []
train_embeddings, test_embeddings = {}, {}

for cls, items in class_to_items.items():
    # 각 클래스에서 10% 데이터를 테스트 세트로 분리
    train_items, test_items = train_test_split(items, test_size=0.1, random_state=42)
    train_captions.extend(train_items)
    test_captions.extend(test_items)

    # 임베딩도 동일하게 분리
    for item in train_items:
        train_embeddings[item["image_id"]] = medclip_embedding_dict[item["image_id"]]
    for item in test_items:
        test_embeddings[item["image_id"]] = medclip_embedding_dict[item["image_id"]]

# 학습용 데이터 저장
train_output = {
    "medclip_embedding": train_embeddings,
    "captions": train_captions
}
with open("medclipcap_dataset_train.pkl", "wb") as f:
    pickle.dump(train_output, f)

# 테스트용 데이터 저장
test_output = {
    "medclip_embedding": test_embeddings,
    "captions": test_captions
}
with open("medclipcap_dataset_test.pkl", "wb") as f:
    pickle.dump(test_output, f)

print("✅ 정제 및 저장 완료: train → medclipcap_dataset_train.pkl, test → medclipcap_dataset_test.pkl")
