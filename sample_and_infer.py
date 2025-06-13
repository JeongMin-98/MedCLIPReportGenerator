import pickle
import random
from collections import defaultdict
from tqdm import tqdm
import torch
from transformers import GPT2Tokenizer
from model import ClipCaptionPrefix, generate2, MappingType  # 모델 정의와 generate 함수

# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = "checkpoints/foot_caption-009.pt"
prefix_length = 40
clip_length = 40  # 학습 시 설정했던 clip_length
sample_ratio = 0.1  # 10% 샘플링

# tokenizer & 모델 로드
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = ClipCaptionPrefix(
    prefix_length=prefix_length,
    clip_length=clip_length,
    mapping_type=MappingType.Transformer  # 중요: Transformer 기반 모델
)
model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
model = model.eval().to(device)

# clipcap_dataset.pkl 로드
with open("clipcap_dataset_test.pkl", "rb") as f:
    dataset = pickle.load(f)

clip_embeddings = dataset["clip_embedding"]
captions = dataset["captions"]

# 클래스별로 균등 샘플링
class_to_items = defaultdict(list)
for cap in captions:
    class_to_items[cap["class"]].append(cap)

sampled_items = []
for cls, items in class_to_items.items():
    sample_size = max(1, int(len(items) * sample_ratio))
    sampled_items.extend(random.sample(items, sample_size))

# Inference 시작
print("🔍 Inference 진행 중...")
for item in tqdm(sampled_items, desc="Inference"):
    image_id = item["image_id"]
    true_caption = item["caption"]
    clip_embedding = torch.tensor(clip_embeddings[image_id], dtype=torch.float32).unsqueeze(0).to(device)

    # cosine normalize (normalize_prefix 옵션 사용 시 필수)
    clip_embedding = clip_embedding / clip_embedding.norm(2, dim=-1, keepdim=True)

    # TransformerMapper는 embed가 아닌 prefix 입력으로 처리
    generated_caption = generate2(model, tokenizer, embed=model.clip_project(clip_embedding))

    print(f"🖼️ Image ID: {image_id}")
    print(f"   Original Caption: {true_caption}")
    print(f"   Generated Caption: {generated_caption}")
    print("-" * 50)

print("✅ Inference 완료!")
