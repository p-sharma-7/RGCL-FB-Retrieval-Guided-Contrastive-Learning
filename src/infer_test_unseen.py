import os
import torch
import pandas as pd
from tqdm import tqdm
from model.classifier import classifier_hateClipper
from data_loader.dataset import load_feats_from_CLIP
from data_loader.rac_dataloader import CLIP2Dataloader

# ------------------ Configuration ------------------
DATASET = "FB"
MODEL_NAME = "openai_clip-vit-large-patch14-336_HF"
CKPT_PATH = "/workspace/RGCL/logging/Retrieval/FB/RAC/RAC_lr0.0001_Bz8_Ep30_cosSim_triplet_drop[0.2, 0.4, 0.1]_topK20__PseudoGold_positive_1_hard_negative_1_seed0_hybrid_loss/ckpt/last_model_29_0.788.pt"
CLIP_EMB_PATH = "/workspace/RGCL/data/CLIP_Embedding"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Output CSV file
OUTPUT_CSV = "test_unseen_predictions.csv"

# ------------------ Load test_unseen embeddings ------------------
print("[INFO] Loading CLIP features for test_unseen...")
train, dev, test_seen, test_unseen = load_feats_from_CLIP(CLIP_EMB_PATH, DATASET, MODEL_NAME)
_, _, test_unseen_dl = CLIP2Dataloader(train, dev, test_unseen, batch_size=BATCH_SIZE)

# ------------------ Load model ------------------
print("[INFO] Loading model...")
image_dim = list(enumerate(test_unseen_dl))[0][1]["image_feats"].shape[1]
text_dim = list(enumerate(test_unseen_dl))[0][1]["text_feats"].shape[1]

class Args:
    def __init__(self):
        self.dataset = 'FB'
        self.map_dim = 1024
        self.last_layer = 'none'
        self.device = 'cuda'

args = Args()

for batch in test_unseen_dl:
    image_feat_dim = batch["image_feats"].shape[1]
    text_feat_dim = batch["text_feats"].shape[1]
    break  # Only need one batch to get the shape

num_layers = 3
proj_dim = 1024
map_dim = 1024
fusion_mode = "align"
dropout = [0.2, 0.4, 0.1]

model = classifier_hateClipper(
    image_feat_dim, text_feat_dim, num_layers, proj_dim, map_dim, fusion_mode,
    dropout=(0.1, 0.4, 0.2), args=args
)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ------------------ Inference ------------------
print("[INFO] Running inference on test_unseen...")

results = []

with torch.no_grad():
    for batch in tqdm(test_unseen_dl, desc="Inferencing"):
        image_feats = batch["image_feats"].to(DEVICE)
        text_feats = batch["text_feats"].to(DEVICE)
        ids = batch["ids"]
        
        logits = model(image_feats, text_feats)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        labels = (probs > 0.5).astype(int)

        for img_id, label, prob in zip(ids, labels, probs):
            results.append({
                "id": img_id,
                "predicted_label": int(label),
                "confidence": float(prob)
            })

# ------------------ Save Output ------------------
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"[INFO] Saved predictions to {OUTPUT_CSV}")
