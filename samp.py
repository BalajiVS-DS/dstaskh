import os
import pickle
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import argparse

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Database file
DB_FILE = "class_db.pkl"


def get_image_embedding(image_path):
    #"""Generate image embedding using CLIP."""
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
        embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
    return embedding.cpu().squeeze(0)


def load_db():
    #"""Load stored class embeddings."""
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_db(db):
    #"""Save class embeddings."""
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)


def add_class(class_name, folder_path):
    #"""Add new document type with sample images."""
    db = load_db()
    embeddings = []

    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, file)
            emb = get_image_embedding(img_path)
            embeddings.append(emb)

    if not embeddings:
        print("No valid images found.")
        return

    avg_embedding = torch.stack(embeddings).mean(dim=0)
    db[class_name] = avg_embedding
    save_db(db)

    print(f"✅ Added document type: '{class_name}' with {len(embeddings)} images.")


def classify_image(image_path):
    #"""Classify document image."""
    db = load_db()
    if not db:
        print(" No classes available. Add document types first.")
        return

    image_emb = get_image_embedding(image_path)

    best_class, best_score = None, -1
    for label, class_emb in db.items():
        score = torch.nn.functional.cosine_similarity(image_emb.unsqueeze(0), class_emb.unsqueeze(0)).item()
        if score > best_score:
            best_score = score
            best_class = label

    print(f" Document classified as: **{best_class}** (score: {best_score:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--add", nargs=2, metavar=("class_name", "folder_path"), help="Add a new document class")
    parser.add_argument("--predict", metavar="image_path", help="Classify an image")

    args = parser.parse_args()

    if args.add:
        class_name, folder = args.add
        add_class(class_name, folder)
    elif args.predict:
        classify_image(args.predict)
    else:
        print("Usage:")
        print("  Add class:     python doc_classifier.py --add aadhar images/aadhar")
        print("  Classify:      python doc_classifier.py --predict test_image.jpg")

