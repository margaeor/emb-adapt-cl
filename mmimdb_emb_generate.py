import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor
from PIL import Image
import pickle
import pandas as pd
import argparse
import os
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, dataframe, mode, transform=None, tokenizer=None, feature_extractor=None):
        self.dataframe = dataframe
        self.mode = mode
        self.transform = transform
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        if self.mode == "image":
            #image = Image.open(row["image"]).convert("RGB")
            image = row['image']
            if self.transform:
                image = self.transform(image)
            return row["id"], row["set"], image, row["genre_index"]
        elif self.mode == "text":
            text = row["text"]
            if self.tokenizer:
                tokens = self.tokenizer(
                    text, truncation=True, padding="max_length", max_length=128, return_tensors="pt"
                )
                return row["id"], row["set"], tokens, row["genre_index"]


def generate_image_embeddings(dataframe, batch_size, output_file, device):
    feature_extractor = AutoFeatureExtractor.from_pretrained(f"google/vit-base-patch16-224-in21k")
    model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")
    
    model.to(device)
    model.eval()

    dataset = CustomDataset(dataframe, mode="image", transform=feature_extractor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    results = []
    with torch.no_grad():
        for ids, sets, images, genres in tqdm(dataloader, desc="Processing images"):
            images = torch.stack(images['pixel_values'])[0].to(device)
            outputs = model(images)
            embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()  # CLS token embedding
            for id_, set_, embedding, genre in zip(ids, sets, embeddings, genres):
                results.append({"id": id_, "set": set_, "embedding": embedding, "genre_index": genre.item()})

    pd.DataFrame(results).to_pickle(output_file)


def generate_text_embeddings(dataframe, batch_size, output_file, device):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.to(device)
    model.eval()

    dataset = CustomDataset(dataframe, mode="text", tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    results = []
    with torch.no_grad():
        for ids, sets, tokens, genres in tqdm(dataloader, desc="Processing text"):
            input_ids = tokens["input_ids"].squeeze(1).to(device)
            attention_mask = tokens["attention_mask"].squeeze(1).to(device)
            embeddings = model(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :].cpu().numpy()
            for id_, set_, embedding, genre in zip(ids, sets, embeddings, genres):
                results.append({"id": id_, "set": set_, "embedding": embedding, "genre_index": genre.item()})

    print(f"Saving to {output_file}")
    pd.DataFrame(results).to_pickle(output_file)


input = './mmimdb/df_drama_comedy_preprocessed.pickle'

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate embeddings from a dataframe")
    parser.add_argument("--mode", type=str, default='text', choices=["image", "text"],  help="Mode: image or text")
    parser.add_argument("--input_file", type=str, default=input, help="Path to input dataframe in pickle format")
    parser.add_argument("--output_dir", type=str, default='./mmimdb/', help="Path to save the output dataframe in pickle")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: cuda or cpu")
    args = parser.parse_args()

    df = pd.read_pickle(args.input_file)#.head(100)

    
    if args.mode == "image":
        output_fname = f'2ViT_emb_mmimdb.pickle'
        generate_image_embeddings(df, args.batch_size, os.path.join('mmimdb/', output_fname), args.device)
    elif args.mode == "text":
        output_fname = f'bert_emb_mmimdb.pickle'
        generate_text_embeddings(df, args.batch_size, os.path.join('mmimdb/', output_fname), args.device)