import os
import pickle
import yaml
import math
import json
import clip
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from feature_utils import set_seeds
from video_decoder import VideoDecoder
from sklearn.neighbors import KernelDensity

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

def shot_boundary_detection(video_path, model, config):
    output_dir = config['video_feature_dir']
    video_name = os.path.basename(video_path).replace('.mp4', '')
    os.makedirs(f"{output_dir}/{video_name}", exist_ok=True)
    pkl_file = f"{output_dir}/{video_name}/shot_boundary_detection.pkl"

    if os.path.exists(pkl_file):
        print(f"\t[SBD] Found pkl: {pkl_file}, skipping", flush=True)
        return
    
    video_decoder = VideoDecoder(video_path, fps=config["fps"], max_dimension=[27,48])
    frames = np.stack([frame.get('frame') for frame in video_decoder], axis=0)
    video = frames.reshape([-1, 27, 48, 3])
    single_frame_predictions, _ = model.predict_frames(video)
    shot_list = model.predictions_to_scenes(single_frame_predictions)
    output_data = [(x[0].item() / video_decoder.fps(), x[1].item() / video_decoder.fps()) for x in shot_list]

    with open(pkl_file, 'wb') as pkl:
        pickle.dump(output_data, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'\t[SBD] Result saved: {pkl.name}', flush=True)

def shot_density(video_path, config):
    output_dir = config['video_feature_dir']
    video_name = os.path.basename(video_path).replace('.mp4', '')
    pkl_file = f"{output_dir}/{video_name}/shot_density.pkl"
    sbd_pkl = f"{output_dir}/{video_name}/shot_boundary_detection.pkl"

    if os.path.exists(pkl_file):
        print(f"\t[Shot Density] Found pkl: {pkl_file}, skipping", flush=True)
        return

    if not os.path.exists(sbd_pkl):
        print("\t[Shot Density] No SBD results found, skipping", flush=True)
        return

    with open(sbd_pkl, 'rb') as pkl:
        shots_data = pickle.load(pkl)

    shots = [shot[0] for shot in shots_data]
    last_shot_end = max(shot[1] for shot in shots_data)

    time = np.linspace(0, last_shot_end, math.ceil(last_shot_end * config["fps"]) + 1)[:, np.newaxis]
    shots = np.asarray(shots).reshape(-1, 1)
    kde = KernelDensity(kernel="gaussian", bandwidth=10.0).fit(shots)
    log_dens = kde.score_samples(time)
    shot_density = np.exp(log_dens)
    shot_density = (shot_density - shot_density.min()) / (shot_density.max() - shot_density.min())

    output_data = {
        "y": shot_density.squeeze(),
        "time": time.squeeze().astype(np.float64),
        "delta_time": 1 / config["fps"]
    }
    
    with open(pkl_file, 'wb') as pkl:
        pickle.dump(output_data, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\t[Shot Density] Result saved: {pkl.name}", flush=True)

def process_query_batch(batch, text_features, model, preprocessor, device, result_matrices, queries):
    inputs = torch.stack([preprocessor(img) for img in batch]).to(device)
    
    with torch.no_grad(), torch.autocast(device):
        image_features = model.encode_image(inputs)
    
    for domain, domain_queries in queries.items():
        batch_results = []
        for _ in range(len(batch)):
            max_queries = []
            for label, texts in domain_queries.items():
                text_embeds = text_features[domain][label]
                logits_per_image = torch.matmul(image_features, text_embeds.t())
                max_query = max(zip(texts, [label] * len(texts), logits_per_image[_].tolist()), key=lambda item: item[2])
                max_queries.append(max_query)
            
            similarity_scores = torch.tensor([x[2] for x in max_queries], dtype=torch.float32)
            probs = similarity_scores.softmax(dim=-1).cpu().numpy()
            batch_results.append(list(zip(domain_queries.keys(), probs.tolist())))
        
        result_matrices[domain].extend(batch_results)

def predict_CLIP_queries(video_path, config, queries, model, preprocessor, device, batch_size=128):
    output_dir = config['video_feature_dir']
    video_name = os.path.basename(video_path).replace('.mp4', '')
    os.makedirs(f"{output_dir}/{video_name}", exist_ok=True)
    pkl_file = f"{output_dir}/{video_name}/clip_qas.pkl"

    if os.path.exists(pkl_file):
        print(f"\t[CLIP] Found pkl: {pkl_file}, skipping", flush=True)
        return

    video_decoder = VideoDecoder(video_path, fps=config["fps"])
    result_matrices = {domain: [] for domain in queries.keys()}

    text_features = {domain: {label: model.encode_text(clip.tokenize(texts).to(device)) 
                              for label, texts in domain_queries.items()}
                     for domain, domain_queries in queries.items()}
    
    batch = []
    for frame in tqdm(video_decoder, desc="Processing frames"):
        batch.append(Image.fromarray(np.uint8(frame.get('frame'))))
        
        if len(batch) == batch_size:
            process_query_batch(batch, text_features, model, preprocessor, device, result_matrices, queries)
            batch = []

    if batch:
        process_query_batch(batch, text_features, model, preprocessor, device, result_matrices, queries)

    with open(pkl_file, 'wb') as pkl:
        pickle.dump(result_matrices, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\t[CLIP] Result saved: {pkl.name}", flush=True)

def process_embedding_batch(batch, model, preprocessor, device):
    inputs = torch.stack([preprocessor(img) for img in batch]).to(device)
    with torch.no_grad(), torch.autocast(device):
        return model.encode_image(inputs).cpu().numpy()

def clip_image_embeddings(video_path, config, model, preprocessor, device, batch_size=128):
    output_dir = config['video_feature_dir']
    video_name = os.path.basename(video_path).replace('.mp4', '')
    os.makedirs(f"{output_dir}/{video_name}", exist_ok=True)
    pkl_file = f"{output_dir}/{video_name}/image_embedding.pkl"

    if os.path.exists(pkl_file):
        print(f"\t[Image Embeddings] Found pkl: {pkl_file}, skipping", flush=True)
        return

    video_decoder = VideoDecoder(video_path, fps=config["fps"])
    
    embeddings = []
    batch = []
    for frame in tqdm(video_decoder, desc="Processing frames"):
        batch.append(Image.fromarray(np.uint8(frame.get('frame'))))

        if len(batch) == batch_size:
            embeddings.extend(process_embedding_batch(batch, model, preprocessor, device))
            batch = []

    if batch:
        embeddings.extend(process_embedding_batch(batch, model, preprocessor, device))

    with open(pkl_file, 'wb') as pkl:
        pickle.dump(embeddings, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\t[Image Embeddings] Result saved: {pkl.name}", flush=True)

def main():
    parser = argparse.ArgumentParser(description='Extract visual features from videos')
    parser.add_argument('--feature', type=str, default='shot-bd', choices=['shot-bd', 'clip'], help='Feature to extract')
    args = parser.parse_args()

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    set_seeds(config['seed'])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.feature == 'shot-bd':
        from transnetv2 import TransNetV2
        model = TransNetV2("TransNetV2/inference/transnetv2-weights/")
    elif args.feature == 'clip':
        model, preprocessor = clip.load('ViT-B/32', device=device)
        with open(config['CLIP_queries']) as f:
            queries = json.load(f)

    os.makedirs(config['tmp_dir'], exist_ok=True)

    for v, video in enumerate(os.listdir(config['videos_dir']), 1):
        print(f"[{v}/{len(os.listdir(config['videos_dir']))}] Processing video: {video}", flush=True)
        video_path = f"{config['videos_dir']}/{video}"

        if args.feature == "shot-bd":
            shot_boundary_detection(video_path, model, config)
            shot_density(video_path, config)
        elif args.feature == "clip":
            predict_CLIP_queries(video_path, config, queries, model, preprocessor, device)
            clip_image_embeddings(video_path, config, model, preprocessor, device)

        print()

if __name__ == "__main__":
    main()