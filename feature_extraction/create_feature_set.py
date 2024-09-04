import pickle
import yaml
import os
import argparse
import pandas as pd
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_pickle(path):
    with open(path, 'rb') as pkl:
        return pickle.load(pkl)

def process_clip_data(clip_data, segment_start, segment_end, config):
    clip_features = []
    for domain in clip_data.keys():
        relevant_frames = [
            data for i, data in enumerate(clip_data[domain])
            if segment_start <= i / config['fps'] <= segment_end
        ]
        
        label_sums = defaultdict(float)
        for frame in relevant_frames:
            for label, prob in frame:
                label_sums[label] += prob
        
        feature_vec = [label_sums[label] / len(relevant_frames) for label in label_sums]
        clip_features.append(feature_vec)
    
    return clip_features

def process_density_data(sd_data, segment_start, segment_end):
    density_with_time = list(zip(sd_data['time'], sd_data['y']))
    relevant_frames = [
        data[1] for data in density_with_time
        if segment_start <= data[0] <= segment_end
    ]
    
    return np.mean(relevant_frames) if relevant_frames else 0

def process_shots_data(sbd_data, segment_start, segment_end):
    return sum(
        1 for shot_start, shot_end in sbd_data
        if max(0, min(segment_end, shot_end) - max(segment_start, shot_start)) / (segment_end - segment_start) > 0
    )

def process_sentiment_data(sentiment_data, segment_start, segment_end):
    sentiments_sum = defaultdict(float)
    count = 0
    for sentence in sentiment_data:
        ovp = max(0, min(segment_end, sentence['end']) - max(segment_start, sentence['start'])) / (sentence['end'] - sentence['start'])
        if ovp >= 0.8:
            count += 1
            for sentiment, prob in sentence['sentiment_probs']:
                sentiments_sum[sentiment] += prob
    
    return [sentiments_sum[sentiment] / count if count else 0 for sentiment in ['positive', 'negative', 'neutral']]

def process_pos_data(pos_data, segment_start, segment_end):
    for pos_seg in pos_data:
        ovp = max(0, min(segment_end, pos_seg['end_time']) - max(segment_start, pos_seg['start_time'])) / (pos_seg['end_time'] - pos_seg['start_time'])
        if ovp > 0.7:
            pos_vector = np.array(pos_seg['vector'])
            return (pos_vector / np.sum(pos_vector)).tolist() if np.sum(pos_vector) else pos_vector.tolist()
    
    return np.zeros(len(pos_data[0]['vector'])).tolist()

def process_ner_data(ner_data, segment_start, segment_end):
    for ner_seg in ner_data:
        ovp = max(0, min(segment_end, ner_seg['end_time']) - max(segment_start, ner_seg['start_time'])) / (ner_seg['end_time'] - ner_seg['start_time'])
        if ovp > 0.7:
            ner_vector = np.array(ner_seg['vector'])
            ner_vector = np.delete(ner_vector, 4)  # remove events
            return (ner_vector > 0).astype(int)
    
    return np.zeros(len(ner_data[0]['vector']) - 1).tolist()

def image_context_similarity(imgemb_data, segment_start, segment_end, sid, segments, config):
    fps = config['fps']
    context_size = config['context_size']

    segment_frames = imgemb_data[round(segment_start * fps):round(segment_end * fps)]
    
    def get_context_embeddings(offset):
        context_embeddings = []
        for i in range(1, context_size + 1):
            idx = sid + i * offset
            if 0 <= idx < len(segments):
                start, end = segments[idx][1]['start'], segments[idx][1]['end']
                context_embeddings.append(imgemb_data[round(start * fps):round(end * fps)])
            else:
                context_embeddings.append([])
        return context_embeddings

    before_embeddings = get_context_embeddings(-1)
    after_embeddings = get_context_embeddings(1)

    def compute_similarities(context_embeddings):
        return [
            np.quantile(cosine_similarity(context, segment_frames).flatten(), 0.8)
            if len(context) > 0 else 0
            for context in context_embeddings
        ]

    return compute_similarities(before_embeddings) + compute_similarities(after_embeddings)

def get_matching_textsegment(textemb_data, segment_start, segment_end):
    for text_segment in textemb_data:
        text_start, text_end = text_segment['start_time'], text_segment['end_time']
        ovp = max(0, min(segment_end, text_end) - max(segment_start, text_start)) / (text_end - text_start)
        if ovp > 0.7:
            return text_segment["sentence_embeddings"]
    return []

def text_context_similarity(textemb_data, segment_start, segment_end, sid, segments, config):
    context_size = config['context_size']
    reference_segment = get_matching_textsegment(textemb_data, segment_start, segment_end)
    
    if len(reference_segment) == 0:
        return [0] * (2 * context_size)
    
    def get_context_embeddings(offset):
        context_embeddings = []
        for i in range(1, context_size + 1):
            idx = sid + i * offset
            if 0 <= idx < len(segments):
                start, end = segments[idx][1]['start'], segments[idx][1]['end']
                context_embeddings.append(get_matching_textsegment(textemb_data, start, end))
            else:
                context_embeddings.append([])
        return context_embeddings

    before_embeddings = get_context_embeddings(-1)
    after_embeddings = get_context_embeddings(1)

    def compute_similarities(context_embeddings):
        return [
            np.max(cosine_similarity(reference_segment, context).flatten())
            if len(context) > 0 else 0
            for context in context_embeddings
        ]

    return compute_similarities(before_embeddings) + compute_similarities(after_embeddings)

def segment_based_aggregation(speaker_start, speaker_end, clip_data, sbd_data,
                              sd_data, imgemb_data, sentiment_data, pos_data,
                              ner_data, textemb_data, sid, segments, config):
    vector = []

    clip_features = process_clip_data(clip_data, speaker_start, speaker_end, config)
    for feature_list in clip_features:
        vector.extend(feature_list)

    vector.append(speaker_end - speaker_start)
    vector.append(process_density_data(sd_data, speaker_start, speaker_end))
    vector.append(process_shots_data(sbd_data, speaker_start, speaker_end))
    vector.extend(process_sentiment_data(sentiment_data, speaker_start, speaker_end))
    vector.extend(process_pos_data(pos_data, speaker_start, speaker_end))
    vector.extend(process_ner_data(ner_data, speaker_start, speaker_end))
    vector.extend(image_context_similarity(imgemb_data, speaker_start, speaker_end, sid, segments, config))
    vector.extend(text_context_similarity(textemb_data, speaker_start, speaker_end, sid, segments, config))

    return vector


def window_based_aggregation(speaker_start, speaker_end, clip_data, sbd_data,
                            sd_data, imgemb_data, sentiment_data, pos_data,
                            ner_data, textemb_data, sid, segments, config):
    segment_features = []

    # Calculate FPS-adjusted start and end times once
    speaker_start_fps = round(speaker_start * config['fps']) / config['fps']
    speaker_end_fps = round(speaker_end * config['fps']) / config['fps']
    
    for window_length in config['window_lengths']:
        

        windows = [
            (start, min(start + window_length, speaker_end_fps))
            for start in np.arange(speaker_start_fps, speaker_end_fps, window_length)
        ]

        window_features = [
            {
                'clip': process_clip_data(clip_data, start, end, config),
                'shot_density': process_density_data(sd_data, start, end),
                'sentiments': process_sentiment_data(sentiment_data, start, end),
                'shots': process_shots_data(sbd_data, start, end)
            }
            for start, end in windows
        ]

        clip_features = [f['clip'] for f in window_features]
        for domain_features in zip(*clip_features):
            segment_features.extend(np.mean(domain_features, axis=0))
        
        segment_features.append(np.mean([f['shot_density'] for f in window_features]))
        segment_features.extend(np.quantile([f['sentiments'] for f in window_features], 0.8, axis=0))
        segment_features.append(np.mean([f['shots'] for f in window_features]))

    segment_features.append(speaker_end - speaker_start)
    segment_features.extend(process_pos_data(pos_data, speaker_start, speaker_end))
    segment_features.extend(process_ner_data(ner_data, speaker_start, speaker_end))
    segment_features.extend(image_context_similarity(imgemb_data, speaker_start, speaker_end, sid, segments, config))
    segment_features.extend(text_context_similarity(textemb_data, speaker_start, speaker_end, sid, segments, config))

    return segment_features

def main():
    parser = argparse.ArgumentParser(description='Aggregate features for the ground truth speaker turns')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    parser.add_argument('--task', type=str, default='speaker', choices=['speaker', 'situations'], help='Task to perform')
    parser.add_argument('--feature_type', type=str, default='segmentbased', choices=['segmentbased', 'windowbased'], help='Feature type to use for speaker embeddings')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    feature_dir = config["speaker_feature_dir"] if args.task == "speaker" else config["situation_feature_dir"]
    clip_path = config["speaker_clips_dir"] if args.task == "speaker" else config["situation_clips_dir"]
    
    clip_metadata = pd.read_csv(os.path.join(clip_path, "clip_list.txt"))

    video_segments_map = defaultdict(list)
    for _, row in clip_metadata.iterrows():
        base_video_name = row['clip_name'].rsplit('_', 1)[0]
        dt = {"start": row['start_time'], "end": row['end_time']}
        if args.task == "speaker":
            dt.update({"label_0": row['label_0'], "label_1": row['label_1']})
        else:
            dt.update({"label": row['label']})
        video_segments_map[base_video_name].append((row['clip_name'], dt))

    print(f"Total unique videos: {len(video_segments_map)}")
    print(f"Total number of samples: {sum(len(segments) for segments in video_segments_map.values())}\n")

    feature_dict = {source: {} for source in ["com", "tag", "bild"]}
    for video_name, segments in video_segments_map.items():
        print(f"Processing video: {video_name}")

        source = "tag" if video_name.startswith("TV") else "com" if "com" in video_name.lower() else "bild"

        feature_files = {
            'clip_data': "clip_qas.pkl",
            'sbd_data': "shot_boundary_detection.pkl",
            'sd_data': "shot_density.pkl",
            'imgemb_data': "image_embedding.pkl",
            'sentiment_data': "sentiment.pkl",
            'pos_data': "pos_tags.pkl",
            'ner_data': "ner_tags.pkl",
            'textemb_data': "sentence_embedding.pkl"
        }

        feature_data = {
            key: load_pickle(os.path.join("dataset/features", video_name, filename))
            for key, filename in feature_files.items()
        }

        aggregation_func = segment_based_aggregation if args.feature_type == "segmentbased" else window_based_aggregation
        
        for sid, (key, data) in enumerate(segments):
            speaker_start, speaker_end = data['start'], data['end']
            vector = aggregation_func(speaker_start, speaker_end, sid=sid, segments=segments, config=config, **feature_data)
            print(len(vector))
            feature_entry = {"vector": np.array(vector), "start": speaker_start, "end": speaker_end}
            if args.task == "speaker":
                feature_entry.update({"label_0": data['label_0'], "label_1": data['label_1']})
            else:
                feature_entry.update({"label": data['label']})
            
            feature_dict[source][key] = feature_entry

    print("\tFeature processing done!")
    for source in feature_dict:
        print(f"\tNumber of samples: {source.capitalize()}: {len(feature_dict[source])}")
    print(f"\tTotal number of samples: {sum(len(samples) for samples in feature_dict.values())}")

    for source, features in feature_dict.items():
        feature_file = f"{feature_dir}/{source}_features_{args.feature_type}.pkl"
        with open(feature_file, 'wb') as pkl:
            pickle.dump(features, pkl)
        print(f"Saved features for {source} to {feature_file}")

if __name__ == "__main__":
    main()