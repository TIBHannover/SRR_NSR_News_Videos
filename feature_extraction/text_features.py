import os
import pickle
import stanza
import spacy
import yaml
import argparse
from feature_utils import set_seeds
from nlp_features import feat_functions
from germansentiment import SentimentModel
from sentence_transformers import SentenceTransformer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

def sentiment_analysis(video_path, model, config):
    output_dir = config['video_feature_dir']
    video_name = os.path.basename(video_path).replace('.mp4', '')
    os.makedirs(f"{output_dir}/{video_name}", exist_ok=True)
    sentiment_pkl = f"{output_dir}/{video_name}/sentiment.pkl"
    asr_pkl = f"{output_dir}/{video_name}/asr.pkl"

    if os.path.exists(sentiment_pkl):
        print(f"\t[Sentiment] Found pkl: {sentiment_pkl}, skipping", flush=True)
        return
    if not os.path.exists(asr_pkl):
        print("\t[Sentiment] ASR transcript not found, skipping", flush=True)
        return
    
    with open(asr_pkl, 'rb') as pkl:
        asr_data = pickle.load(pkl)
    
    texts = [segment['text'].strip() for segment in asr_data['segments']]
    preds, probs = model.predict_sentiment(texts, output_probabilities=True)
    
    sentiments = [
        {
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text'].strip(),
            'sentiment': preds[i],
            'sentiment_probs': probs[i]
        }
        for i, segment in enumerate(asr_data['segments'])
    ]
    
    with open(sentiment_pkl, 'wb') as pkl:
        pickle.dump(sentiments, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'\t[Sentiment] Result saved: {pkl.name}', flush=True)

def pos_tagger(video_path, spacy_nlp, config):
    output_dir = config['video_feature_dir']
    video_name = os.path.basename(video_path).replace('.mp4', '')
    os.makedirs(f"{output_dir}/{video_name}", exist_ok=True)
    tags_pkl = f"{output_dir}/{video_name}/pos_tags.pkl"
    diarization_pkl = f"{output_dir}/{video_name}/speaker_diarization.pkl"

    if os.path.exists(tags_pkl):
        print(f"\t[PoS Tagger] Found pkl: {tags_pkl}, skipping", flush=True)
        return
    if not os.path.exists(diarization_pkl):
        print("\t[PoS Tagger] Speaker Diarization data not found, skipping", flush=True)
        return

    with open(diarization_pkl, 'rb') as pkl:
        diarization_data = pickle.load(pkl)

    pos_dict = config['pos_dict']

    all_pos = []
    for segment in diarization_data:
        pos_vector = [0] * 14
        text = spacy_nlp(segment['text'])
        for token in text:
            pos_index = pos_dict.get(token.pos_, pos_dict['X'])
            pos_vector[pos_index] += 1

        all_pos.append({
            "start_time": segment['start_time'],
            "end_time": segment['end_time'],
            "vector": pos_vector
        })
    
    with open(tags_pkl, 'wb') as pkl:
        pickle.dump(all_pos, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\t[PoS Tagger] Result saved: {pkl.name}", flush=True)

def ner_tagger(video_path, stanza_nlp, config):
    output_dir = config['video_feature_dir']
    video_name = os.path.basename(video_path).replace('.mp4', '')
    os.makedirs(f"{output_dir}/{video_name}", exist_ok=True)
    tags_pkl = f"{output_dir}/{video_name}/ner_tags.pkl"
    asr_pkl = f"{output_dir}/{video_name}/asr.pkl"
    diarization_pkl = f"{output_dir}/{video_name}/speaker_diarization.pkl"

    if os.path.exists(tags_pkl):
        print(f"\t[NER Tagger] Found pkl: {tags_pkl}, skipping", flush=True)
        return
    if not os.path.exists(asr_pkl) or not os.path.exists(diarization_pkl):
        print("\t[NER Tagger] Required data not found, skipping", flush=True)
        return
    
    with open(asr_pkl, 'rb') as pkl:
        asr_data = pickle.load(pkl)
    with open(diarization_pkl, 'rb') as pkl:
        diarization_data = pickle.load(pkl)

    proc_text = stanza_nlp(asr_data["text"])
    proc_segments = [stanza_nlp(segment["text"]) for segment in diarization_data]

    ner_dict = {"EPER": 0, "LPER": 1, "LOC": 2, "ORG": 3, "EVENT": 4, "MISC": 5}
    event_set = set(line.strip() for line in open('feature_extraction/nlp_features/eventKG.csv'))

    sent_nes, seg_nes = feat_functions.get_ner_outputs(proc_text, proc_segments, ner_dict, event_set, config["wikifier_key"])

    all_ner = [
        {
            "start_time": diarization_data[i]['start_time'],
            "end_time": diarization_data[i]['end_time'],
            "text": diarization_data[i]['text'],
            "vector": seg['vector'],
            "tags": seg['tags']
        }
        for i, seg in enumerate(seg_nes)
    ]

    with open(tags_pkl, 'wb') as pkl:
        pickle.dump(all_ner, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\t[NER Tagger] Result saved: {pkl.name}", flush=True)

def sentence_embeddings(video_path, spacy_nlp, model, config):
    output_dir = config['video_feature_dir']
    video_name = os.path.basename(video_path).replace('.mp4', '')
    os.makedirs(f"{output_dir}/{video_name}", exist_ok=True)
    embeddings_pkl = f"{output_dir}/{video_name}/sentence_embedding.pkl"
    diarization_pkl = f"{output_dir}/{video_name}/speaker_diarization.pkl"

    if os.path.exists(embeddings_pkl):
        print(f"\t[Sentence Embeddings] Found pkl: {embeddings_pkl}, skipping", flush=True)
        return
    if not os.path.exists(diarization_pkl):
        print("\t[Sentence Embeddings] Speaker Diarization data not found, skipping", flush=True)
        return
    
    with open(diarization_pkl, 'rb') as pkl:
        diarization_data = pickle.load(pkl)

    embeddings = [
        {
            "start_time": segment['start_time'],
            "end_time": segment['end_time'],
            "sentences": [sent.text for sent in spacy_nlp(segment['text']).sents],
            "sentence_embeddings": model.encode([sent.text for sent in spacy_nlp(segment['text']).sents])
        }
        for segment in diarization_data
    ]
    
    with open(embeddings_pkl, 'wb') as pkl:
        pickle.dump(embeddings, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\t[Sentence Embeddings] Result saved: {pkl.name}", flush=True)

def main():
    parser = argparse.ArgumentParser(description='Extract NLP features from videos')
    parser.add_argument('--features', nargs='+', choices=['sentiment', 'pos', 'ner', 'embeddings'], 
                        default=['sentiment', 'pos', 'ner', 'embeddings'],
                        help='Features to extract')
    args = parser.parse_args()

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    set_seeds(config['seed'])

    models = {}
    if 'sentiment' in args.features:
        models['sentiment'] = SentimentModel('mdraw/german-news-sentiment-bert')
    if 'pos' in args.features or 'embeddings' in args.features:
        models['spacy'] = spacy.load("de_core_news_lg")
    if 'ner' in args.features:
        models['stanza'] = stanza.Pipeline(lang='de', processors='tokenize,ner,pos', download_method=None, logging_level='ERROR')
    if 'embeddings' in args.features:
        models['embeddings'] = SentenceTransformer('distiluse-base-multilingual-cased-v1')

    videos = os.listdir(config['videos_dir'])

    for v, video in enumerate(videos, 1):
        print(f"[{v}/{len(videos)}] Processing video: {video}", flush=True)
        video_path = f"{config['videos_dir']}/{video}"

        if 'sentiment' in args.features:
            print("\t[Sentiment] Processing SENTIMENT Analysis...", flush=True)
            sentiment_analysis(video_path, models['sentiment'], config)

        if 'pos' in args.features:
            print("\t[PoS Tagger] Processing POS Tagging...", flush=True)
            pos_tagger(video_path, models['spacy'], config)

        if 'ner' in args.features:
            print("\t[NER Tagger] Processing NER Tagging...", flush=True)
            ner_tagger(video_path, models['stanza'], config)
    
        if 'embeddings' in args.features:
            print("\t[Sentence Embeddings] Processing Sentence Embeddings...", flush=True)
            sentence_embeddings(video_path, models['spacy'], models['embeddings'], config)

        print()

if __name__ == "__main__":
    main()