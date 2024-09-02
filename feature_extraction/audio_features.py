import os
import pickle
import yaml
import tempfile
import subprocess
import argparse
from typing import Dict, Any
import torch
from moviepy.editor import VideoFileClip
from feature_utils import set_seeds

def asr(video_path: str, model: Any, config: Dict[str, Any]) -> None:
    """
    Perform Automatic Speech Recognition (ASR) using OpenAI's Whisper.

    Args:
        video_path (str): Path to the input video file.
        model (Any): Loaded Whisper model.
        config (Dict[str, Any]): Configuration dictionary.

    Note:
        Uses OpenAI's Whisper (https://github.com/openai/whisper)
    """
    output_dir = config['video_feature_dir']
    video_name = os.path.basename(video_path).replace('.mp4', '')
    os.makedirs(f"{output_dir}/{video_name}", exist_ok=True)
    pkl_file = f"{output_dir}/{video_name}/asr.pkl"

    if os.path.exists(pkl_file):
        print(f"\t[ASR] Found pkl: {pkl_file}, skipping Whisper ASR", flush=True)
        return

    result = model.transcribe(video_path, language='de')
    
    with open(pkl_file, 'wb') as pkl:
        pickle.dump(result, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'\t[ASR] Successfully applied Whisper ASR! Result: {pkl.name}', flush=True)

def speaker_diarization(video_path: str, config: Dict[str, Any]) -> None:
    """
    Perform speaker diarization using OpenAI's Whisper ASR.

    Args:
        video_path (str): Path to the input video file.
        config (Dict[str, Any]): Configuration dictionary.

    Note:
        Uses the whisper-diarization github (https://github.com/MahmoudAshraf97/whisper-diarization)
    """
    output_dir = config['video_feature_dir']
    video_name = os.path.basename(video_path).replace('.mp4', '')
    os.makedirs(f"{output_dir}/{video_name}", exist_ok=True)
    pkl_file = f"{output_dir}/{video_name}/speaker_diarization.pkl"

    if os.path.exists(pkl_file):
        print(f"\t[Speaker Diarization] Found pkl: {pkl_file}, skipping Whisper Speaker Diarization", flush=True)
        return

    with tempfile.NamedTemporaryFile(suffix='.wav', dir=config['tmp_dir']) as audio_file:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_file.name, verbose=False, logger=None)
        print("\t[Speaker Diarization] Processing Whisper Speaker Diarization...", flush=True)

        subprocess.run(
            f"conda run -n fn1_diar python {config['spk_diarization_script']} "
            f"-a {audio_file.name} --whisper-model {config['whisper_model']} --no-stem --output {pkl_file}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    if os.path.exists(pkl_file):
        print(f"\t[Speaker Diarization] Successfully applied Whisper Speaker Diarization! Result: {pkl_file}", flush=True)
    else:
        print(f"\t[Speaker Diarization] ERROR: Whisper Speaker Diarization failed!", flush=True)

def main():
    parser = argparse.ArgumentParser(description='Extract audio features from videos')
    parser.add_argument('--feature', type=str, default='asr', choices=['asr', 'diar'],
                        help='Feature to extract (asr: Automatic Speech Recognition, diar: Speaker Diarization)')
    args = parser.parse_args()

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    set_seeds(config['seed'])
    os.makedirs(config['tmp_dir'], exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    if args.feature == 'asr':
        import whisper
        model = whisper.load_model(config['whisper_model']).to(device)

    videos = os.listdir(config['videos_dir'])
    for v, video in enumerate(videos, 1):
        print(f"[{v}/{len(videos)}] Processing video: {video}", flush=True)
        video_path = os.path.join(config['videos_dir'], video)
        
        if args.feature == 'asr':
            print("\t[ASR] Applying Whisper ASR...", flush=True)
            asr(video_path, model, config)
        elif args.feature == 'diar':
            print("\t[Speaker Diarization] Applying Whisper Speaker Diarization...", flush=True)
            speaker_diarization(video_path, config)
        print()

if __name__ == "__main__":
    main()