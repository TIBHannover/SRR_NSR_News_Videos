<div align="center"> 
  
# Identification of Speaker Roles and Situation Types in News Videos

</div> 

This is the official github page for the [paper](https://dl.acm.org/doi/10.1145/3652583.3658101):

> Gullal S. Cheema, Judi Arafat, Chiao-I Tseng, John A. Bateman, Ralph Ewerth, and Eric Müller-Budack. 2024. Identification of Speaker Roles and Situation Types in News Videos. In Proceedings of the 2024 International Conference on Multimedia Retrieval (ICMR '24). Association for Computing Machinery, New York, NY, USA, 506–514. https://doi.org/10.1145/3652583.3658101

## Environment Setup

- Create environment for training/evaluation

  `conda env create -f environment_main.yml`

- Install transnetv2, clip and download spacy model [For feature extraction]

  - Clone [TransNetV2](https://github.com/soCzech/TransNetV2)
      ```
      cd TransNetV2
      python setup.py install
      ```

  - `pip install git+https://github.com/openai/CLIP.git`

  - `python -m spacy download de_core_news_lg`

-  Additionally setup environment for diarization and feature extraction

    `conda env create -f environment_diarization.yml`

   - Install whisperx: `pip install git+https://github.com/m-bain/whisperX.git@4cb167a225c0ebaea127fd6049abfaa3af9f8bb4`


## Dataset

- Splits, features and speaker turn clip names used in the paper are available in `dataset/`
- Drop an [email](mailto:gullalcheema@gmail.com) for access to speaker turn clips and full videos
- Structure of feature pickle files (`bild_features_segmentbased.pkl` or `bild_features_windowbased.pkl`):
  - Dictionary with keys as clip names (e.g., '20220105_Corona_Regeln_Unsere_Freiheit_gerät_0RM8KQi3Muk_1')
    - Each _speaker type clip name_ key contains a dictionary with the following keys:
      - 'feature': Array containing the feature vector
      - 'label_0': Speaker role label for level 0 (_anchor, reporter, external_)
      - 'label_1': Speaker role label for level 1 (_anchor, reporter, expert, politician, layperson, other_)
      - 'start': Start time of the speaker turn in seconds
      - 'end': End time of the speaker turn in seconds
    - Each _situation type clip name_ key contains a dictionary with the following keys:
      - 'feature': Array containing the feature vector
      - 'label': News situation label (_talking-head, voiceover, interview, commenting, speech_)
      - 'start': Start time of the speaker turn in seconds
      - 'end': End time of the speaker turn in seconds
    - Example:
      ```
      {
        '20220105_Corona_Regeln_Unsere_Freiheit_gerät_0RM8KQi3Muk_1': {
          'feature': array([7.32914681e-02, 6.23861488e-03, ...],  # Feature vector
          'label_0': 2,
          'label_1': 3,
          'start': 1.579,
          'end': 29.7
        },
        '20220120_Omikron_Welle_Diese_Impfpflicht_ist_pWZDF3rJ744_231': {
          ...
        },
        ...
      }
      ```

## Feature Extraction
- Features used in the paper are already extracted in `/dataset/{task}/features`
- Otherwise, features for the dataset can be extracted from scripts in `feature_extraction/`
- **Audio Features:**
  - ASR: `conda activate fn1_main & python feature_extraction/audio_features.py --feature asr`
  - Diarization: `conda activate fn1_diar & python feature_extraction/audio_features.py --feature asr`
- **Text Features**
  - `conda activate fn1_main & python feature_extraction/text_features.py --features sentiment pos ner embeddings`
- **Visual Features**
  - Shot Features: `conda activate fn1_main & python feature_extraction/visual_features.py --feature shot-bd`
  - CLIP Features: `conda activate fn1_main & python feature_extraction/visual_features.py --feature clip`
- **Create Aggregated Feature set**  
  - `conda activate fn1_main & python feature_extraction/create_feature_set.py --task speaker --feature_type segmentbased`
  - Choose task and feature type options as required.


## Training & Evaluation
- K-Fold: `python training_eval/classifiers_kfold.py --rf --srr --seg --hierarchy 0`
  - Trains a random forest model for speaker role recognition (level-1) over segment-based features in a K-Fold setting
  - Outputs the metric scores for K test splits
- Cross-domain: `python training_eval/classifiers_cross.py --rf --srr --seg --hierarchy 0 --test tag`
  - Trains a random forest model for speaker role recognition (level-1) over segment-based features, with training data from BildTV + CompactTV and test data from Tagesschau
  - Outputs metric score for only one test split, in this case, Tagesschau

