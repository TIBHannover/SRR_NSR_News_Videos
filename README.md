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



## To-Dos
- [ ] Comparison methods code
