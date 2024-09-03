
from .ner_functions import *
import numpy as np
from sortedcollections import OrderedSet

def get_ner_outputs(proc_text, proc_segments, ner_dict, event_set, wikifier_key):

    stanza_nes = get_stanza_ner_annotations(proc_text)

    wikifier_nes = get_wikifier_annotations(proc_text, wikifier_key)

    linked_entities = link_annotations(stanza_nes, wikifier_nes)

    linked_entities = fix_entity_types(linked_entities, event_set)
    ent_map = {}
    for ent in linked_entities:
        ent_map[ent["text"]] = ent["type"]


    ## Sent-wise ent vectors
    sent_ent_vectors = []
    for sent in proc_text.sentences:
        temp_nes = []
        ner_vector = np.zeros(6)
        for ent in sent.ents:
            if ent.text in ent_map:
                temp_nes.append(ent.text+"_"+ent_map[ent.text])
                ner_vector[ner_dict[ent_map[ent.text]]] += 1

        sent_ent_vectors.append({"tags": temp_nes, "vector": ner_vector})


    ## Segment-wise ent vectors
    seg_ent_vectors = []
    for seg_doc in proc_segments:
        temp_nes = []
        ner_vector = np.zeros(6)
        for sent in seg_doc.sentences:
            for ent in sent.ents:
                if ent.text in ent_map:
                    temp_nes.append(ent.text+"_"+ent_map[ent.text])
                    ner_vector[ner_dict[ent_map[ent.text]]] += 1

        seg_ent_vectors.append({"tags": temp_nes, "vector": ner_vector})


    return sent_ent_vectors, seg_ent_vectors