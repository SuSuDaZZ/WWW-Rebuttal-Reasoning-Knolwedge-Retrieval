"""Preprocess the corpus with spacy."""

import spacy
import json
import os
from tqdm import tqdm 
import numpy as np
import language_tool_python
from spacy.matcher import Matcher
from spellchecker import SpellChecker


nlp = spacy.load('en_core_web_lg')
# nlp.pipeline = [('tagger', nlp.tagger), ('parser', nlp.parser)]


DATA_ROOT = "data/GenericsKB"
CORPUS_PATH = "GenericsKB-Best.tsv"
OUTPUT_JSON_PATH = "GKB_best.jsonl"



def formatter_gkbcorpus(filepath):
    with open(filepath) as f:
        lines = f.read().split("\n")
    gkb_corpus = {}
    sentence_set = set()
    print("# sentence (original):", len(lines))
    for line in lines[1:-1]:
        index = len(gkb_corpus)
        sent_id = "gkb-best#%d"%index
        source, term, quantifier, sent, score = line.split("\t")
        sent = sent.replace(" isa ", " is a ")
        sent = sent.replace(" have (part) ", " have ")
        sent = sent.replace(" has-part ", " have ")
        sent = sent.replace(" lat me ", " have ")
        if sent.lower() in sentence_set:
            continue
        sentence_set.add(sent.lower())
        remark = dict(source=source, title=term, quantifier=quantifier, score=float(score))
        gkb_corpus[sent_id]=dict(sent_id=sent_id, sentence=sent, remark=remark )
    return gkb_corpus


def main():
    gkb_corpus = formatter_gkbcorpus(os.path.join(DATA_ROOT, CORPUS_PATH))
    
    terms = [item["remark"]["title"] for sent_id, item in gkb_corpus.items()]
    
    sentences = [item["sentence"] for sent_id, item in gkb_corpus.items()]
    print("# sentence (unique):", len(sentences))
    docs = nlp.pipe(sentences)  # multi-threading

    for index, doc in tqdm(enumerate(docs), total=len(sentences)):
        sent_id = "gkb-best#%d"%index
        tokens = [t.text for t in doc]
        pos_tags = [t.pos_ for t in doc]
        lemmas = [t.lemma_ for t in doc]
        deps = [t.dep_ for t in doc]
        noun_chunks = []
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.split()
            start = chunk.start
            end = chunk.end
            while start < end and pos_tags[start] not in ["NOUN", "ADJ", "PROPN"]:
                start += 1
            if start == end:
                continue
            chunk_text = " ".join(lemmas[start:end])
            chunk_deps = deps[start:end]
            noun_chunks.append(chunk_text)
        gkb_corpus[sent_id]["term"] = terms[index]
        gkb_corpus[sent_id]["tokens"] = tokens
        gkb_corpus[sent_id]["noun_chunks"] = noun_chunks
    
    json_lines = [json.dumps(item) for _, item in gkb_corpus.items()]
    with open(os.path.join(DATA_ROOT, OUTPUT_JSON_PATH), "w") as f:
        f.write("\n".join(json_lines))
    

if __name__ == "__main__":
    
    main()
