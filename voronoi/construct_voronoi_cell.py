import spacy
import json
import os
from tqdm import tqdm 
from rank_bm25 import BM25Okapi



nlp = spacy.load('en_core_web_lg')

GKB_ROOT = "..."
GKB_BEST_PATH = "..."
GKB_BEST_VOCAB_PATH = "..."
GKB_SENT_PATH = "..."
OUTPUT_CELL_PATH = "..."
EXAPNDED_CELL_PATH = "..."
EXAPNDED_SENT_CELL_PATH = "..."




with open(os.path.join(GKB_ROOT, GKB_BEST_VOCAB_PATH), "r") as f:
    GKB_best_vocab = f.read().strip().split("\n")


tokenized_GKB_best_vocab = [chunk.split(" ") for chunk in GKB_best_vocab]
GKB_best_bm25 = BM25Okapi(tokenized_GKB_best_vocab)


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
        if sent.lower() in sentence_set:
            continue
        sentence_set.add(sent.lower())
        remark = dict(source=source, title=term, quantifier=quantifier, score=float(score))
        gkb_corpus[sent_id]=dict(sent_id=sent_id, sentence=sent, remark=remark )
    return gkb_corpus


def add_knowledge(term, cell_info):
    add_sents = []
    for info in cell_info:
        tokens = info["tokens"]
        if term in tokens:
            add_sents.append(info)
    return add_sents




def main():
    gkb_corpus = formatter_gkbcorpus(os.path.join(GKB_ROOT, GKB_BEST_PATH))
    terms = [item["remark"]["title"] for sent_id, item in gkb_corpus.items()]
    sentences = [item["sentence"] for sent_id, item in gkb_corpus.items()]
    print("# sentence (unique):", len(sentences))
    docs = nlp.pipe(sentences)  # multi-threading

    voronoi_cell = dict()

    for index, doc in tqdm(enumerate(docs), total=len(sentences)):
        cell_term = terms[index]
        cell_term = cell_term.lower()
        if cell_term not in voronoi_cell:
            voronoi_cell[cell_term] = []
        tokens = [t.text for t in doc]
        pos_tags = [t.pos_ for t in doc]
        lemmas = [t.lemma_ for t in doc]
          
        
        # find utterance noun chunks
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
            if len(chunk_text) > 1:
                noun_chunks.append(chunk_text)


        # loacte in GKB vocab
        GKB_chunks = []
        for chunk in noun_chunks:
            chunk_text = chunk
            if chunk_text in GKB_best_vocab:
                GKB_chunks.append((chunk_text, "GKB_BEST"))
            elif " " in chunk_text:
                span_flag = False
                tokenized_query = chunk_text.split(" ")
                GKB_span = GKB_best_bm25.get_top_n(tokenized_query, GKB_best_vocab, n=1)[0]
                if GKB_span in GKB_best_vocab:
                    span_flag = True
                    GKB_chunks.append((GKB_span, "BEST_BM25"))
                if not span_flag:
                    GKB_chunks.append((GKB_span, "Copy"))
            else:
                GKB_chunks.append((GKB_span, "Copy"))
        
        voronoi_cell[cell_term].append({"sent_id":index, "GKB_chunks": GKB_chunks, "tokens":tokens})
    
    
    with open(os.path.join(GKB_ROOT, OUTPUT_CELL_PATH), "w") as f:
        json.dump(voronoi_cell, f, )


def search_expand():
    for term, cell_info in tqdm(voronoi_cell.items(), total=len(voronoi_cell)):
        num = len(cell_info)
        if num < 20:
            for term, cell_info in voronoi_cell.items():
                for info in cell_info:
                    chunks = [chunk[0] for chunk in info["GKB_chunks"]]
                    if term in chunks:
                        voronoi_cell[term].append(info)
    return voronoi_cell



if __name__ == "__main__":
    with open(os.path.join(GKB_ROOT, GKB_SENT_PATH)) as f:
        GKB_sents = f.read().splitlines()
    
    with open(os.path.join(GKB_ROOT, OUTPUT_CELL_PATH), "r") as f:
        voronoi_cell = json.load(f)

    voronoi_cell = search_expand()
    
    with open(os.path.join(GKB_ROOT, EXAPNDED_CELL_PATH), "w") as f:
        json.dump(voronoi_cell, f, )

    with open(os.path.join(GKB_ROOT, EXAPNDED_CELL_PATH), "r") as f:
        voronoi_cell = json.load(f)

    for term, cell_info in tqdm(voronoi_cell.items(), total=len(voronoi_cell)):
        for i, cell in enumerate(cell_info):
            if isinstance(cell, int):
                cell_sent = GKB_sents[cell]
                doc = nlp(cell_sent)
                tokens = [t.text for t in doc]
                pos_tags = [t.pos_ for t in doc]
                lemmas = [t.lemma_ for t in doc]
                
                # find utterance noun chunks
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
                    if len(chunk_text) > 1:
                        noun_chunks.append(chunk_text)


                # loacte in GKB vocab
                GKB_chunks = []
                for chunk in noun_chunks:
                    chunk_text = chunk
                    if chunk_text in GKB_best_vocab:
                        GKB_chunks.append((chunk_text, "GKB_BEST"))
                    elif " " in chunk_text:
                        span_flag = False
                        tokenized_query = chunk_text.split(" ")
                        GKB_span = GKB_best_bm25.get_top_n(tokenized_query, GKB_best_vocab, n=1)[0]
                        if GKB_span in GKB_best_vocab:
                            span_flag = True
                            GKB_chunks.append((GKB_span, "BEST_BM25"))
                        if not span_flag:
                            GKB_chunks.append((chunk_text, "Copy"))
                    else:
                        GKB_chunks.append((chunk_text, "Copy"))

                cell_info[i] = {"sent_id": -1, "GKB_chunks": GKB_chunks, "tokens":tokens}
                voronoi_cell[term] = cell_info

    with open(os.path.join(GKB_ROOT, EXAPNDED_SENT_CELL_PATH), "r") as f:
        json.dump(f)
    
    main()


