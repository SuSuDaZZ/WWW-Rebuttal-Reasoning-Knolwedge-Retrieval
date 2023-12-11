import os
import time
import json
import nltk
import torch
import spacy
import pickle
import cupy as cp
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain
from nltk.stem.snowball import SnowballStemmer
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
from cuml.cluster import HDBSCAN
from cuml.preprocessing import normalize



# nlp = spacy.load('en_core_web_lg')

GKB_ROOT = "/data/GenericsKB"
GKB_BEST_PATH = "GenericsKB-Best.tsv"
GKB_BEST_VOCAB_PATH = "GKB_best.vocab.txt"
GKB_SENT_PATH = "GKB_best.sentences.txt"
OUTPUT_CELL_PATH = "GKB_best_cell.jsonl"
OUTPUT_EXPAND_CELL_PATH = "GKB_best_expand_cell.jsonl"
GKB_EMBEDS_CACHE_PATH = "GKB-sents-chunks-embeddings-mpnet.pkl"
CHUNK_TO_ID_PATH = "GKB_chunk_to_id.jsonl"
GKB_CHUNK_GRAPH_PATH = "GKB_chunk_path.jsonl"
SYNONYMS_JSON_PATH = "projects/MCTS_ConceptNet/conceptnet/conceptnet.synonyms.jsonl"


GKB_sents_chunks_path = "GKB_sents_chunks.pkl"
GKB_sentence_embeddings_path = "GKB_sents_embeds.pkl"
GKB_chunk_to_sents_path = "GKB_chunk_to_sents.pkl"
GKB_chunks_embeddings_path = "GKB_chunks_embeds.pkl"

GKB_voronoi_path = "GKB_voronois.pkl"
GKB_invert_table_path = "GKB_invert_table.pkl"
GKB_sents_embeds_path = "GKB_sents_embeds_new.pkl"
GKB_chunks_embeds_path = "GKB_chunks_embeds_new.pkl"
central_chunks_path = "central_chunks_new.pkl"
central_chunks_embeds_path = "central_chunks_embeds_new.pkl"
central_chunks_cluster_path = "central_chunks_cluster_new.pkl"
central_chunks_cluster_label_path = "central_chunks_cluster_label_new.pkl"
split_chunks_path = "split_chunks_new.pkl"
GKB_chunks_path = "GKB_chunks_new.pkl"


GKB_sents_embeds_path = "GKB_sents_dpr_embeds.pkl"
GKB_sents_embeds_path = "GKB_sents_SGPT_embeds.pkl"



if __name__ == "__main__":

    with open(os.path.join(GKB_ROOT, GKB_voronoi_path), "rb") as f:
        cache_data = pickle.load(f)
        GKB_sentences = cache_data["GKB_sents"]
        GKB_chunks = cache_data["GKB_chunks"]
        voronoi_cell = cache_data["voronoi_cell"]
    
    print("data loaded")
    central_chunks = list(voronoi_cell.keys())


    split_chunks = dict()
    for chunk in tqdm(central_chunks, total=len(central_chunks)):
        if " " in chunk:
            splits = chunk.split(" ")
            for split in splits:
                if split not in split_chunks:
                    split_chunks[split] = [chunk]
                else:
                    split_chunks[split].append(chunk)

    with open(os.path.join(GKB_ROOT, split_chunks_path), "wb") as f:
        pickle.dump(split_chunks, f)


    # sentence embeddings
    mpnet = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")  
    sent_embedds = mpnet.encode(GKB_sentences, show_progress_bar=True, convert_to_numpy=True)

    dpr_encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base')
    dpr_encoder.save('data/baselines/facebook-dpr-ctx_encoder-single-nq-base')
    
    
    sent_embedds = dpr_encoder.encode(GKB_sentences, show_progress_bar=True, convert_to_numpy=True)

    with open("data/baselines/GKB_sents_dpr_embeds.pkl", "wb") as f:
        pickle.dump({"GKB_sents": GKB_sentences, "GKB_sents_embeds": sent_embedds}, f)


    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Muennighoff/SGPT-125M-weightedmean-nli-bitfit")
    model = AutoModel.from_pretrained("Muennighoff/SGPT-125M-weightedmean-nli-bitfit")
    # Deactivate Dropout (There is no dropout in the above models so it makes no difference here but other SGPT models may have dropout)
    model.eval()

    batch_tokens = tokenizer(GKB_sentences, padding=True, truncation=True, return_tensors="pt")

    # Get the embeddings
    with torch.no_grad():
        # Get hidden state of shape [bs, seq_len, hid_dim]
        last_hidden_state = model(**batch_tokens, output_hidden_states=True, return_dict=True).last_hidden_state

    # Get weights of shape [bs, seq_len, hid_dim]
    weights = (
        torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float().to(last_hidden_state.device)
    )

    # Get attn mask of shape [bs, seq_len, hid_dim]
    input_mask_expanded = (
        batch_tokens["attention_mask"]
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float()
    )

    # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
    sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

    sent_embedds = sum_embeddings / sum_mask

    with open(os.path.join(GKB_ROOT, GKB_sents_embeds_path), "wb") as f:
        pickle.dump({"GKB_sents": GKB_sentences, "GKB_sents_embeds": sent_embedds}, f)


    
    with open(os.path.join(GKB_ROOT, central_chunks_embeds_path), "rb") as f:
        central_chunk_embedds = pickle.load(f)

    # central_chunk_embedds = torch.from_numpy(central_chunk_embedds).to(device)

    X = normalize(cp.array(central_chunk_embedds))

    hdbscan_min_samples=25
    hdbscan_min_cluster_size=5
    hdbscan_max_cluster_size=1000
    hdbscan_cluster_selection_method="leaf"
    print(X.shape)

    hdbscan = HDBSCAN(min_samples=hdbscan_min_samples, 
                      min_cluster_size=hdbscan_min_cluster_size, 
                      max_cluster_size=hdbscan_max_cluster_size,
                      cluster_selection_method=hdbscan_cluster_selection_method)

    labels = hdbscan.fit_predict(X)

    

    print("Start clustering")
    start_time = time.time()
    clusters = util.community_detection(central_chunk_embedds, min_community_size=10, threshold=0.75)
    print("Clustering done after {:.2f} sec".format(time.time() - start_time))

    cluster_chunks = []
    for i, cluster in enumerate(clusters):
        chunks = []
        for chunk_id in cluster:
            chunk = central_chunks[chunk_id]
            chunks.append(chunk)
        cluster_chunks.append(chunks)
    
    
    with open(os.path.join(GKB_ROOT, central_chunks_cluster_path), "wb") as f:
        pickle.dump(cluster_chunks, f)

    chunk_cluster_label = dict()
    for i, cluster in enumerate(clusters):
        for chunk_id in cluster:
            chunk = central_chunks[chunk_id]
            if chunk not in chunk_cluster_label:
                chunk_cluster_label[chunk] = [i]
            else:
                chunk_cluster_label[chunk].append(i)

    split_chunks = dict()
    for chunk in central_chunks:
        if " " in chunk:
            splits = chunk.split(" ")
            for split in splits:
                if split not in split_chunks:
                    split_chunks[split] = [chunk]
                else:
                    split_chunks[split].append(chunk)
    
    
    chunk_scopes = dict()
    for chunk in central_chunks:
        chunk_scopes[chunk] = []
        # check cluster
        if chunk in chunk_cluster_label:
            cluster_labels = chunk_cluster_label[chunk]
            for label in cluster_labels:
                chunks = cluster_chunks[label]
                chunk_scopes[chunk] += chunks

        # check splits
        if chunk in split_chunks:
            splits = split_chunks[chunk]
            for split in splits:
                if split not in chunk_scopes[chunk]:
                    chunk_scopes[chunk].append(split)

        if " " in chunk:
            part_chunks = chunk.split(" ")
            higher_chunk = part_chunks[-1]
            if higher_chunk not in chunk_scopes[chunk]:
                chunk_scopes[chunk].append(split)

    
    with open(os.path.join(GKB_ROOT, central_chunks_cluster_label_path), "wb") as f:
        pickle.dump(chunk_scopes, f)
    

    hdbscan_min_samples=25
    hdbscan_min_cluster_size=5
    hdbscan_max_cluster_size=1000
    hdbscan_cluster_selection_method="leaf"

    norm_central_embeds = normalize(cp.array(central_chunk_embedds))
    print(norm_central_embeds.shape)


    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    cluster_labels = clusterer.fit_predict(central_chunk_embedds)



    with open(os.path.join(GKB_ROOT, OUTPUT_CELL_PATH), "r") as f:
        voronoi_cell = json.load(f)

    with open(SYNONYMS_JSON_PATH) as f:
        synonyms_dict = json.load(f)

    GKB_sentences = []
    GKB_chunks = []
    cell_index = dict()
    start = 0

    central_chunks = []

    for central_chunk, cell_info in tqdm(voronoi_cell.items(), total=len(voronoi_cell)):  
        start = len(GKB_sentences) 
        for info in cell_info:
            sent = " ".join(info["tokens"])
            GKB_sentences.append(sent)
            central_chunks.append(central_chunk)
            if len(info["GKB_chunks"]) == 0:
                GKB_chunks.append([])
            else:
                GKB_columns = list(zip(*info["GKB_chunks"]))
                chunk = GKB_columns[0]
                GKB_chunks.append(chunk)
        if central_chunk in synonyms_dict:
            knowledge_synonyms = synonyms_dict[central_chunk]
            for synonyms in knowledge_synonyms:
                if synonyms != central_chunk:
                    syn_sent = " ".join([central_chunk, "is", synonyms, "."])
                    GKB_sentences.append(syn_sent)
                    central_chunks.append(central_chunk)
                    GKB_chunks.append([central_chunk, synonyms])
        end = len(GKB_sentences) 
        cell_index[central_chunk] = [start, end]

    with open(os.path.join(GKB_ROOT, GKB_voronoi_path), "wb") as f:
        pickle.dump({"GKB_sents": GKB_sentences, "GKB_chunks": GKB_chunks, "voronoi_cell": cell_index}, f)

    with open(os.path.join(GKB_ROOT, central_chunks_path), "wb") as f:
        pickle.dump(central_chunks, f)

        
    

    # inverted table
    with open(os.path.join(GKB_ROOT, GKB_voronoi_path), "rb") as f:
        cache_data = pickle.load(f)
        GKB_sentences = cache_data["GKB_sents"]
        GKB_chunks = cache_data["GKB_chunks"]
        voronoi_cell = cache_data["voronoi_cell"]
    
    chunk_sent_table = voronoi_cell.copy()
    central_chunks = list(voronoi_cell.keys())
    for sent_id, (corpus_sent, corpus_chunks) in tqdm(enumerate(zip(GKB_sentences, GKB_chunks)), total=len(GKB_sentences)):
        if len(corpus_chunks)==0:
            continue
        for chunk in corpus_chunks:
            if chunk in central_chunks:
                central_span = voronoi_cell[chunk]
                central_range = [*range(central_span[0], central_span[1], 1)]
                if sent_id not in central_range:
                    chunk_sent_table[chunk].append(sent_id)
    
    # dope_ids = chunk_sent_table["dope"]
    with open(os.path.join(GKB_ROOT, GKB_invert_table_path), "wb") as f:
        pickle.dump(chunk_sent_table, f)

    
    # central chunk embeddings
    mpnet = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")  
    central_chunks = list(voronoi_cell.keys())
    central_chunk_embedds = mpnet.encode(central_chunks, show_progress_bar=True, convert_to_numpy=True)

    with open(os.path.join(GKB_ROOT, central_chunks_embeds_path), "wb") as f:
        pickle.dump(central_chunk_embedds, f)


    
    # sentence embeddings
    mpnet = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")  
    sent_embedds = mpnet.encode(GKB_sentences, show_progress_bar=True, convert_to_numpy=True)

    with open(os.path.join(GKB_ROOT, GKB_sents_embeds_path), "wb") as f:
        pickle.dump(sent_embedds, f)


    # chunk embeddings
    mpnet = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")  
    chunk_embeddings = []
    for sent_id, corpus_chunks in tqdm(enumerate(GKB_chunks), total=len(GKB_chunks)):
        if len(GKB_chunks)==0:
            chunk_embeddings.append(np.zeros(shape=(1, 768)))
            continue
        chunk_embed = mpnet.encode(corpus_chunks, convert_to_numpy=True)
        chunk_embeddings.append(chunk_embed)
    with open(os.path.join(GKB_ROOT, GKB_chunks_embeds_path), "wb") as f:
        pickle.dump(chunk_embeddings, f)    
    
    
    with open(os.path.join(GKB_ROOT, GKB_sents_chunks_path), "rb") as fIn:
        cache_data = pickle.load(fIn)
        cell_index = cache_data['cell_index']
        corpus_sentences = cache_data['sentences']
        corpus_GKB_chunks = cache_data['GKB_chunks']
        print("cache data loaded.")
    
    
    
    
    # inverted index -> central chunk to sentences table 
    chunk_sent_table = cell_index.copy()
    central_chunks = list(cell_index.keys())
    for sent_id, (corpus_sent, corpus_chunks) in tqdm(enumerate(zip(corpus_sentences, corpus_GKB_chunks)), total=len(corpus_sentences)):
        if corpus_chunks is None:
            continue
        for chunk in corpus_chunks:
            if chunk in central_chunks:
                central_span = cell_index[chunk]
                central_range = [*range(central_span[0], central_span[1], 1)]
                if sent_id not in central_range:
                    chunk_sent_table[chunk].append(sent_id)
    
    with open(os.path.join(GKB_ROOT, GKB_chunk_to_sents_path), "wb") as f:
        pickle.dump(chunk_sent_table, f)


   
    with open(os.path.join(GKB_ROOT, OUTPUT_CELL_PATH), "r") as f:
        voronoi_cell = json.load(f)

    with open(SYNONYMS_JSON_PATH) as f:
        synonyms_dict = json.load(f)

    for central_chunk, cell_info in tqdm(voronoi_cell.items(), total=len(voronoi_cell)):
        
        # Add Synonyms
        if central_chunk in synonyms_dict:
            knowledge_synonyms = synonyms_dict[central_chunk]
            for synonyms in knowledge_synonyms:
                if synonyms != central_chunk:
                    knowledge = {"GKB_chunks": [central_chunk, synonyms], "tokens": [central_chunk, "is", synonyms, "."]}
                    voronoi_cell[central_chunk].append(knowledge)


        # Add other sentences contains central_chunk
        for term, cell_info in tqdm(voronoi_cell.items(), total=len(voronoi_cell)):
            if central_chunk != term:
                for info in cell_info:
                    flatten_list = list(chain.from_iterable(info["GKB_chunks"]))
                    if central_chunk in flatten_list:
                        voronoi_cell[central_chunk].append(info)

    
    with open(os.path.join(GKB_ROOT, OUTPUT_EXPAND_CELL_PATH), "w") as f:
        json.dump(voronoi_cell, f)
    
    
    neighbor_connections = dict()
    
    # mpnet embeddings for sentence embedding and sum of chunks embeddings
    mpnet = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    
    corpus_sentences = []
    corpus_sentence_embeddings = []
    corpus_GKB_chunks = []
    corpus_chunk_embeddings = []
    cell_index = dict()


    for sent_id, (central_chunk, cell_info) in tqdm(enumerate(voronoi_cell.items()), total=len(voronoi_cell)):
        cell_index[central_chunk] = [sent_id, sent_id + len(cell_info)]
        for info_id, info in enumerate(cell_info):
            GKB_sent = " ".join(info["tokens"])
            corpus_sentences.append(GKB_sent)

            # sent_embedds = mpnet.encode(GKB_sent, convert_to_numpy=True)
            # corpus_sentence_embeddings.append(sent_embedds)

            GKB_columns = list(zip(*info["GKB_chunks"]))
            if len(GKB_columns) > 0:
                GKB_chunks = GKB_columns[0]
                corpus_GKB_chunks.append(GKB_chunks)

                # chunk_embedds = mpnet.encode(GKB_chunks, convert_to_numpy=True)
                # # chunk_embedds_sum = np.sum(chunk_embedds, axis=0)
                # corpus_chunk_embeddings.append(chunk_embedds)
            else:
                corpus_GKB_chunks.append(None)
                # corpus_chunk_embeddings.append(None)

    with open(os.path.join(GKB_ROOT, GKB_sents_chunks_path), "wb") as f:
        pickle.dump({'cell_index': cell_index, 'sentences': corpus_sentences, 'GKB_chunks':corpus_GKB_chunks}, f)
    
    
    with open(GKB_EMBEDS_CACHE_PATH, "wb") as fOut:
        pickle.dump({'cell_index': cell_index, 'sentences': corpus_sentences, 'GKB_chunks':corpus_GKB_chunks, 'sentence_embeddings': corpus_sentence_embeddings, 'GKB_chunk_embeddings':corpus_chunk_embeddings}, fOut)

    with open(GKB_EMBEDS_CACHE_PATH, "rb") as fIn:
        cache_data = pickle.load(fIn)
        corpus_sentences = cache_data['sentences']
        corpus_embeddings = cache_data['embeddings']
        print()
    

