import os
import pickle
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer, util



if __name__ == "__main__":

    with open("GKB_voronois.pkl", "rb") as f:
        cache_data = pickle.load(f)
        GKB_sentences = cache_data["GKB_sents"]
        GKB_chunks = cache_data["GKB_chunks"]
        voronoi_cell = cache_data["voronoi_cell"]

    dpr_encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base')
    dpr_encoder.save('data/baselines/facebook-dpr-ctx_encoder-single-nq-base')
    
    
    sent_embedds = dpr_encoder.encode(GKB_sentences, show_progress_bar=True, convert_to_numpy=True)

    with open("GKB_sents_dpr_embeds.pkl", "wb") as f:
        pickle.dump({"GKB_sents": GKB_sentences, "GKB_sents_embeds": sent_embedds}, f)


    mpnet = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")  
    sent_embedds = mpnet.encode(GKB_sentences, show_progress_bar=True, convert_to_numpy=True)

    with open("GKB_sents_mpnet_embeds.pkl", "wb") as f:
        pickle.dump({"GKB_sents": GKB_sentences, "GKB_sents_embeds": sent_embedds}, f)

    model = SentenceTransformer('nthakur/contriever-base-msmarco')
    sent_embedds = model.encode(GKB_sentences, show_progress_bar=True, convert_to_numpy=True)
    with open("GKB_sents_contriever_embeds.pkl", "wb") as f:
        pickle.dump({"GKB_sents": GKB_sentences, "GKB_sents_embeds": sent_embedds}, f)

    model = SentenceTransformer('thenlper/gte-large')
    sent_embedds = model.encode(GKB_sentences, show_progress_bar=True, convert_to_numpy=True)

    with open("/GKB_sents_gte_embeds.pkl", "wb") as f:
        pickle.dump({"GKB_sents": GKB_sentences, "GKB_sents_embeds": sent_embedds}, f)

    model = INSTRUCTOR('hkunlp/instructor-large')
    sent_embedds = model.encode(GKB_sentences, show_progress_bar=True, convert_to_numpy=True)
    with open("/GKB_sents_instructor_embeds.pkl", "wb") as f:
        pickle.dump({"GKB_sents": GKB_sentences, "GKB_sents_embeds": sent_embedds}, f)

