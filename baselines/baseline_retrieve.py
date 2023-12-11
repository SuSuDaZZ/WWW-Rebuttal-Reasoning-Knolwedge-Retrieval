
import os
import time
import json
import torch
import pickle
import argparse
from tqdm import tqdm
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer, util
from transformers import BartTokenizer, BartForConditionalGeneration


import sys
class Logger(object):
    def __init__(self, filename='xxx.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger()


def cos_sim(a, b):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    
    return torch.mm(a_norm, b_norm.transpose(0, 1))   


class DialogSum():
    def __init__(self, model_path):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = BartForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained(model_path)
    
    def generate(self, conversation):
        model_input = self.tokenizer(conversation, return_tensors="pt").to(self.device)
        max_length = len(model_input["input_ids"].squeeze())

        generated_ids = self.model.generate(
            input_ids=model_input["input_ids"],
            min_length=int(0.5*max_length), 
            max_length=2*max_length)

        summary = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)  
        return summary   


if __name__=="__main__":

    # Data Loading
    start_time = time.time()
    with open("GKB/processed_dialog.jsonl", "r") as f:
        DailyDialog_corpus = json.load(f)

    GKB_ROOT = "GKB"
    GKB_voronoi_path = "GKB_voronois.pkl"

    with open(os.path.join(GKB_ROOT, GKB_voronoi_path), "rb") as f:
        cache_data = pickle.load(f)
        GKB_sentences = cache_data["GKB_sents"]
        GKB_chunks = cache_data["GKB_chunks"]
        voronoi_cell = cache_data["voronoi_cell"]

    mpnet_embeds_pth = "GKB_sents_mpnet_embeds.pkl"
    dpr_embeds_pth = "GKB_sents_dpr_embeds.pkl"
    contriever_embeds_pth = "GKB_sents_contriever_embeds.pkl"
    gte_embeds_pth = "GKB_sents_gte_embeds.pkl"
    instructor_embeds_pth = "GKB_sents_instructor_embeds.pkl"
    
    with open(gte_embeds_pth, "rb") as f:
        data = pickle.load(f)  
        passage_embeddings = data["GKB_sents_embeds"]


    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dialogsum = DialogSum("philschmid/bart-large-cnn-samsum")

    query_encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")  
    query_encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base')
    query_encoder = SentenceTransformer('thenlper/gte-large') 
    query_encoder = INSTRUCTOR('hkunlp/instructor-large')


    dataset_context_sents = []
    dataset_attn_context_sents = []

        
    for dialog_id, dialog in tqdm(enumerate(DailyDialog_corpus), total=len(DailyDialog_corpus)):

        print("DIALOG ID:", dialog_id+1, "\n")
        
        dialog_context = ""

        dialog_context_sents = []
        dialog_attn_context_sents = []

        for utter_id, corpus_sent in enumerate(dialog):

            print("*"*50, "utter", utter_id, "*"*50)
            utterance = corpus_sent["utterance"]
            if utter_id == len(dialog)-1:
                break
            elif (utter_id % 2) == 0:
                utterance = "PersonX: " + utterance
            else:
                utterance = "PersonY: " + utterance


            dialog_context = dialog_context + utterance + "\n"
            dialog_context_embed = query_encoder.encode(dialog_context)
                            
            context_summ = dialogsum.generate(dialog_context)[0]
            context_summ_embed = query_encoder.encode(context_summ)

            scores = util.dot_score(context_summ_embed, passage_embeddings).squeeze()
            scores = util.cos_sim(context_summ_embed, passage_embeddings).squeeze()
            topk_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
            topk_sents = [GKB_sentences[id] for id in topk_ids]
            dialog_context_sents.append(topk_sents)

        dataset_context_sents.append(dialog_context_sents)
            
    
    with open('xxx.json', "w") as fp:
        json.dump({"context_sents": dataset_context_sents}, fp) 



