import os
import time
import math
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
from mcts import mcts
from tqdm import tqdm
from copy import deepcopy
from itertools import chain
from operator import itemgetter

print("Program CUDA VERSION:", torch.version.cuda)
print("Program CUDA AVAILABLE:", torch.cuda.is_available())

import subprocess
command = 'nvidia-smi'  
open_process = subprocess.Popen(
    command,  
    stdout=subprocess.PIPE,
    shell=True  
)
cmd_out = open_process.stdout.read()  
open_process.stdout.close()  
print("NVIDIA-SMI OUTPUT:", cmd_out.decode(encoding="gbk"))


from rank_bm25 import BM25Okapi
from transformers import pipeline
from scipy.stats import wasserstein_distance
from sentence_transformers import SentenceTransformer, util
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from load_data import load_dialog, load_embeddings, load_voronoi, load_invert_table, load_synonyms, load_central_chunks, load_cluster, load_split_chunks, load_GKB_chunks, load_stop_words, get_rels_transation
from comet_generation import Comet, ContextGPT2

stop_words = set(load_stop_words())


K = 5
C = 3

print("Program CUDA VERSION:", torch.version.cuda)
print("Program CUDA AVAILABLE:", torch.cuda.is_available())

parser = argparse.ArgumentParser()
parser.add_argument('--START', type=int, default=0)
parser.add_argument('--END', type=int, default=10)
args = parser.parse_args()

import sys
class Logger(object):
    def __init__(self, filename='xxx.log'.format(args.START, args.END), stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger()

retrieved_knowledge_path = "xxx.txt".format(args.START, args.END)


class DialogSum():
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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


class DSE():
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def get_average_embedding(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        # Calculate the sentence embeddings by averaging the embeddings of non-padding words
        with torch.no_grad():
            embeddings = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            embeddings = torch.sum(embeddings[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
            return embeddings


class DialogRPT():
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
    
    def score(self, batch_sents):
        # context_str = "<|endoftext|>".join(context)
        # input_str = context_str + "<|endoftext|>" + hypoth + "<|endoftext|>" + event
        model_input = self.tokenizer(batch_sents, return_tensors="pt", padding=True).to(self.device)
        result = self.model(input_ids=model_input["input_ids"], attention_mask=model_input["attention_mask"], return_dict=True)
        return torch.sigmoid(result.logits)



class KnowledgeTransferState():
    def __init__(self, corpus_sent, comet, mpnet):
        self.state = []  # (knowledge sentence, sentence score, next central chunk)
        self.utter = corpus_sent["utterance"]
        self.utter_verb_chunks = corpus_sent["utter_verb_chunks"]
        self.K0_tokens = [token[0] for token in corpus_sent["K0"]] 
        self.dialog_context = []
        self.event_mpnet_score = dict()
        self.central_chunk = None
        self.possible_cell_sents_ids = []
        self.EVENT_FLAG = False
        self.DIALOG_FLAG = False
        self.possible_cell_centrals = []
        self.utter_visited_chunks = []

    
    def cos_sim(self, a, b):
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


    def DDP(self, kernel_matrix, max_length=5, epsilon=0.001):     
        item_size = kernel_matrix.shape[0]
        cis = torch.zeros((max_length, item_size)).to(device)
        di2s = torch.diag(kernel_matrix)
        selected_items = list()
        selected_item = torch.argmax(di2s)
        selected_items.append(selected_item)
        while len(selected_items) < max_length:
            k = len(selected_items) - 1
            ci_optimal = cis[:k, selected_item]
            di_optimal = torch.sqrt(di2s[selected_item])
            elements = kernel_matrix[selected_item, :]
            eis = (elements - torch.matmul(ci_optimal, cis[:k, :])) / di_optimal
            cis[k, :] = eis
            di2s -= torch.square(eis)
            selected_item = torch.argmax(di2s)
            if di2s[selected_item] < epsilon:
                break
            selected_items.append(selected_item)
        return selected_items


    def getPossibleEvents(self, comet, attn_context):
        # get possible event knowledges
        rels_transation = get_rels_transation()
        possible_rels = list(rels_transation.keys())

        comet_events = []
        comet_phrases = []
        comet_sents = []
        full_comet_phrases = []

        
        # for verb_chunk in context_events:  
        comet.model.zero_grad()
        queries = []
        
        head = attn_context
        for rel in possible_rels:
            # rel = "xEffect"
            query = "{} {} [GEN]".format(head, rel)
            queries.append(query)
        results = comet.generate(queries, decode_method="beam", num_generate=5)

        for query, result in zip(queries, results):
            for comet_result in result:
                if comet_result == " none":
                    continue
                elif comet_result in comet_phrases:
                    continue
                else:
                    full_comet_phrases.append(comet_result)
                    comet_result_ = comet_result.replace("PersonX ", "")
                    for chunk in dialog_visited:
                        comet_result_ = comet_result_.replace(chunk, "")
                    comet_phrases.append(comet_result_)
                    rel = query.split(" ")[-2]         # -1 is [GEN]
                    rel_text = rels_transation[rel]
                    # event_knowledge = get_rels_transation(head, comet_result)[rel_text]
                    event_knowledge = head + ". "+ rel_text +comet_result 
                    comet_events.append((head, rel_text))
                    comet_sents.append(rel_text + comet_result)
                    

        ##### ------------------- DPP -------------------- ##### 
        # relevance between dialogue history events and comet_phrase: 
        #              dse(context, phrase) - mpnet(context, phrase)

        
        # dialogue semantic coherence = 1/N sum(cosine(verb_chunk, previous_event))         
        comet_sents_embeds = dse.get_average_embedding(comet_phrases)
        context_summ_embeds = dse.get_average_embedding(context_summ.split(" .")) 
        dialog_events_embeds = dse.get_average_embedding(dialog_history) 

        comet_dialog_score = self.cos_sim(comet_sents_embeds, dialog_events_embeds)
        comet_dialog_score = torch.mean(comet_dialog_score, dim=1, keepdim=True)

        comet_context_score = self.cos_sim(comet_sents_embeds, context_summ_embeds)
        comet_context_score = torch.mean(comet_context_score, dim=1, keepdim=True)

        # History coherence
        # comet_history_score = context_gpt2.cal_ppl(comet_sents, context_summ)
        comet_history_score = self.cos_sim(comet_sents_embeds, dialog_history_embeds)
        comet_history_score = torch.mean(comet_history_score, dim=1, keepdim=True)

        # Dialogue central chunks Relevance: Event should focus on central words but no person.
        comet_embeds = mpnet.encode(comet_phrases, convert_to_tensor=True) 
        compared_chunks_score = self.cos_sim(comet_embeds, compared_chunks_embeds)
        compared_chunks_score = torch.mean(compared_chunks_score, dim=1, keepdim=True)


        # Not repeat
        tokenized_corpus = [doc.split(" ") for doc in comet_phrases]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = self.utter.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_scores = torch.tensor(bm25_scores).unsqueeze(1).to(device)

        verb_chunk_embeds = mpnet.encode(self.utter_verb_chunks, convert_to_tensor=True) 
        
        comet_verb_score = self.cos_sim(comet_embeds, verb_chunk_embeds)
        comet_verb_score = torch.mean(comet_verb_score, dim=1, keepdim=True)
        
        # previous cuasal event prediction
        if len(dialog_comet_phrases) > 0:
            dialog_comet_embeds = mpnet.encode(dialog_comet_phrases, convert_to_tensor=True)
            dialog_comet_score = self.cos_sim(comet_embeds, dialog_comet_embeds)
            dialog_comet_score = torch.mean(dialog_comet_score, dim=1, keepdim=True)
            
            rank_score = torch.exp(torch.sub(torch.add(torch.add(comet_history_score, compared_chunks_score), dialog_comet_score), bm25_scores))

        else:
            rank_score = torch.exp(torch.sub(torch.add(comet_history_score, compared_chunks_score), bm25_scores))

        
        # phrases similarity
        item_count = len(comet_phrases)
        sim_matrix = torch.matmul(comet_embeds, comet_embeds.T)  
        kernel_matrix = rank_score.reshape((item_count, 1)) * sim_matrix * rank_score.reshape((1, item_count))

        # DPP process - Top K Comet Phrases (Diversity)
        selected_items = self.DDP(kernel_matrix, max_length=5)
        selected_items = torch.stack(selected_items, dim=0).cpu().numpy()

        selected_phrases = list(itemgetter(*selected_items)(full_comet_phrases))
        selected_sents = list(itemgetter(*selected_items)(comet_sents))
        
        return selected_phrases, selected_sents

    def getEventSents(self):
        possible_cell_centrals = []        
        all_cluster_centrals = []
        event_sents_rank = []
        possible_cell_sents = []
        possible_cell_sent_ids = []

        event_embed = mpnet.encode(event_phrase_)
        event_sents_rank += util.semantic_search(event_embed, sents_embeds, top_k=10)[0]

        for hit in event_sents_rank:
            id = hit["corpus_id"]
            if id not in possible_cell_sent_ids:
                central_chunk = order_central_chunks[id]
                possible_cell_centrals.append(central_chunk)
                possible_cell_sents.append(corpus_sentences[id])
                possible_cell_sent_ids.append(id)

        return possible_cell_centrals, possible_cell_sents, possible_cell_sent_ids

    
    
    
    def getEventCellCentrals(self, entity_connection, entity_possibles):
        possible_cell_centrals = []        
        all_cluster_centrals = []
        event_sents_rank = []
        possible_cell_sents = []
        possible_cell_sent_ids = []
        topK_cluster_centrals = []

        # event-related cell chunks
        for entity in entity_connection:
            entity_event_embed = mpnet.encode(entity + " and" + event_phrase_)
            event_sents_rank += util.semantic_search(entity_event_embed, sents_embeds, top_k=10)[0]

        attn_context_event_embeds = mpnet.encode(attn_context + event_sent)
        event_sents_rank += util.semantic_search(attn_context_event_embeds, sents_embeds, top_k=10)[0]


        for hit in event_sents_rank:
            id = hit["corpus_id"]
            if id not in possible_cell_sent_ids:
                central_chunk = order_central_chunks[id]
                possible_cell_centrals.append(central_chunk)
                possible_cell_sents.append(corpus_sentences[id])
                possible_cell_sent_ids.append(id)

        possible_cell_centrals_embeds = mpnet.encode(possible_cell_centrals + entity_connection)
        entity_connection_embed = np.stack((mpnet.encode(event_phrase), connection_embed))
        
        # get cluster chunks 
        for central in entity_connection:
            if central in cluster_chunks:
                all_cluster_centrals += cluster_chunks[central]

            elif central in split_chunks:
                all_cluster_centrals += split_chunks[central]

        all_cluster_centrals = list(set(all_cluster_centrals))
        all_central_ids = [central_chunks.index(central) for central in all_cluster_centrals]

        if len(all_central_ids) > 0:
            all_central_embeds = np.take(central_chunk_embeds, all_central_ids, axis=0)

            sim_score = self.cos_sim(all_central_embeds, entity_connection_embed)
            avg_sim_score = torch.mean(sim_score, dim=1)
            topK_cell_ids = torch.argsort(avg_sim_score, dim=0, descending=True)[:10].tolist()
            topK_cluster_centrals = list(itemgetter(*topK_cell_ids)(all_cluster_centrals))

        else:
            topK_cluster_centrals = possible_cell_centrals

        return possible_cell_centrals, topK_cluster_centrals, possible_cell_sents, possible_cell_sent_ids



    def getCellCentrals(self, attn_context_embed):
        possible_cell_centrals = []        
        possible_cell_sents = []
        possible_cell_sents_ids = []
        all_cluster_centrals = []
        topK_cluster_centrals = []

        # context-related cell chunks
        context_sents_rank = util.semantic_search(attn_context_embed, sents_embeds, top_k=10)[0]

        for hit in context_sents_rank:
            id = hit["corpus_id"]
            possible_cell_sents_ids.append(id)
            central_chunk = order_central_chunks[id]
            chunks = GKB_chunks[id]
            GKBs = [chunk for chunk in chunks if chunk not in stop_words]
            if " ".join(GKBs[0:2]) == central_chunk:
                possible_cell_centrals += GKBs[2:]
            possible_cell_centrals.append(central_chunk)
            possible_cell_sents.append(corpus_sentences[id])


        # event-related cell chunks
        event_sent_embed = mpnet.encode(event_sent)
        event_sents_rank = util.semantic_search(event_sent_embed, sents_embeds, top_k=K)[0]

        for hit in event_sents_rank:
            id = hit["corpus_id"]
            chunk = GKB_chunks[id]
            possible_cell_centrals += chunks
            possible_cell_sents.append(corpus_sentences[id])


        # In-dialogue cell chunks
        possible_cell_centrals = list(set(possible_cell_centrals) | set(dialog_visited))
        possible_cell_centrals_embeds = mpnet.encode(possible_cell_centrals)

        # get cluster chunks 
        for central in list(set(possible_cell_centrals + self.K0_tokens)):
            if central in cluster_chunks:
                all_cluster_centrals += cluster_chunks[central]

            elif central in split_chunks:
                all_cluster_centrals += split_chunks[central]

        all_cluster_centrals = list(set(all_cluster_centrals))
        all_central_ids = [central_chunks.index(central) for central in all_cluster_centrals]
        # all_central_ids = torch.tensor(all_central_ids).tolist()
        if len(all_central_ids) > 0:
            # all_central_embeds = np.array(itemgetter(*all_central_ids)(central_chunk_embeds))
            all_central_embeds = np.take(central_chunk_embeds, all_central_ids, axis=0)
            # all_central_embeds = torch.index_select(central_chunk_embeds, dim=0, index=all_central_ids)

            sim_score = self.cos_sim(all_central_embeds, possible_cell_centrals_embeds)
            avg_sim_score = torch.mean(sim_score, dim=1)
            topK_cell_ids = torch.argsort(avg_sim_score, dim=0, descending=True)[:10].tolist()
            topK_cluster_centrals = list(itemgetter(*topK_cell_ids)(all_cluster_centrals))

        return possible_cell_centrals, topK_cluster_centrals, possible_cell_sents, possible_cell_sents_ids


    
    def getCellSents(self, cell_centrals):
        cell_sents_ids = []
        for chunk in cell_centrals:
            if chunk in invert_table:
                cell_sent_ids = invert_table[chunk]
                id_list = [*range(cell_sent_ids[0], cell_sent_ids[1], 1)] + cell_sent_ids[2:]
                cell_sents_ids += id_list        
        return list(set(cell_sents_ids))

    
    def pad_chunks(self, sequences):
        sequences_ = []
        for seq in sequences:
            seq_tensor = torch.tensor(seq)
            if len(seq_tensor.shape) == 1:
                seq_tensor = seq_tensor.unsqueeze(0)
            sequences_.append(seq_tensor)
        max_len = max([s.shape[0] for s in sequences_])
        out_tensors = []
        for seq in sequences_:
            if seq.shape[0] <= max_len:            
                tensor = torch.cat([seq, torch.tensor(np.zeros(shape=(max_len-seq.shape[0], 768)))], dim=0) 
                out_tensors.append(tensor)
        out_tensors = torch.stack(out_tensors, dim=0)
        return out_tensors


    def getNextChunk(self, sent_id, possible_cell_centrals_embeds, utter_visited, pos_central):
        FIND_FLAG = False
        next_central = None

        GKBs = GKB_chunks[sent_id]
        pos_next_chunks = [chunk for chunk in GKBs if chunk not in stop_words]

        central_chunk = order_central_chunks[sent_id]
        if " ".join(pos_next_chunks[0:2]) == central_chunk:
            pos_next_chunks = pos_next_chunks[2:]  
        
        next_chunks = list(set(pos_next_chunks + [central_chunk]))
        next_chunks_embeds = mpnet.encode(next_chunks)


        # next_chunks_embeds = mpnet.encode(next_chunks, output_value="token_embeddings")
        # next_chunks_embeds = torch.stack(next_chunks_embeds, dim=0)

        next_chunks_score = self.cos_sim(next_chunks_embeds, possible_cell_centrals_embeds)
        next_chunks_score = torch.mean(next_chunks_score, dim=1).tolist()

        if pos_central:
            candidate_chunks = [pos_central]
        else:
            candidate_chunks = []

            
        for search_index in range(len(next_chunks)):
            try:
                next_id = np.argsort(next_chunks_score)[-1-search_index]
                next_central = next_chunks[next_id]
            except IndexError:
                FIND_FLAG = False
                break
            if next_central in candidate_chunks:
                search_index += 1
                FIND_FLAG = False

            else:
                FIND_FLAG = True
                break

        return FIND_FLAG, next_central

    def cal_distinct_score(self, sentence):
        tokens = sentence.split(" ")
        unique_tokens = set(tokens)
        num_unique_tokens = len(unique_tokens)
        num_total_tokens = len(tokens)
        score = num_unique_tokens / num_total_tokens
        return score


    def getEventSearches(self):
        event_searches = []
        for chunk in entity_connection:
            event_searches.append(chunk + " and" + event_phrase)
        event_search_embeds = mpnet.encode(event_searches)
        return event_search_embeds


    def getInitialEventActions(self):
        topK_actions = []
        added_sent_ids = []
        possible_cell_sents_embeds = mpnet.encode(possible_cell_sents)
        # event_context_embed = np.stack((event_embed, context_summ_embed))
        scores = self.cos_sim(possible_cell_sents_embeds, event_embed)
        scores = torch.mean(scores, dim=1).tolist()
        
        for sent, sent_id, score in zip(possible_cell_sents, possible_cell_sent_ids, scores):
            if sent_id not in utter_visited_sents and sent_id not in added_sent_ids:
                added_sent_ids.append(sent_id)
                sent_central = order_central_chunks[sent_id]
                knowledge_sent = corpus_sentences[sent_id]
                NEXT_FIND_FLAG, next_central = self.getNextChunk(sent_id, self.possible_cell_centrals_embeds, utter_visited, pos_central=sent_central)
                add_action = " [SEP] ".join((knowledge_sent, str(1.0), sent_central, next_central, str(sent_id)))        
                topK_actions.append(add_action) 
        
        return topK_actions, added_sent_ids


    def getInitialEntityActions(self):
        topK_actions = []
        added_sent_ids = []
        possible_cell_sents_embeds = mpnet.encode(entity_possible_cell_sents)
        event_context_embed = np.stack((attn_context_embeds, utter_embed))
        
        scores = self.cos_sim(possible_cell_sents_embeds, event_context_embed)
        scores = torch.mean(scores, dim=1).tolist()
        
        for sent, sent_id, score in zip(entity_possible_cell_sents, entity_possible_cell_sents_ids, scores):
            added_sent_ids.append(sent_id)
            sent_central = order_central_chunks[sent_id]
            knowledge_sent = corpus_sentences[sent_id]
            # NEXT_FIND_FLAG, next_central = self.getNextChunk(sent_id, self.possible_cell_centrals_embeds, utter_visited, pos_central=sent_central)
            CENTRAL_FIND_FLAG, pos_central_chunk = self.getNextChunk(sent_id, self.possible_cell_centrals_embeds, utter_visited, pos_central=None)
            NEXT_FIND_FLAG, next_central = self.getNextChunk(sent_id, self.possible_cell_centrals_embeds, utter_visited, pos_central_chunk)
            if CENTRAL_FIND_FLAG and NEXT_FIND_FLAG:          
                if sent_id not in utter_visited_sents:
                    add_action = " [SEP] ".join((knowledge_sent, str(score), sent_central, next_central, str(sent_id)))
                    topK_actions.append(add_action) 
        
        return topK_actions, added_sent_ids



    def getPossibleActions(self):

        topK_actions = []

        next_topK_cluster_centrals = []
        
        utter_embed = mpnet.encode(self.utter)

        possible_centrals = self.possible_cell_centrals

        if self.EVENT_FLAG:
            possible_centrals = list(set(possible_centrals) | set(event_pos_centrals))
            self.possible_cell_sents_ids = list(set(self.getCellSents(possible_centrals)) | set(utter_event_sents_ids))

            if len(self.state) > 0:
                previous_knowledge = self.state[-1]
                self.pre_knowledge_sent, score, previous_central, self.central_chunk, sent_id = previous_knowledge.split(" [SEP] ")
                utter_visited.append(previous_central)
                utter_visited_sents.append(int(sent_id))
                utter_visited_chunks.extend([previous_central, self.central_chunk])
                

            for add_sent_id, add_action in zip(added_sent_ids, previous_event_topk_actions):
                if add_sent_id not in utter_visited_sents:
                    topK_actions.append(add_action)
            return topK_actions
        else:
            event_context_embed = np.stack((attn_context_embeds, utter_embed))
        
        
        if len(self.state) > 0:
            previous_knowledge = self.state[-1]
            self.pre_knowledge_sent, score, previous_central, self.central_chunk, sent_id = previous_knowledge.split(" [SEP] ")
            utter_visited.append(previous_central)
            utter_visited_sents.append(int(sent_id))
            utter_visited_chunks.extend([previous_central, self.central_chunk])
            dialog_visited_chunks.extend(utter_visited_chunks)


            # get cluster chunks

            if self.central_chunk in cluster_chunks:

                next_cluster_centrals = cluster_chunks[self.central_chunk]

                next_cluster_centrals = [chunk for chunk in next_cluster_centrals if chunk not in utter_visited]
            
                if len(next_cluster_centrals) > 0:

                    next_central_ids = [central_chunks.index(central) for central in next_cluster_centrals]
                    # next_central_ids = torch.tensor(next_central_ids).tolist()
                    next_central_embeds = np.array(itemgetter(*next_central_ids)(central_chunk_embeds))

                    sim_score = self.cos_sim(next_central_embeds, event_context_embed)
                    avg_sim_score = torch.mean(sim_score, dim=1)
                    next_topK_cell_ids = torch.argsort(avg_sim_score, dim=0, descending=True)[:10].tolist()
                    # next_topK_cluster_centrals = list(itemgetter(*next_topK_cell_ids)(next_cluster_centrals))
                    next_topK_cluster_centrals = np.take(next_cluster_centrals, next_topK_cell_ids, axis=0)
                    possible_centrals = np.append(next_topK_cluster_centrals, self.central_chunk)
                    # self.possible_cell_centrals_embeds = mpnet.encode(next_topK_cluster_centrals)


            if len(next_cluster_centrals) == 0 or self.central_chunk not in cluster_chunks:
                next_central_embed = mpnet.encode(self.central_chunk)
                possible_next_central = util.semantic_search(next_central_embed, self.possible_cell_centrals_embeds, top_k=1)[0]
                possible_next_central = self.possible_cell_centrals[possible_next_central[0]["corpus_id"]]
                possible_centrals = list(set([self.central_chunk, possible_next_central]))


            
            
            
            if self.DIALOG_FLAG:
                self.possible_cell_sents_ids = self.getCellSents(list(set(possible_centrals) | set(dialog_pos_centrals)))

            else:
                dialog_visited_chunks.extend(utter_visited_chunks)
                possible_centrals = list(set(possible_centrals) | set(pos_centrals))
                self.possible_cell_sents_ids = list(set(self.getCellSents(possible_centrals)) | set(entity_possible_cell_sents_ids))


            # select top sentences on context, event, previous knowledge.  
            previous_knowledge_embeds = np.take(sents_embeds, utter_visited_sents, axis=0)
            event_context_embed = np.stack((event_embed, context_summ_embed))
            event_context_embed = np.concatenate((event_context_embed, previous_knowledge_embeds), axis=0)
            
            
        # select top sentences on context, event, utterance.    
        event_context_embed = np.stack((event_embed, context_summ_embed, utter_embed))

        possible_sents_embed = np.take(sents_embeds, self.possible_cell_sents_ids, axis=0)

        possible_event_score = self.cos_sim(possible_sents_embed, event_context_embed)
        possible_event_score = torch.sum(possible_event_score, dim=1)

        if self.EVENT_FLAG:
            topk = 50 if len(self.possible_cell_sents_ids)>50 else len(self.possible_cell_sents_ids)
        else:
            topk = 20 if len(self.possible_cell_sents_ids)>20 else len(self.possible_cell_sents_ids)

        assert topk > 0
        
        top_possible_scores, top_possible_ids = torch.topk(possible_event_score, k=topk, dim=0)

        top_possible_ids = top_possible_ids.squeeze().tolist()
        top_possible_scores = top_possible_scores.squeeze().tolist()

        top_possible_sent_ids = list(itemgetter(*top_possible_ids)(self.possible_cell_sents_ids))
        self.event_mpnet_score = dict(zip(top_possible_sent_ids,top_possible_scores))

        # Determine central chunk and next chunk

        if self.EVENT_FLAG:

            for index, (sent_id, score) in enumerate(self.event_mpnet_score.items()):
                sent_central = order_central_chunks[sent_id]
                # CENTRAL_FIND_FLAG, pos_central_chunk = self.getNextChunk(sent_id, self.possible_cell_centrals_embeds, pos_central=None)
                NEXT_FIND_FLAG, next_central = self.getNextChunk(sent_id, self.possible_cell_centrals_embeds, utter_visited, pos_central=sent_central)
                if sent_id not in utter_visited_sents and sent_id not in added_sent_ids and next_central != ".":
                    central_chunk = order_central_chunks[sent_id]
                    add_action = " [SEP] ".join((corpus_sentences[sent_id], str(score), sent_central, next_central, str(sent_id)))
                    topK_actions.append(add_action)   

            for add_sent_id, add_action in zip(added_sent_ids, previous_event_topk_actions):
                if add_sent_id not in utter_visited_sents:
                    topK_actions.append(add_action)

            return topK_actions

        elif self.DIALOG_FLAG:
            
            for index, (sent_id, score) in enumerate(self.event_mpnet_score.items()):
                sent_central = order_central_chunks[sent_id]
                # CENTRAL_FIND_FLAG, pos_central_chunk = self.getNextChunk(sent_id, self.possible_cell_centrals_embeds, pos_central=None)
                NEXT_FIND_FLAG, next_central = self.getNextChunk(sent_id, self.possible_cell_centrals_embeds, utter_visited, pos_central=sent_central)
                if sent_id not in utter_visited_sents and next_central != ".":
                    add_action = " [SEP] ".join((corpus_sentences[sent_id], str(score), sent_central, next_central, str(sent_id)))
                    topK_actions.append(add_action)  
            
            return topK_actions

        else:
        # K largest top knowledge sentences
            for index, (sent_id, score) in enumerate(self.event_mpnet_score.items()):
                sent_central = order_central_chunks[sent_id]

                CENTRAL_FIND_FLAG, pos_central_chunk = self.getNextChunk(sent_id, self.possible_cell_centrals_embeds, utter_visited, pos_central=None)
                NEXT_FIND_FLAG, next_central = self.getNextChunk(sent_id, self.possible_cell_centrals_embeds, utter_visited, pos_central_chunk)
                if CENTRAL_FIND_FLAG and NEXT_FIND_FLAG:
                    if sent_id not in utter_visited_sents and sent_id not in entity_added_sent_ids and next_central != ".":
                        add_action = " [SEP] ".join((corpus_sentences[sent_id], str(score), pos_central_chunk, next_central, str(sent_id))) 
                        topK_actions.append(add_action) 
    
                    if pos_central_chunk in possible_centrals:
                        add_action = " [SEP] ".join((corpus_sentences[sent_id], str(score), pos_central_chunk, next_central, str(sent_id)))
                        topK_actions.append(add_action)  
                    elif next_central in possible_centrals:
                        add_action = " [SEP] ".join((corpus_sentences[sent_id], str(score), next_central, pos_central_chunk, str(sent_id)))
                        topK_actions.append(add_action)        

            if len(topK_actions) == 0:
                topK_actions = entity_topK_actions

            return topK_actions



    def takeAction(self, action):
        newState = deepcopy(self)
        newState.state.append(action)
        return newState
    

    def isTerminal(self):
        terminalFlag = False
        if self.EVENT_FLAG:
            return False

        if self.DIALOG_FLAG:
            return False

        if set(terminal_K0).issubset(set(dialog_visited_chunks)):
            return True

        state_chunks = []
        for state in self.state:
            state_chunks.extend([state.split(" [SEP] ")[2], state.split(" [SEP] ")[3]])
        if set(terminal_K0).issubset(set(state_chunks)):
            return True

        elif len(self.state) == 3:
            return True
        else:
            return False
    

    def getReward(self):
        reward = 0
        # sentence level score
        #    context similarity - mpnet cosine - done in getActions()
        #    Dialogue history coherence - DSE  - done
        #    informative - Distinct - done
        #    context metric - MoverScore - done
        #    Comet Event 

        # dialogue coherence score
        knowledge_sents = [knowledge_state.split(" [SEP] ")[0] for knowledge_state in self.state]
        knowledge_sents_embeds = dse.get_average_embedding(knowledge_sents)
        
        dialog_coherence_score = self.cos_sim(knowledge_sents_embeds, dialog_history_embeds)
        dialog_coherence_score = torch.sum(torch.mean(dialog_coherence_score, dim=1), dim=0).item()

        # attn context score
        attn_context_score = self.cos_sim(knowledge_sents_embeds, dse_attn_context_embeds) 
        attn_context_score = torch.sum(torch.mean(attn_context_score, dim=1), dim=0).item()

        # utter coherence score
        utter_coherence_score = self.cos_sim(knowledge_sents_embeds, dse_utter_embed) 
        utter_coherence_score = torch.sum(torch.mean(utter_coherence_score, dim=1), dim=0).item()

        # not repeat
        if len(knowledge_sents)>1:
            prev_sents = knowledge_sents[0:-1]
            prev_sent_embed = mpnet.encode(prev_sents)
            current_sent_embed = mpnet.encode(knowledge_sents[-1])
            prev_sim_score = self.cos_sim(prev_sent_embed, current_sent_embed)
            prev_sim_score = torch.max(torch.mean(prev_sim_score, dim=1), dim=0)[0].item()

        else:
            prev_sim_score = 0


        length_score = 0
        distinct_score = 0
        wasserstein_score = 0
        neighbor_score = 0

        prev_sents = []
        
        for knowledge_state in self.state:
            knowledge_sent, sent_score, current_central, next_central, sent_id = knowledge_state.split(" [SEP] ")

            # context mover score - Wasserstein distance from context distribution and selected knowledge sentence distribution
            knowledge_sent_embed = sents_embeds[int(sent_id)]
            wasserstein_score += wasserstein_distance(context_summ_embed, knowledge_sent_embed)

            # distinct score
            distinct_score += self.cal_distinct_score(knowledge_sent)

            # connective score
            utter_visited_centrals = [knowledge_state.split(" [SEP] ")[2] for knowledge_state in self.state]
            utter_next_centrals = [knowledge_state.split(" [SEP] ")[3] for knowledge_state in self.state]
            utter_central_embeds = mpnet.encode(list(set(utter_visited_centrals + utter_next_centrals)), convert_to_tensor=True)
            compare_score = self.cos_sim(utter_central_embeds, compared_chunks_embeds)
            connective_score = torch.mean(torch.mean(compare_score, dim=1), dim=0).item()

            # length score
            length_score += len(knowledge_sent.split(" ")) / 10

            # neighbor score       
            if not self.EVENT_FLAG and not self.DIALOG_FLAG:         

                if current_central in terminal_K0:
                    neighbor_score += 3.0

                if next_central in terminal_K0:
                    neighbor_score += 3.0

                if current_central in self.possible_cell_centrals: 
                    neighbor_score += 1.0
                if next_central in self.possible_cell_centrals:
                    neighbor_score += 1.0

                if current_central in dialog_visited:
                    neighbor_score += 1.0

                if next_central in dialog_visited:
                    neighbor_score += 1.0



        
        if self.DIALOG_FLAG:
            
            # reward = dialog_coherence_score + attn_context_score + utter_coherence_score + 0.5*distinct_score + length_score - 1000*wasserstein_score - 2*no_repeat_score
            reward = dialog_coherence_score + attn_context_score + utter_coherence_score + 0.5*distinct_score + length_score - 1000*wasserstein_score - prev_sim_score

        elif self.EVENT_FLAG:
            # comet event coherence score
            comet_coherence = entail_classifier(knowledge_sents, [event_sent])
            comet_coherence_score = [coherence["scores"][0] for coherence in comet_coherence]
            comet_coherence_score = np.sum(comet_coherence_score, axis=0)

            comet_sent_embed = dse.get_average_embedding(event_sent)
            dse_comet_coherence_score = self.cos_sim(knowledge_sents_embeds, comet_sent_embed)
            dse_comet_coherence_score = torch.sum(dse_comet_coherence_score, dim=0).item() 

            
            reward =  dialog_coherence_score + comet_coherence_score + dse_comet_coherence_score + 0.5*distinct_score + length_score - prev_sim_score
        
        else:     
            
            reward = connective_score + neighbor_score + distinct_score - 1000*wasserstein_score
        
        return reward



if __name__=="__main__":

    # Data Loading
    start_time = time.time()
    DailyDialog_corpus = load_dialog()
    corpus_sentences, sents_embeds, central_chunk_embeds = load_embeddings()
    # sents_embeds, chunk_embeds, central_chunk_embeds = load_embeddings()
    # corpus_sentences, corpus_GKB_chunks, voronoi_cell = load_voronoi()
    order_central_chunks = load_central_chunks()
    invert_table = load_invert_table()
    cluster_chunks = load_cluster()
    split_chunks = load_split_chunks()
    GKB_chunks= load_GKB_chunks()

    device = "cuda" if torch.cuda.is_available() else "cpu"


    central_chunks = list(invert_table.keys())
    
    data_loaded = time.time()
    data_loading_time = data_loaded-start_time
    print("\n" + "Load Corpus, Strored Embeddings, Invert Table in %.2fs." % data_loading_time)


    # Pretrained Models Loading
    dse = DSE("dse-roberta-base")
    comet = Comet("comet-atomic_2020_BART")
    mpnet = SentenceTransformer("all-mpnet-base-v2")
    dialogsum = DialogSum("bart-large-cnn-samsum")        
    # context_gpt2 = ContextGPT2("gpt2")
    entail_classifier = pipeline("zero-shot-classification", model='bart-large-mnli', device=0)
    model_loaded = time.time()
    model_loading_time = model_loaded-data_loaded
    print("\n" + "Load Pretrained Models, Probase Concepts in %.2fs." % model_loading_time)


    DailyDialog_corpus = DailyDialog_corpus[args.START:args.END]


    with open(retrieved_knowledge_path, "w", encoding="utf8") as f_out:
        for dialog_id, dialog in enumerate(DailyDialog_corpus):
            f_out.write(str(dialog_id+args.START+1)+"\n")

            print("DIALOG ID:", dialog_id+args.START+1, "\n")

            dialog_visited = []
            dialog_visited_chunks = []
            dialog_events = []       
            dialog_history = []
            dialog_visited_sents = []
            previous_comets = []
            dialog_context = ""
            dialog_comet_sents = []
            dialog_comet_phrases = []
            dialog_entity_connection = []

            dialog_start_time = time.time()

            for utter_id, corpus_sent in enumerate(dialog):

                print("*"*50, "utter", utter_id, "*"*50)

                utterance = corpus_sent["utterance"]

                if utter_id == len(dialog)-1:
                    break
                elif (utter_id % 2) == 0:
                    utterance = "PersonX: " + utterance
                else:
                    utterance = "PersonY: " + utterance

                f_out.write(str(utter_id+1)+"\n")

                dialog_context = dialog_context + utterance + "\n"
                dialog_history.append(corpus_sent["utterance"])   

                dialog_history_embeds = dse.get_average_embedding(dialog_history)     
                
                print("\n" + utterance)
                utter_start_time = time.time()

                initialState = KnowledgeTransferState(corpus_sent, comet, mpnet)
                utter_embed = mpnet.encode(initialState.utter)
                dse_utter_embed = dse.get_average_embedding(initialState.utter)
                
                terminal_K0 = [token for token in initialState.K0_tokens if token in central_chunks]
                dialog_visited.extend(terminal_K0)
                compared_chunks = dialog_visited
                if len(compared_chunks) > 0:
                    compared_chunks_embeds = mpnet.encode(compared_chunks, convert_to_tensor=True)
                else:
                    compared_chunks_embeds = torch.from_numpy(utter_embed).to(device)

                dialog_events += initialState.utter_verb_chunks
                
                if len(initialState.state) > 0:

                    initialState.previous_knowledge = initialState.state[-1]
                    initialState.pre_knowledge_sent, score, previous_central, initialState.central_chunk, sent_id = initialState.previous_knowledge.split(" [SEP] ")

                
                # Possiblle cell centrals
                context_summ = dialogsum.generate(dialog_context)[0]
                context_summ_embed = mpnet.encode(context_summ)
                print("-"*50)

                # select most similar sentence from the context for event prediction
                context_summ = context_summ.strip()
                context_events = context_summ.split(". ")
                context_events = [event for event in context_events if event != ""]
                print(context_events)
                f_out.write("Context Summary: " + "    ".join(context_events) + "\n")
                context_events_embed = mpnet.encode(context_events)
                
                if len(context_events) > 1:
                    context_utter_sim = initialState.cos_sim(context_events_embed, utter_embed).squeeze()
                    argmax_context_id = torch.argmax(context_utter_sim, dim=0).item()
                    attn_context = context_events[argmax_context_id]
                elif len(context_summ) > 0:
                    attn_context = context_summ
                else:
                    attn_context = dialog_context
                
                attn_context_embeds = mpnet.encode(attn_context)
                dse_attn_context_embeds = dse.get_average_embedding(attn_context)

                print("-"*50)
                print("Most Simillar Sentence:", attn_context)

                utter_visited = []
                utter_visited_chunks = []
                utter_visited_sents = []

                # Possible sents 
                initialState.possible_cell_centrals, topK_cluster_centrals, entity_possible_cell_sents, entity_possible_cell_sents_ids = initialState.getCellCentrals(attn_context_embeds)
                initialState.possible_cell_centrals_embeds = mpnet.encode(initialState.possible_cell_centrals)

                assert len(initialState.possible_cell_centrals) > 0

                pos_centrals = list(set(initialState.possible_cell_centrals) | set(topK_cluster_centrals))
                
                initialState.possible_cell_sents_ids = initialState.getCellSents(pos_centrals)
                utter_possible_cell_sents_ids = initialState.possible_cell_sents_ids
                entity_possible_cell_centrals = initialState.possible_cell_centrals
                utter_possible_cell_centrals_embeds = initialState.possible_cell_centrals_embeds

                entity_topK_actions, entity_added_sent_ids = initialState.getInitialEntityActions()


                # MCTS based on each Entity
                print("-"*50, "ENTITY CONNECTION", "-"*50)
                entity_connection = []
                entity_connection_sents = []
                if len(terminal_K0)>0:
                    print(terminal_K0)

                    searcher = mcts(iterationLimit=15)
                    initialState.EVENT_FLAG = False
                    action = searcher.search(initialState=initialState)

                    if action:

                        ActionChain = action["ActionChain"]
                        UCTValue = action["UCTValue"]

                        for action in ActionChain:
                            print("      ->", action)
                            print("      UCTValue:", UCTValue)
                            print("-"*50)
                            knowledge_sent, score, central, next_central, sent_id = action.split(" [SEP] ")
                            if central != "people":
                                entity_connection.append(central)
                            if next_central != "people":
                                entity_connection.append(next_central)
                            dialog_entity_connection.extend(entity_connection)
                            entity_connection_sents.append(knowledge_sent)
                        f_out.write("Entity Connection: " + "    ".join(entity_connection_sents) + "\n")

                            

                dialog_entity_connection_embeds = mpnet.encode(dialog_entity_connection)
                
                # Dialogue response
                print("\n", "-"*50, "DIALOGUE RESPONSE", "-"*50)
                dialog_sents = []
                initialState.DIALOG_FLAG = True
                utter_visited = []
                utter_visited_chunks = []
                utter_visited_sents = []

                entity_connection = dialog_entity_connection if len(entity_connection) == 0 else entity_connection
                dialog_pos_centrals = list(set(entity_connection) | set(pos_centrals))

                initialState.possible_cell_sents_ids = initialState.getCellSents(dialog_pos_centrals)  
                utter_dialog_sents_ids = initialState.possible_cell_sents_ids

                searcher = mcts(iterationLimit=3)
                action = searcher.search(initialState=initialState)

                ActionChain = action["ActionChain"]
                UCTValue = action["UCTValue"]

                for action in ActionChain:
                    print("      ->", action)
                    print("      UCTValue:", UCTValue)
                    print("-"*50)
                    knowledge_sent, score, central, next_central, sent_id = action.split(" [SEP] ")
                    dialog_sents.append(knowledge_sent)

                f_out.write("Dialog Response: " + "    ".join(dialog_sents) + "\n")


                # Event Knowledge

                top_comet_phrases, top_comet_sents = initialState.getPossibleEvents(comet, attn_context)
                dialog_comet_sents += top_comet_sents
                dialog_comet_phrases += top_comet_phrases

                for event_phrase, event_sent in zip(top_comet_phrases, top_comet_sents):
                    dialog_event_sents = []

                    print("-"*50)
                    print("      ->", event_sent)  

                    event_phrase_ = event_phrase
                    
                    if "Because PersonX wanted" in event_sent:
                        event_phrase_ = "you wanted" + event_phrase
                    elif "PersonX will then" in event_sent:
                        event_phrase_ = "you will then" + event_phrase
                    elif "PersonX wants" in event_sent:
                        event_phrase_ = "you want" + event_phrase
                    elif "Because" in event_sent:
                        event_phrase_ = event_phrase.replace("PersonX is", "you are").replace("PersonX", "you")
		    

                    event_sent = event_sent.replace("PersonX is", "you are")  
                    event_sent = event_sent.replace("PersonX", "you")  

                    initialState.EVENT_FLAG = True
                    initialState.DIALOG_FLAG = False

                    # Possible sents 
                    entity_connection = dialog_entity_connection if len(entity_connection) == 0 else entity_connection
                    event_embed = mpnet.encode(event_phrase)
                    connection_embed = mpnet.encode(entity_connection)
                    connection_embed = np.mean(connection_embed, axis=0, keepdims=False)
                    if len(entity_connection) > 0:
                        event_possible_cell_centrals, event_topK_cluster_centrals, possible_cell_sents, possible_cell_sent_ids = initialState.getEventCellCentrals(entity_connection, topK_cluster_centrals)
                    
                        initialState.possible_cell_centrals_embeds = np.stack((connection_embed, event_embed))
                        initialState.possible_cell_centrals = list(set(entity_connection) | set(event_topK_cluster_centrals))

                        event_pos_centrals = list(set(entity_connection) | set(event_topK_cluster_centrals))
                    
                        initialState.possible_cell_sents_ids = initialState.getCellSents(event_pos_centrals)  
                        utter_event_sents_ids = initialState.possible_cell_sents_ids

                        # event_compare_embeds = mpnet.encode(event_pos_centrals)
                        # event_context_embed = np.stack((event_embed, connection_embed))

                        previous_event_topk_actions, added_sent_ids = initialState.getInitialEventActions()

                    else:
                        event_possible_cell_centrals, possible_cell_sents, possible_cell_sent_ids = initialState.getEventSents()
                        initialState.possible_cell_centrals_embeds = np.expand_dims(event_embed, axis=0)
                        initialState.possible_cell_centrals = event_possible_cell_centrals

                        event_pos_centrals = event_possible_cell_centrals

                        initialState.possible_cell_sents_ids = initialState.getCellSents(event_possible_cell_centrals) 
                        utter_event_sents_ids = initialState.possible_cell_sents_ids

                        previous_event_topk_actions, added_sent_ids = initialState.getInitialEventActions()


                    
                    searcher = mcts(iterationLimit=3)
                    
                    action = searcher.search(initialState=initialState)

                    ActionChain = action["ActionChain"]
                    UCTValue = action["UCTValue"]

                    for action in ActionChain:
                        print("          ", action)
                        
                        knowledge_sent, score, central, next_central, sent_id = action.split(" [SEP] ")
                        dialog_event_sents.append(knowledge_sent)

                    print("      UCTValue:", UCTValue)
                    print("-"*50)
                    print("\n")

                    f_out.write("Dialog Event:" + event_sent + "\n")

                    f_out.write("Dialog Event Sents:" + "    ".join(dialog_event_sents) + "\n")

                
        
                print("="*100)
                print("\n\n")

                


            print("="*100)
            print("\n\n")
            f_out.write("\n")



