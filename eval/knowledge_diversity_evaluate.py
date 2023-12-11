import sys
import json
import torch
import numpy as np
from itertools import chain
from itertools import combinations
from bert_score import score
from bart_score import BARTScorer
from transformers import pipeline
from lm_scorer.models.auto import AutoLMScorer as LMScorer

from bart_score import BARTScorer
from moverscore_v2 import corpus_score
from rouge_score import rouge_scorer
from torchmetrics.text.rouge import ROUGEScore


device = "cuda" if torch.cuda.is_available()else "cpu"
bart_scorer = BARTScorer(device=device, checkpoint='projects/MCTS_GenericsKB/MCTS_GKB_DailyDialog/EVAL/bart_large_cnn')

START = 0
step=65

chatgpt_commonsense_list_path = "xxx.txt"
retrieve_knowledge_list_path ="xxx.txt"
human_label_list_path = "xxx.txt"
log_path = "xxx.log"


dataset_chatgpt_P, dataset_chatgpt_R, dataset_chatgpt_F1 = [], [], []
dataset_mcts_P, dataset_mcts_R, dataset_mcts_F1 = [], [], []


dataset_chatgpt_R1, dataset_chatgpt_R2, dataset_chatgpt_RL = [], [], []
dataset_mcts_R1, dataset_mcts_R2, dataset_mcts_RL = [], [], []

dataset_mcts_human_event_cover, dataset_chatgpt_human_event_cover = [], []

dataset_mcts_human_event_cover_max, dataset_chatgpt_human_event_cover_max = [], []



dataset_chatgpt_mean_P, dataset_chatgpt_mean_R, dataset_chatgpt_mean_F1 = [], [], []
dataset_chatgpt_max_P, dataset_chatgpt_max_R, dataset_chatgpt_max_F1 = [], [], []

dataset_chatgpt_bart = []
dataset_chatgpt_max_bart = []

dataset_chatgpt_mover = []
dataset_chatgpt_max_mover = []


rouge_score = ROUGEScore()


entail_classifier = pipeline("zero-shot-classification", model='bart-large-mnli', device=0)



class Logger(object):
    def __init__(self, filename=log_path, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "w")

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



def process_events(events):
    event_phrases = []
    
    for event in events:
        if "Because you " in event:
            event_ = event.replace("Because you ", "")
        elif "you will then " in event:
            event_ = event.replace("you will then ", "")
        elif "you wants " in event:
            event_ = event.replace("you wants ", "")
        elif "you needed " in event:
            event_ = event.replace("you needed ", "")
        elif "Because " in event:
            event_ = event.replace("Because ", "")
        event_phrases.append(event_)

    return event_phrases




def main():
    with open(chatgpt_commonsense_list_path, "r", encoding="utf8") as f_in:
        dataset_commonsense = json.load(f_in)

    with open(retrieve_knowledge_list_path, "r", encoding="utf8") as f_in:
        retrieve_knowledge = json.load(f_in)
        # retrieve_knowledge = retrieve_knowledge["context_sents"]

    with open(human_label_list_path, "r", encoding="utf8") as f_in:
        dataset_human_labels = json.load(f_in)
    
    retrieve_knowledge = retrieve_knowledge[0]

    dialog_num = len(retrieve_knowledge["dataset_context_summ"])

    dialog_len_coverage_max = dict()
    dialog_len_coverage_theta = dict()

    for dialog_id in range(dialog_num):
    
        dataset_context_summ = retrieve_knowledge["dataset_context_summ"][dialog_id]
        dataset_events = retrieve_knowledge["dataset_events"][dialog_id]
        dataset_event_sents = retrieve_knowledge["dataset_event_sents"][dialog_id]

        chatgpt_labels = dataset_commonsense[dialog_id+START]
        human_labels = dataset_human_labels[dialog_id+START]

        if len(chatgpt_labels) not in dialog_len_coverage_max.keys():
            dialog_len_coverage_max[len(chatgpt_labels)] = []
            dialog_len_coverage_theta[len(chatgpt_labels)] = []

        dialog_chatgpt_R1, dialog_chatgpt_R2, dialog_chatgpt_RL = [], [], []
        dialog_mcts_R1, dialog_mcts_R2, dialog_mcts_RL = [], [], []

        dialog_chatgpt_P, dialog_chatgpt_R, dialog_chatgpt_F1 = [], [], []
        dialog_mcts_P, dialog_mcts_R, dialog_mcts_F1 = [], [], []


        dialog_mcts_human_event_cover, dialog_chatgpt_human_event_cover = [], []
        dialog_mcts_human_event_cover_max, dialog_chatgpt_human_event_cover_max = [], []

        dialog_chatgpt_mean_P, dialog_chatgpt_mean_R, dialog_chatgpt_mean_F1 = [], [], []
        dialog_chatgpt_max_P, dialog_chatgpt_max_R, dialog_chatgpt_max_F1 = [], [], []

        dialog_chatgpt_bart = []
        dialog_chatgpt_max_bart = []

        dialog_chatgpt_mover = []
        dialog_chatgpt_max_mover = []

    
        for utter_id in range(len(chatgpt_labels)):
            chatgpt_label = chatgpt_labels[utter_id]

            events = dataset_events[utter_id]
            event_sents = dataset_event_sents[utter_id]
            flatten_event_sents = list(chain.from_iterable(event_sents))


            # Event-oriented Diversity
            event_phrases = process_events(events)

            human_event_labels = []
            for human_label in human_labels:
                res = entail_classifier(human_label, events)
                human_event_labels.append([res["labels"][0], res["scores"][0]])
                

            chatgpt_event_labels = []
            for chatgpt_sent in chatgpt_label:
                res = entail_classifier(chatgpt_sent, events)
                chatgpt_event_labels.append([res["labels"][0], res["scores"][0]])

            mcts_retrieves = flatten_event_sents
            mcts_retrieves = list(set(mcts_retrieves))

            mcts_event_labels = []
            for mcts_sent in mcts_retrieves:
                if len(mcts_sent) > 0:
                    res = entail_classifier(mcts_sent, events)
                    mcts_event_labels.append([res["labels"][0], res["scores"][0]])

            # mcts-human, chatgpt-human event coverage
            # set Threshold = 0.5

            human_events = []
            all_human_events = []
            for human_item in human_event_labels:
                human_event, human_score = human_item
                all_human_events.append(human_event)
                if human_score > 0.5:
                    human_events.append(human_event)

            mcts_events = []
            all_mcts_events = []
            for mcts_item in mcts_event_labels:
                mcts_event, mcts_score = mcts_item
                all_mcts_events.append(mcts_event)
                if mcts_score > 0.5:
                    mcts_events.append(mcts_event)

            chatgpt_events = []
            all_chatgpt_events = []
            for chatgpt_item in chatgpt_event_labels:
                chatgpt_event, chatgpt_score = chatgpt_item
                all_chatgpt_events.append(chatgpt_event)
                if chatgpt_score > 0.5:
                    chatgpt_events.append(chatgpt_event)

            utter_mcts_human_event_cover = len(set(mcts_events) & set(human_events)) / len(set(human_events)) if len(set(human_events))>0 else 0
            utter_chatgpt_human_event_cover = len(set(chatgpt_events) & set(human_events)) / len(set(human_events)) if len(set(human_events))>0 else 0

            utter_mcts_human_event_cover_max = len(set(all_mcts_events) & set(all_human_events)) / len(set(all_human_events)) if len(set(all_human_events))>0 else 0
            utter_chatgpt_human_event_cover_max = len(set(all_chatgpt_events) & set(all_human_events)) / len(set(all_human_events)) if len(set(all_human_events))>0 else 0

            dialog_mcts_human_event_cover.append(utter_mcts_human_event_cover)
            dialog_chatgpt_human_event_cover.append(utter_chatgpt_human_event_cover)

            dialog_mcts_human_event_cover_max.append(utter_mcts_human_event_cover_max)
            dialog_chatgpt_human_event_cover_max.append(utter_chatgpt_human_event_cover_max)


            # Inner Sentences Diversity

            mcts_combine = list(combinations(flatten_event_sents, 2))

            chatgpt_combine = list(combinations(chatgpt_label, 2))

            chatgpt_rouge_preds = [i[0] for i in chatgpt_combine]
            chatgpt_rouge_refs = [i[1] for i in chatgpt_combine]

            chatgpt_rouge_scores = rouge_score(preds=chatgpt_rouge_preds, target=chatgpt_rouge_refs)

            dialog_chatgpt_R1.append(chatgpt_rouge_scores["rouge1_fmeasure"].item())
            dialog_chatgpt_R2.append(chatgpt_rouge_scores["rouge2_fmeasure"].item())
            dialog_chatgpt_RL.append(chatgpt_rouge_scores["rougeL_fmeasure"].item())

            mcts_rouge_preds = [i[0] for i in mcts_combine]
            mcts_rouge_refs = [i[1] for i in mcts_combine]

            # n-gram rouge score -> smaller is better
            mcts_rouge_scores = rouge_score(preds=mcts_rouge_preds, target=mcts_rouge_refs)
            dialog_mcts_R1.append(mcts_rouge_scores["rouge1_fmeasure"].item())
            dialog_mcts_R2.append(mcts_rouge_scores["rouge2_fmeasure"].item())
            dialog_mcts_RL.append(mcts_rouge_scores["rougeL_fmeasure"].item())

            # Inner pairwise Bert Score -> smaller is better
            chatgpt_P, chatgpt_R, chatgpt_F1 = score(chatgpt_rouge_preds, chatgpt_rouge_refs, model_type='roberta-large', verbose=True)
            
            utter_chatgpt_P = torch.mean(chatgpt_P).item()
            utter_chatgpt_R = torch.mean(chatgpt_R).item()
            utter_chatgpt_F1 = torch.mean(chatgpt_F1).item()

            dialog_chatgpt_P.append(utter_chatgpt_P)
            dialog_chatgpt_R.append(utter_chatgpt_R)
            dialog_chatgpt_F1.append(utter_chatgpt_F1)


            mcts_P, mcts_R, mcts_F1 = score(mcts_rouge_preds, mcts_rouge_refs, lang="en", verbose=True)

            utter_mcts_P = torch.mean(mcts_P).item()
            utter_mcts_R = torch.mean(mcts_R).item()
            utter_mcts_F1 = torch.mean(mcts_F1).item()

            dialog_mcts_P.append(utter_mcts_P)
            dialog_mcts_R.append(utter_mcts_R)
            dialog_mcts_F1.append(utter_mcts_F1)


            # Inter Sentences Similarity
            mcts_retrieves = flatten_event_sents

            mcts_retrieves = list(set(mcts_retrieves))

            repeat_chatgpt_labels = [label for label in chatgpt_label for i in range(len(mcts_retrieves))]
            repeat_mcts_retrives = mcts_retrieves * len(chatgpt_label)

            assert len(repeat_chatgpt_labels) == len(repeat_mcts_retrives)

            inter_chatgpt_P, inter_chatgpt_R, inter_chatgpt_F1 = score(repeat_mcts_retrives, repeat_chatgpt_labels, lang="en", verbose=True)

            inter_chatgpt_P = inter_chatgpt_P.reshape((len(mcts_retrieves), len(chatgpt_label)))
            max_chatgpt_P = torch.max(inter_chatgpt_P, dim=0)
            inter_chatgpt_R = inter_chatgpt_R.reshape((len(mcts_retrieves), len(chatgpt_label)))
            max_chatgpt_R = torch.max(inter_chatgpt_R, dim=0)
            inter_chatgpt_F1 = inter_chatgpt_F1.reshape((len(mcts_retrieves), len(chatgpt_label)))
            max_chatgpt_F1 = torch.max(inter_chatgpt_F1, dim=0)

            max_utter_chatgpt_P = torch.mean(max_chatgpt_P[0]).item()
            max_utter_chatgpt_R = torch.mean(max_chatgpt_R[0]).item()
            max_utter_chatgpt_F1 = torch.mean(max_chatgpt_F1[0]).item()

            dialog_chatgpt_max_P.append(max_utter_chatgpt_P)
            dialog_chatgpt_max_R.append(max_utter_chatgpt_R)
            dialog_chatgpt_max_F1.append(max_utter_chatgpt_F1)

            mean_utter_chatgpt_P = torch.mean(inter_chatgpt_P).item()
            mean_utter_chatgpt_R = torch.mean(inter_chatgpt_R).item()
            mean_utter_chatgpt_F1 = torch.mean(inter_chatgpt_F1).item()

            dialog_chatgpt_mean_P.append(mean_utter_chatgpt_P)
            dialog_chatgpt_mean_R.append(mean_utter_chatgpt_R)
            dialog_chatgpt_mean_F1.append(mean_utter_chatgpt_F1)

            chatgpt_bart = bart_scorer.score(repeat_mcts_retrives, repeat_chatgpt_labels)
            chatgpt_bart = np.reshape(chatgpt_bart, (len(mcts_retrieves), len(chatgpt_label)))
            max_chatgpt_bart = np.max(chatgpt_bart, axis=0)

            max_utter_chatgpt_bart = np.mean(max_chatgpt_bart[0])
            dialog_chatgpt_max_bart.append(max_utter_chatgpt_bart)

            utter_chatgpt_bart = np.mean(chatgpt_bart)
            dialog_chatgpt_bart.append(utter_chatgpt_bart)

            chatgpt_mover = corpus_score(repeat_mcts_retrives, [repeat_chatgpt_labels])
            chatgpt_mover = np.reshape(chatgpt_mover, (len(mcts_retrieves), len(chatgpt_label)))
            max_chatgpt_mover = np.max(chatgpt_mover, axis=0)

            max_utter_chatgpt_mover = np.mean(max_chatgpt_mover[0])
            dialog_chatgpt_max_mover.append(max_utter_chatgpt_mover)

            utter_chatgpt_mover = np.mean(chatgpt_mover)
            dialog_chatgpt_mover.append(utter_chatgpt_mover)


        print("-"*100)
        print("Dialog ID:", dialog_id)
        print("-"*100)

        dialog_avg_chatgpt_P = np.mean(dialog_chatgpt_P, keepdims=False)
        dialog_avg_chatgpt_R = np.mean(dialog_chatgpt_R, keepdims=False)
        dialog_avg_chatgpt_F1 = np.mean(dialog_chatgpt_F1, keepdims=False)


        dataset_chatgpt_P.append(dialog_avg_chatgpt_P)
        dataset_chatgpt_R.append(dialog_avg_chatgpt_R)
        dataset_chatgpt_F1.append(dialog_avg_chatgpt_F1)

        print("-"*100)
        print("Dialog ID  |    Chatgpt BERT Score Precision in Sentences   |  Chatgpt BERT Score Recall in Sentences    |    Chatgpt BERT Score F1 in Sentences")
        print("Chatgpt Dialog BERT Score in Sentences:", dialog_id, "  |    ", dialog_avg_chatgpt_P, "    |    ", dialog_avg_chatgpt_R, "    |    ", dialog_avg_chatgpt_F1)
        print("-"*100)

        dialog_avg_mcts_P = np.mean(dialog_mcts_P, keepdims=False)
        dialog_avg_mcts_R = np.mean(dialog_mcts_R, keepdims=False)
        dialog_avg_mcts_F1 = np.mean(dialog_mcts_F1, keepdims=False)


        dataset_mcts_P.append(dialog_avg_mcts_P)
        dataset_mcts_R.append(dialog_avg_mcts_R)
        dataset_mcts_F1.append(dialog_avg_mcts_F1)

        print("-"*100)
        print("Dialog ID  |    MCTS BERT Score Precision in Sentences   |  MCTS BERT Score Recall in Sentences    |    MCTS BERT Score F1 in Sentences")
        print("MCTS Dialog BERT Score in Sentences:", dialog_id, "  |    ", dialog_avg_mcts_P, "    |    ", dialog_avg_mcts_R, "    |    ", dialog_avg_mcts_F1)
        print("-"*100)

        
        dialog_avg_chatgpt_R1 = np.mean(dialog_chatgpt_R1, keepdims=False)
        dialog_avg_chatgpt_R2 = np.mean(dialog_chatgpt_R2, keepdims=False)
        dialog_avg_chatgpt_RL = np.mean(dialog_chatgpt_RL, keepdims=False)


        dataset_chatgpt_R1.append(dialog_avg_chatgpt_R1)
        dataset_chatgpt_R2.append(dialog_avg_chatgpt_R2)
        dataset_chatgpt_RL.append(dialog_avg_chatgpt_RL)

        print("-"*100)
        print("Dialog ID  |    Chatgpt Rouge-1 Score in Sentences   |  Chatgpt Rouge-2 Score in Sentences    |    Chatgpt Rouge-L Score in Sentences")
        print("Chatgpt Dialog Rouge Score in Sentences:", dialog_id, "  |    ", dialog_avg_chatgpt_R1, "    |    ", dialog_avg_chatgpt_R2, "    |    ", dialog_avg_chatgpt_RL)
        print("-"*100)


        dialog_avg_mcts_R1 = np.mean(dialog_mcts_R1, keepdims=False)
        dialog_avg_mcts_R2 = np.mean(dialog_mcts_R2, keepdims=False)
        dialog_avg_mcts_RL = np.mean(dialog_mcts_RL, keepdims=False)


        dataset_mcts_R1.append(dialog_avg_mcts_R1)
        dataset_mcts_R2.append(dialog_avg_mcts_R2)
        dataset_mcts_RL.append(dialog_avg_mcts_RL)

        print("-"*100)
        print("Dialog ID  |    MCTS Rouge-1 Score in Sentences   |  MCTS Rouge-2 Score in Sentences    |    MCTS Rouge-L Score in Sentences")
        print("Chatgpt Dialog Rouge Score in Sentences:", dialog_id, "  |    ", dialog_avg_mcts_R1, "    |    ", dialog_avg_mcts_R2, "    |    ", dialog_avg_mcts_RL)
        print("-"*100)



        dialog_avg_chatgpt_human_event_cover = np.mean(dialog_chatgpt_human_event_cover, keepdims=False)

        dataset_chatgpt_human_event_cover.append(dialog_avg_chatgpt_human_event_cover)

        print("-"*100)
        print("Dialog ID  |    Chatgpt Human Event Coverage")
        print("Chatgpt Human Event Coverage Score:", dialog_id, "  |    ", dialog_avg_chatgpt_human_event_cover)
        print("-"*100)



        dialog_avg_mcts_human_event_cover = np.mean(dialog_mcts_human_event_cover, keepdims=False)

        dataset_mcts_human_event_cover.append(dialog_avg_mcts_human_event_cover)

        print("-"*100)
        print("Dialog ID  |    MCTS Human Event Coverage")
        print("MCTS Human Event Coverage Score:", dialog_id, "  |    ", dialog_avg_mcts_human_event_cover)
        print("-"*100)

        dialog_avg_chatgpt_human_event_cover_max = np.mean(dialog_chatgpt_human_event_cover_max, keepdims=False)

        dataset_chatgpt_human_event_cover_max.append(dialog_avg_chatgpt_human_event_cover_max)

        print("-"*100)
        print("Dialog ID  |    Chatgpt Human Event Coverage Max")
        print("Chatgpt Human Event Coverage Score:", dialog_id, "  |    ", dialog_avg_chatgpt_human_event_cover_max)
        print("-"*100)



        dialog_avg_mcts_human_event_cover_max = np.mean(dialog_mcts_human_event_cover_max, keepdims=False)

        dataset_mcts_human_event_cover_max.append(dialog_avg_mcts_human_event_cover_max)

        print("-"*100)
        print("Dialog ID  |    MCTS Human Event Coverage Max")
        print("MCTS Human Event Coverage Score:", dialog_id, "  |    ", dialog_avg_mcts_human_event_cover_max)
        print("-"*100)



        print("-"*100)
        print("Dialog ID  |    Mean Chatgpt MCTS Similarity BERT Precision    |  Mean Chatgpt MCTS Similarity BERT Recall    |    Mean Chatgpt MCTS Similarity BERT F1")

        dialog_avg_chatgpt_P = np.mean(dialog_chatgpt_mean_P, keepdims=False)
        dialog_avg_chatgpt_R = np.mean(dialog_chatgpt_mean_R, keepdims=False)
        dialog_avg_chatgpt_F1 = np.mean(dialog_chatgpt_mean_F1, keepdims=False)


        dataset_chatgpt_mean_P.append(dialog_avg_chatgpt_P)
        dataset_chatgpt_mean_R.append(dialog_avg_chatgpt_R)
        dataset_chatgpt_mean_F1.append(dialog_avg_chatgpt_F1)

        print("-"*100)
        print("Mean Chatgpt MCTS Similarity BERT Score:", dialog_id, "  |    ", dialog_avg_chatgpt_P, "    |    ", dialog_avg_chatgpt_R, "    |    ", dialog_avg_chatgpt_F1)
        print("-"*100)

        print("Dialog ID  |    Max Chatgpt MCTS Similarity BERT Precision       |    Max Chatgpt MCTS Similarity BERT Recall      |      Max Chatgpt MCTS Similarity BERT F1")

        dialog_max_chatgpt_P = np.mean(dialog_chatgpt_max_P, keepdims=False)
        dialog_max_chatgpt_R = np.mean(dialog_chatgpt_max_R, keepdims=False)
        dialog_max_chatgpt_F1 = np.mean(dialog_chatgpt_max_F1, keepdims=False)


        dataset_chatgpt_max_P.append(dialog_max_chatgpt_P)
        dataset_chatgpt_max_R.append(dialog_max_chatgpt_R)
        dataset_chatgpt_max_F1.append(dialog_max_chatgpt_F1)

        print("-"*100)
        print("Max Chatgpt MCTS Similarity Max BERT Score", dialog_id, "  |    ", dialog_max_chatgpt_P, "    |    ", dialog_max_chatgpt_R, "    |    ", dialog_max_chatgpt_F1)
        print("-"*100)


        print("-"*100)
        print("Dialog ID  |    Mean BART Score")

        dialog_avg_chatgpt_bart = np.mean(dialog_chatgpt_bart, keepdims=False)

        dataset_chatgpt_bart.append(dialog_avg_chatgpt_bart)

        print("-"*100)
        print("Dialog Mean BART Score:", dialog_id, "  |    ", dialog_avg_chatgpt_bart)
        print("-"*100)

        print("Dialog ID  |    Max BART Score")

        dialog_max_chatgpt_bart = np.mean(dialog_chatgpt_max_bart, keepdims=False)

        dataset_chatgpt_max_bart.append(dialog_max_chatgpt_bart)

        print("-"*100)
        print("Dialog Max BART Score", dialog_id, "  |    ", dialog_max_chatgpt_bart)
        print("-"*100)


        print("-"*100)
        print("Dialog ID  |    Mean Mover Score")

        dialog_avg_chatgpt_mover = np.mean(dialog_chatgpt_mover, keepdims=False)

        dataset_chatgpt_mover.append(dialog_avg_chatgpt_mover)

        print("-"*100)
        print("Dialog Mean Mover Score:", dialog_id, "  |    ", dialog_avg_chatgpt_mover)
        print("-"*100)

        print("Dialog ID  |    Max Mover Score")

        dialog_max_chatgpt_mover = np.mean(dialog_chatgpt_max_mover, keepdims=False)

        dataset_chatgpt_max_mover.append(dialog_max_chatgpt_mover)

        print("-"*100)
        print("Dialog Max Mover Score", dialog_id, "  |    ", dialog_max_chatgpt_mover)
        print("-"*100)



    # dataset output

    print("-"*100)
    print("-"*100)

    avg_dataset_chatgpt_R1 = np.mean(dataset_chatgpt_R1, keepdims=False)
    avg_dataset_chatgpt_R2 = np.mean(dataset_chatgpt_R2, keepdims=False)
    avg_dataset_chatgpt_RL = np.mean(dataset_chatgpt_RL, keepdims=False)


    print("Dataset  |    Chatgpt Rouge-1 Score in Sentences      |    Chatgpt Rouge-2 Score in Sentences      |      Chatgpt Rouge-L Score in Sentences")
    print("Dialog", "  |    ", avg_dataset_chatgpt_R1, "    |    ", avg_dataset_chatgpt_R2, "    |    ", avg_dataset_chatgpt_RL)



    print("-"*100)
    print("-"*100)

    avg_dataset_mcts_R1 = np.mean(dataset_mcts_R1, keepdims=False)
    avg_dataset_mcts_R2 = np.mean(dataset_mcts_R2, keepdims=False)
    avg_dataset_mcts_RL = np.mean(dataset_mcts_RL, keepdims=False)


    print("Dataset  |    MCTS Rouge-1 Score in Sentences      |    MCTS Rouge-2 Score in Sentences      |      MCTS Rouge-L Score in Sentences")
    print("Dialog", "  |    ", avg_dataset_mcts_R1, "    |    ", avg_dataset_mcts_R2, "    |    ", avg_dataset_mcts_RL)



    print("-"*100)
    print("-"*100)

    avg_dataset_chatgpt_P = np.mean(dataset_chatgpt_P, keepdims=False)
    avg_dataset_chatgpt_R = np.mean(dataset_chatgpt_R, keepdims=False)
    avg_dataset_chatgpt_F1 = np.mean(dataset_chatgpt_F1, keepdims=False)


    print("Dataset  |    Chatgpt BERT Score Precision in Sentences      |    Chatgpt BERT Score Recall in Sentences      |      Chatgpt BERT Score F1 in Sentences")
    print("Dialog", "  |    ", avg_dataset_chatgpt_P, "    |    ", avg_dataset_chatgpt_R, "    |    ", avg_dataset_chatgpt_F1)



    print("-"*100)
    print("-"*100)

    avg_dataset_mcts_P = np.mean(dataset_mcts_P, keepdims=False)
    avg_dataset_mcts_R = np.mean(dataset_mcts_R, keepdims=False)
    avg_dataset_mcts_F1 = np.mean(dataset_mcts_F1, keepdims=False)


    print("Dataset  |    MCTS BERT Score Precision in Sentences      |    MCTS BERT Score Recall in Sentences      |      MCTS BERT Score F1 in Sentences")
    print("Dialog", "  |    ", avg_dataset_mcts_P, "    |    ", avg_dataset_mcts_R, "    |    ", avg_dataset_mcts_F1)

    print("-"*100)
    print("-"*100)


    avg_dataset_chatgpt_human_event_cover = np.mean(dataset_chatgpt_human_event_cover, keepdims=False)

    print("Dataset  |    Chatgpt Human Event Coverage")
    print("Dialog", "  |    ", avg_dataset_chatgpt_human_event_cover)

    print("-"*100)
    print("-"*100)
    

    avg_dataset_mcts_human_event_cover = np.mean(dataset_mcts_human_event_cover, keepdims=False)

    print("Dataset  |    MCTS Human Event Coverage")
    print("Dialog", "  |    ", avg_dataset_mcts_human_event_cover)

    print("-"*100)
    print("-"*100)


    avg_dataset_chatgpt_human_event_cover_max = np.mean(dataset_chatgpt_human_event_cover_max, keepdims=False)

    print("Dataset  |    Chatgpt Human Event Coverage Max")
    print("Dialog", "  |    ", avg_dataset_chatgpt_human_event_cover_max)

    print("-"*100)
    print("-"*100)
    

    avg_dataset_mcts_human_event_cover_max = np.mean(dataset_mcts_human_event_cover_max, keepdims=False)

    print("Dataset  |    MCTS Human Event Coverage Max")
    print("Dialog", "  |    ", avg_dataset_mcts_human_event_cover_max)

    print("-"*100)
    print("-"*100)


    avg_dataset_chatgpt_P = np.mean(dataset_chatgpt_mean_P, keepdims=False)
    avg_dataset_chatgpt_R = np.mean(dataset_chatgpt_mean_R, keepdims=False)
    avg_dataset_chatgpt_F1 = np.mean(dataset_chatgpt_mean_F1, keepdims=False)


    print("Dataset  |    Mean Chatgpt MCTS Similarity BERT Precision       |    Mean Chatgpt MCTS Similarity BERT Recall      |      Mean Chatgpt MCTS Similarity BERT F1")
    print("Dialog Mean", "  |    ", avg_dataset_chatgpt_P, "    |    ", avg_dataset_chatgpt_R, "    |    ", avg_dataset_chatgpt_F1)



    print("-"*100)
    print("-"*100)

    avg_dataset_chatgpt_max_P = np.mean(dataset_chatgpt_max_P, keepdims=False)
    avg_dataset_chatgpt_max_R = np.mean(dataset_chatgpt_max_R, keepdims=False)
    avg_dataset_chatgpt_max_F1 = np.mean(dataset_chatgpt_max_F1, keepdims=False)


    print("Dataset  |    Max Chatgpt MCTS Similarity BERT Precision       |    Max Chatgpt MCTS Similarity BERT Recall      |      Max Chatgpt MCTS Similarity BERT F1")
    print("Dialog Max", "  |    ", avg_dataset_chatgpt_max_P, "    |    ", avg_dataset_chatgpt_max_R, "    |    ", avg_dataset_chatgpt_max_F1)

    print("-"*100)
    print("-"*100)

    avg_dataset_chatgpt_bart = np.mean(dataset_chatgpt_bart, keepdims=False)

    print("Dataset  |    Mean BART Score")
    print("Dialog Mean", "  |    ", avg_dataset_chatgpt_bart)

    print("-"*100)
    print("-"*100)

    avg_dataset_chatgpt_max_bart = np.mean(dataset_chatgpt_max_bart, keepdims=False)

    print("Dataset  |    Max BART Score")
    print("Dialog Max", "  |    ", avg_dataset_chatgpt_max_bart)

    print("-"*100)
    print("-"*100)

    avg_dataset_chatgpt_mover = np.mean(dataset_chatgpt_mover, keepdims=False)

    print("Dataset  |    Mean Mover Score")
    print("Dialog Mean", "  |    ", avg_dataset_chatgpt_mover)

    print("-"*100)
    print("-"*100)

    avg_dataset_chatgpt_max_mover = np.mean(dataset_chatgpt_max_mover, keepdims=False)

    print("Dataset  |    Max Mover Score")
    print("Dialog Max", "  |    ", avg_dataset_chatgpt_max_mover)

    print("-"*100)
    print("-"*100)




if __name__ == "__main__":
    main()
