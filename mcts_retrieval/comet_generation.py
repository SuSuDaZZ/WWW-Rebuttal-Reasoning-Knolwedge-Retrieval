import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel
from utils import use_task_specific_params, trim_batch



class ContextGPT2:
    def __init__(self, model_path):
        # Load pre-trained model (weights)
        with torch.no_grad():
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.model.eval()
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        

    def cal_ppl(self, sents, context):
        ppl_scores = []
        for sent in sents:
            input_sent = " ".join((context, sent))
            inputs = self.tokenizer.encode(input_sent)
            tensor_input = torch.tensor([inputs])
            loss = self.model(tensor_input, labels=tensor_input)[0]
            ppl = torch.exp(loss.detach())
            ppl_scores.append(ppl)
        ppl_scores = torch.stack(ppl_scores)
        return ppl_scores



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class Comet:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarization"
        use_task_specific_params(self.model, task)
        self.batch_size = 1
        self.decoder_start_token_id = None

    def generate(
            self, 
            queries,
            decode_method="beam", 
            num_generate=5, 
            ):

        with torch.no_grad():
            examples = queries

            decs = []
            for batch in list(chunks(examples, self.batch_size)):

                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)

                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=num_generate,
                    num_return_sequences=num_generate,
                    )

                dec = self.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                decs.append(dec)

            return decs


all_relations = [
    "AtLocation",
    "CapableOf",
    "Causes",
    "CausesDesire",
    "CreatedBy",
    "DefinedAs",
    "DesireOf",
    "Desires",
    "HasA",
    "HasFirstSubevent",
    "HasLastSubevent",
    "HasPainCharacter",
    "HasPainIntensity",
    "HasPrerequisite",
    "HasProperty",
    "HasSubEvent",
    "HasSubevent",
    "HinderedBy",
    "InheritsFrom",
    "InstanceOf",
    "IsA",
    "LocatedNear",
    "LocationOfAction",
    "MadeOf",
    "MadeUpOf",
    "MotivatedByGoal",
    "NotCapableOf",
    "NotDesires",
    "NotHasA",
    "NotHasProperty",
    "NotIsA",
    "NotMadeOf",
    "ObjectUse",
    "PartOf",
    "ReceivesAction",
    "RelatedTo",
    "SymbolOf",
    "UsedFor",
    "isAfter",
    "isBefore",
    "isFilledBy",
    "oEffect",
    "oReact",
    "oWant",
    "xAttr",
    "xEffect",
    "xIntent",
    "xNeed",
    "xReact",
    "xReason",
    "xWant",
    ]

if __name__ == "__main__":
    
    # sample usage 
    print("model loading ...")
    comet = Comet("projects/MCTS_GenericsKB/MCTS_GKB_DailyDialog/comet-atomic_2020_BART")
    comet.model.zero_grad()
    print("model loaded")
    queries = []
    head = "PersonX has reduced their energy consumption in their factory by about 15 percent in the last two years."
    rel = "xEffect"
    query = "{} {} [GEN]".format(head, rel)
    queries.append(query)
    print(queries)
    results = comet.generate(queries, decode_method="beam", num_generate=5)
    print(results)