import sys
import time
import json
import openai
import argparse
import numpy as np
from tqdm import tqdm
from itertools import chain
from itertools import combinations

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
 
@retry(wait=wait_random_exponential(min=6, max=200), stop=stop_after_attempt(10))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)



retrieve_knowledge_list_path ="xxx.txt"
dialog_path = "xxx.jsonl"
chatgpt_commonsense_list_path = "xxx.txt"
chatgpt_mcts_response_path = "xxx.txt"



START = 0


original_prompt = "You will be generating the next utterance of a dialogue between two people. Context: {}"
knowledge_prompt = "You will be generating the next utterance of a dialogue between two people based on a piece of knowledge. Context: {} Knowledge: {}" 

original_prompt = "Can we try dialogue generation? I will give you turns, and you can generated the next turn, but only one. Dialogue turns: {}"
knowledge_prompt = "Can we try dialogue generation? I will give you turns, and you can generated the next turn, but only one.\n\n You can also consider the knowledge for your reference in the dialogue. Dialogue turns: {} Knowledge: {}"

def main():
    
    openai.api_key = "xxx"

    with open(dialog_path, "r") as f:
        Dialog_corpus = json.load(f)

    with open(retrieve_knowledge_list_path, "r", encoding="utf8") as f_in:
        retrieve_knowledge = json.load(f_in)

    with open(chatgpt_commonsense_list_path, "r", encoding="utf8") as f_in:
        dataset_commonsense = json.load(f_in)

    retrieve_knowledge = retrieve_knowledge[0]

    dialog_num = len(retrieve_knowledge["dataset_context_summ"])

    dataset_original_chatgpt, dataset_mcts_chatgpt, dataset_knowledge_chatgpt = [], [], []


    with open(chatgpt_mcts_response_path, "w", encoding="utf8") as f:

        for dialog_id in range(dialog_num):

            f.write(str(dialog_id+1)+"\n")
        
            dataset_context_summ = retrieve_knowledge["dataset_context_summ"][dialog_id]
            dataset_entity_connection = retrieve_knowledge["dataset_entity_connection"][dialog_id]
            dataset_dialog_response = retrieve_knowledge["dataset_dialog_response"][dialog_id]
            dataset_events = retrieve_knowledge["dataset_events"][dialog_id]
            dataset_event_sents = retrieve_knowledge["dataset_event_sents"][dialog_id]

            chatgpt_labels = dataset_commonsense[dialog_id+START]

            dialog = Dialog_corpus[dialog_id+START]

            dialog_context_list = []
            
            dialog_original_chatgpt_list = []
            dialog_mcts_chatgpt_list = []
            dialog_knowledge_chatgpt_list = []

            dialog_context = ""

            for utter_id in range(len(chatgpt_labels)):

                chatgpt_label = chatgpt_labels[utter_id]

                context_summ = dataset_context_summ[utter_id]
                entity_connection = dataset_entity_connection[utter_id]
                dialog_response = dataset_dialog_response[utter_id]
                events = dataset_events[utter_id]
                event_sents = dataset_event_sents[utter_id]

                utterance = dialog[utter_id]["utterance"]

                if utter_id == len(dialog)-1:
                    break
                elif (utter_id % 2) == 0:
                    utterance = "Context: PersonX: " + utterance
                else:
                    utterance = "Context: PersonY: " + utterance

                dialog_context = dialog_context + utterance + "\n"

                dialog_context_list.append(dialog_context)
                f.write("Context: "+dialog_context + "\n")

                # Chatgpt original response 
                original_chatgpt_prompt = original_prompt.format(dialog_context)

                original_chatgpt_response = completion_with_backoff(
                    model="gpt-3.5-turbo",
                    messages=[
                            {"role": "system", "content": "You are a helpful assistant that generates dialogue response."},
                            {"role": "user", "content": original_chatgpt_prompt},
                        ]
                )

                original_chatgpt_result = original_chatgpt_response["choices"][0]["message"]["content"]


                dialog_original_chatgpt_list.append(original_chatgpt_result)

                f.write("ChatGPT Original Response: "+original_chatgpt_result+"\n")
                

                f.write("\n"+"MCTS Knowledge Response"+"\n")
                for i in range(len(dialog_response)):
                    response = dialog_response[i]
                    response_knowledge_prompt = knowledge_prompt.format(dialog_context, response)

                    dialog_mcts_response = completion_with_backoff(
                        model="gpt-3.5-turbo",
                        messages=[
                                {"role": "system", "content": "You are a helpful assistant that generates dialogue response."},
                                {"role": "user", "content": response_knowledge_prompt},
                            ]
                    )

                    dialog_mcts_result = dialog_mcts_response["choices"][0]["message"]["content"]

                    f.write("MCTS Knowledge:"+response+"\n")
                    f.write("MCTS Knowledge Response:"+dialog_mcts_result+"\n")

                # With event understanding
                for i in range(len(events)):
                    event = events[i]
                    event_sent = event_sents[i]

                    add_knowledge = event_sent[0]
                    event_knowledge = " ".join((add_knowledge, event))
                    event_knowledge_prompt = knowledge_prompt.format(dialog_context, event_knowledge)

                    dialog_mcts_chatgpt_response = completion_with_backoff(
                        model="gpt-3.5-turbo",
                        messages=[
                                {"role": "system", "content": "You are a helpful assistant that generates dialogue response."},
                                {"role": "user", "content": event_knowledge_prompt},
                            ]
                    )

                    dialog_mcts_chatgpt_result = dialog_mcts_chatgpt_response["choices"][0]["message"]["content"]

                    dialog_mcts_chatgpt_list.append([event_knowledge, dialog_mcts_chatgpt_result])
                    f.write("MCTS Knowledge:"+event_knowledge+"\n")
                    f.write("MCTS Knowledge Response:"+dialog_mcts_chatgpt_result+"\n")



                f.write("\n"+"Chatgpt Knowledge Response"+"\n")
                # With Chatgpt Knowledge
                for chatgpt_knowledge in chatgpt_label:
                    chatgpt_knowledge_prompt = knowledge_prompt.format(dialog_context, chatgpt_knowledge)

                    dialog_knowledge_chatgpt_response = completion_with_backoff(
                            model="gpt-3.5-turbo",
                            messages=[
                                    {"role": "system", "content": "You are a helpful assistant that generates dialogue response."},
                                    {"role": "user", "content": chatgpt_knowledge_prompt},
                                ]
                        )

                    dialog_knowledge_chatgpt_result = dialog_knowledge_chatgpt_response["choices"][0]["message"]["content"]
                    
                    dialog_knowledge_chatgpt_list.append([chatgpt_knowledge, dialog_knowledge_chatgpt_result])

                    f.write("ChatGPT Knowledge:"+chatgpt_knowledge+"\n")
                    f.write("ChatGPT Knowledge Response:"+dialog_knowledge_chatgpt_result+"\n")


                f.write("\n"+"-"*100+"\n")
            f.write("\n\n"+"="*100+"\n")





if __name__ == "__main__":
    main()


