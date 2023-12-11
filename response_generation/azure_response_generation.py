import os
import argparse
import openai
import json

# Load config values
with open('config.json', 'r') as config_file:
    config_details = json.load(config_file)
    
# Setting up the deployment name
chatgpt_model_name = config_details['CHATGPT_MODEL']

# This is set to `azure`
openai.api_type = "azure"

# The API key for your Azure OpenAI resource.
openai.api_key = 'xxx'

# The base URL for your Azure OpenAI resource. e.g. "https://<your resource name>.openai.azure.com"
openai.api_base = config_details['OPENAI_API_BASE']

# Currently Chat Completion API have the following versions available: 2023-03-15-preview
openai.api_version = config_details['OPENAI_API_VERSION']



retrieve_knowledge_list_path ="xxx.txt"
dialog_path = "xxx.jsonl"
chatgpt_commonsense_list_path = "xxx.txt"
chatgpt_mcts_response_path = "xxx.txt"

START = 0

original_prompt = "Can we try dialogue generation? I will give you turns, and you can generated the next turn, but only one. Dialogue turns: {}. Response:"
knowledge_prompt = "Can we try dialogue generation? I will give you turns, and you can generated the next turn, but only one.\n\n You can also consider the knowledge for your reference in the dialogue. Dialogue turns: {} Knowledge: {}. Response:"



def call_chatgpt(chatgpt_prompt):

    result = None

    # A sample API call for chat completions looks as follows:
    # Messages must be an array of message objects, where each object has a role (either "system", "user", or "assistant") and content (the content of the message).
    # For more info: https://learn.microsoft.com/en-us/azure/cognitive-services/openai/reference#chat-completions

    
    try:
        response = openai.ChatCompletion.create(
                    engine=chatgpt_model_name,
                    messages=[
                            {"role": "system", "content": "You are a helpful assistant that generates dialogue response."},
                            {"role": "user", "content": chatgpt_prompt},
                        ]
                    )

        # print the response
        # print(response['choices'][0]['message']['content'])
        result = response['choices'][0]['message']['content']
        
        
    except openai.error.APIError as e:
        # Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")

    except openai.error.AuthenticationError as e:
        # Handle Authentication error here, e.g. invalid API key
        print(f"OpenAI API returned an Authentication Error: {e}")

    except openai.error.APIConnectionError as e:
        # Handle connection error here
        print(f"Failed to connect to OpenAI API: {e}")

    except openai.error.InvalidRequestError as e:
        # Handle connection error here
        print(f"Invalid Request Error: {e}")

    except openai.error.RateLimitError as e:
        # Handle rate limit error
        print(f"OpenAI API request exceeded rate limit: {e}")

    except openai.error.ServiceUnavailableError as e:
        # Handle Service Unavailable error
        print(f"Service Unavailable: {e}")

    except openai.error.Timeout as e:
        # Handle request timeout
        print(f"Request timed out: {e}")
        
    except:
        # Handles all other exceptions
        print("An exception has occured.")

    return result

    


if __name__ == "__main__":
    
    with open(dialog_path, "r") as f:
        Dialog_corpus = json.load(f)

    with open(retrieve_knowledge_list_path, "r", encoding="utf8") as f_in:
        retrieve_knowledge = json.load(f_in)

    with open(chatgpt_commonsense_list_path, "r", encoding="utf8") as f_in:
        dataset_commonsense = json.load(f_in)

    retrieve_knowledge = retrieve_knowledge[0]

    dialog_num = len(retrieve_knowledge["dataset_context_summ"])

    dataset_original_chatgpt, dataset_mcts_chatgpt, dataset_knowledge_chatgpt = [], [], []

    

    with open(chatgpt_mcts_response_path, "a+", encoding="utf8") as f:

        for dialog_id in range(dialog_num):

            f.write(str(dialog_id+1)+"\n")
            print("Dialogue -", str(dialog_id+1))
        
            dataset_context_summ = retrieve_knowledge["dataset_context_summ"][dialog_id - START]
            dataset_entity_connection = retrieve_knowledge["dataset_entity_connection"][dialog_id - START]
            dataset_dialog_response = retrieve_knowledge["dataset_dialog_response"][dialog_id - START]
            dataset_events = retrieve_knowledge["dataset_events"][dialog_id - START]
            dataset_event_sents = retrieve_knowledge["dataset_event_sents"][dialog_id - START]

            chatgpt_labels = dataset_commonsense[dialog_id]

            dialog = Dialog_corpus[dialog_id]

            dialog_context_list = []
            
            dialog_original_chatgpt_list = []
            dialog_mcts_chatgpt_list = []
            dialog_knowledge_chatgpt_list = []

            dialog_context = ""

            for utter_id in range(len(chatgpt_labels)):

                print("utter -", str(utter_id+1))

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

                original_chatgpt_result = call_chatgpt(original_chatgpt_prompt)

                if original_chatgpt_result != None:

                    f.write("ChatGPT Original Response: "+original_chatgpt_result+"\n")


                f.write("\n"+"MCTS Knowledge Response"+"\n")
                for i in range(len(dialog_response)):
                    response = dialog_response[i]
                    response_knowledge_prompt = knowledge_prompt.format(dialog_context, response)

                    dialog_mcts_result = call_chatgpt(response_knowledge_prompt)

                    if dialog_mcts_result != None:

                        f.write("MCTS Knowledge:"+response+"\n")

                        f.write("MCTS Knowledge Response:"+dialog_mcts_result+"\n")

                # With event understanding
                for i in range(len(events)):
                    event = events[i]
                    event_sent = event_sents[i]

                    add_knowledge = event_sent[0]
                    event_knowledge = " ".join((add_knowledge, event))
                    event_knowledge_prompt = knowledge_prompt.format(dialog_context, event_knowledge)

                    dialog_mcts_chatgpt_result = call_chatgpt(event_knowledge_prompt)

                    if dialog_mcts_chatgpt_result!= None:

                        f.write("MCTS Knowledge:"+event_knowledge+"\n")
                        f.write("MCTS Knowledge Response:"+dialog_mcts_chatgpt_result+"\n")

                f.write("\n"+"Chatgpt Knowledge Response"+"\n")
                # With Chatgpt Knowledge
                for chatgpt_knowledge in chatgpt_label:
                    chatgpt_knowledge_prompt = knowledge_prompt.format(dialog_context, chatgpt_knowledge)

                    dialog_knowledge_chatgpt_result = call_chatgpt(chatgpt_knowledge_prompt)

                    if dialog_knowledge_chatgpt_result != None:

                        f.write("ChatGPT Knowledge:"+chatgpt_knowledge+"\n")
                        f.write("ChatGPT Knowledge Response:"+dialog_knowledge_chatgpt_result+"\n")
                
                
                f.write("\n"+"-"*100+"\n")
            
            f.write("\n\n"+"="*100+"\n")
            
            print("\n")
