import os
import re
import argparse
import openai
import json
import time
import numpy as np

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



def original_evaluator(original_prompt, samples):
    try:
        response = openai.ChatCompletion.create(
                      engine=chatgpt_model_name,
                      messages=[
                            {"role": "system", "content": original_prompt},
                            {"role": "user", "content": samples}
                        ]
                    )

        # print the response
        #print(response['choices'][0]['message']['content'])
        # assign the returned string to s
        s = response['choices'][0]['message']['content']
        # convert the string to a list
        l = s.replace(':', ',').strip('.')
        l = [x for x in l.split('Response') if x]
        l0 = l[0].strip().split(',')
        l0 = [x for x in l0 if x.isdigit()]
        original_score = [float(i) for i in l0[-8:]]
        l1 = l[1].strip().split(',')
        l1 = [x for x in l1 if x.isdigit()]
        chatgptk_score = [float(i) for i in l1[-8:]]

        return original_score, chatgptk_score
        
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
    




def evaluator(prompt, samples):
    try:
        response = openai.ChatCompletion.create(
                      engine=chatgpt_model_name,
                      messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": samples}
                        ]
                    )

        # print the response
        s = response['choices'][0]['message']['content']
        print(s)
        # convert the string to a list
        l = s.replace(':', ',').strip('.').split(',')
        scores = [float(i) for i in l[-8:]]
        
        return scores
        
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
        return [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]


file_path = "xxx.txt"
with open(file_path, "r") as f:
    conversations = f.read()

if __name__ == "__main__":

    conv_original = []
    conv_mcts = []
    conv_chatgpt = []

    conversations = conversations.split('====================================================================================================')

    for text in conversations:

        # split the text into lines
        text = text.split('\n')
        # remove quotation marks insides a string
        text = [s.replace('"', '') for s in text]
        # find the indices of the splitter
        splitter_indices = [i for i, x in enumerate(text) if x == "----------------------------------------------------------------------------------------------------"]
        # split the text into utterances
        utterances = []
        for i, idx in enumerate(splitter_indices):
            if i == 0:
                utterance = text[:idx]
                utterances.append(utterance)
            else:
                utterance = text[splitter_indices[i-1]+1:idx]
                utterances.append(utterance)



        mcts_scores = []
        chatgpt_scores = []
        original_scores = []
        chatgpt_logical_scores = []
        for u_id, u in enumerate(utterances):
            print("utter -", u_id)

            prompt = "Given a conversation between two people, now you are supposed to rate candidate responses. \
                        You should evaluate all the candidate responses together, based on eight (8) aspects: coherence, naturalness, engagingness, groundedness, creativity, informativeness, knowledgeability, and logical complexity. \
                        Each aspect ranges from 1-5, and scores can be decimal. Please only return the scores of each aspect of the responses without any text descriptions. \
                        Remember to use the following template for each response when returning results:\
                        Response: response id\
                        coherence:\
                        naturalness:\
                        engagingness:\
                        groundness:\
                        creativity:\
                        informativeness:\
                        knowledgeability:\
                        logical conplexity:\
                        \n"

            
            conv_context = ""
            chatgpt_original = ""
            mcts_combined = ""
            i_mcts = 1
            chatgpt_combined = ""
            chatgpt_ks = []
            i_chatgpt = 1
            temp_idx = 0

            id2type=dict()
            response_list = []

            for i, line in enumerate(u):
                if "Context: " in line:
                    # prompt += line.strip('Context:')
                    conv_context += line.strip('Context:')
                elif "ChatGPT Original Response: " in line:
                    chatgpt_original = line.strip('ChatGPT Original Response: ')
                    temp_idx = i + 1
                    response_list.append(chatgpt_original)
                    temp_idx[len(response_list)] = "original"
                    break
            # concatenate all the mcts responses into one string        
            for j, line in enumerate(u[temp_idx:]):
                if "MCTS Knowledge Response:" in line:
                    mcts = line.strip('MCTS Knowledge Response:')
                    mcts_combined = mcts_combined + " Response " + str(i_mcts+1) + ": " + mcts + " "
                    response_list.append(mcts)
                    temp_idx[len(response_list)] = "mcts"
                    i_mcts += 1
                elif "ChatGPT Knowledge Response" in line:
                    # add the original chatgpt response to the front of the mcts responses
                    mcts_combined = "Response 1: "+ chatgpt_original + mcts_combined
                    temp_idx += j + 1
                    response_list.append(chatgpt)
                    temp_idx[len(response_list)] = "chatgpt"
                    break
            # concatenate all the chatgpt responses into one string        
            for k, line in enumerate(u[temp_idx:]):
                if "ChatGPT Knowledge Response:" in line:
                    chatgpt = line.strip('ChatGPT Knowledge Response:')
                    chatgpt_ks.append(chatgpt)
                    chatgpt_combined = chatgpt_combined + " Response " + str(i_chatgpt+1) + ": " + chatgpt + " "
                    i_chatgpt += 1
            # add the original chatgpt response to the front of the chatgpt responses
            chatgpt_combined = "Response 1: "+ chatgpt_original + chatgpt_combined
            
            # call the evaluator function to evaluate the responses by mcts and chatgpt
            chatgpt_score = evaluator(prompt + "Conversation: " + conv_context,  chatgpt_combined)
            chatgpt_score = np.array(chatgpt_score)
            if len(chatgpt_score) == 8:
                chatgpt_scores.append(chatgpt_score)

            mcts_score = evaluator(prompt, mcts_combined)
            mcts_score = np.array(mcts_score)
            if len(mcts_score) == 8:
                mcts_scores.append(mcts_score)
            

            time.sleep(2)



        print("original:", original_scores)    
        print("MCTS:", mcts_scores)
        print("CoK:", chatgpt_scores)


        # conv_original.append(np.mean(original_scores, axis=0))
        conv_mcts.append(np.mean(mcts_scores, axis=0))
        conv_chatgpt.append(np.mean(chatgpt_scores, axis=0))

        # print("mean original:", np.mean(conv_original, axis=0))    
        print("mean MCTS:", np.mean(conv_mcts, axis=0))  
        print("mean CoT:", np.mean(conv_chatgpt, axis=0))  



