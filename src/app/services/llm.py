from google import genai
from google.genai import types
import json
import os

questionTypeDeterminerPath = "utils/questionTypeChecker.json"
normalizeUserinputPath = "utils/normalizeUserinput.json"

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def createClient(key):
    try:
        client = genai.Client(api_key=key)
        return client
    except:
        print("Not able to get client")
        return None

def getResponse(client, contents):
    response = client.models.generate_content(
        model = "gemini-flash-latest",
        contents=[contents]
    )
    return response

def determineQuestionType(client, message):
    message = normalizeInput(message)

    config_data = load_config(questionTypeDeterminerPath)
    contents = []
    for ex in config_data["examples"]:
        contents.append({"role": "user", "parts": [{"text": ex["input"]}]})
        contents.append({"role": "model", "parts": [{"text": json.dumps(ex["output"])}]})
    
    contents.append({"role": "user", "parts": [{"text": message}]})
    
    api_config = types.GenerateContentConfig(
        system_instruction=config_data["system_instruction"],
        response_mime_type = config_data["output_mime_type"],
    )

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=contents,
        config=api_config
    )
    classification = json.loads(response.text)
    return classification

def normalizeInput(client, message):
    config_data = load_config(questionTypeDeterminerPath)
    contents = []
    for ex in config_data["examples"]:
        contents.append({"role": "user", "parts": [{"text": ex["input"]}]})
        contents.append({"role": "model", "parts": [{"text": json.dumps(ex["output"])}]})
    
    contents.append({"role": "user", "parts": [{"text": message}]})
    
    api_config = types.GenerateContentConfig(
        system_instruction=config_data["system_instruction"],
        response_mime_type = config_data["output_mime_type"],
    )

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=contents,
        config=api_config
    )
    normalzedSentence = json.load(response.text)
    return normalzedSentence["sentence"]