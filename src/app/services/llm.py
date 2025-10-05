# app/services/llm.py
from __future__ import annotations
import json, os
from importlib import resources
from google import genai
from google.genai import types

# 패키지 내 리소스에서 읽기
def _load_config_from_pkg(pkg: str, filename: str) -> dict:
    text = resources.files(pkg).joinpath(filename).read_text(encoding="utf-8")
    return json.loads(text)

def create_client() -> genai.Client:
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is not set")
    return genai.Client(api_key=key)

def normalize_input(client: genai.Client, message: str) -> str:
    cfg = _load_config_from_pkg("app.utils", "normalizeUserinput.json")
    contents = []
    for ex in cfg.get("examples", []):
        contents.append({"role": "user", "parts": [{"text": ex["input"]}]})
        contents.append({"role": "model", "parts": [{"text": json.dumps(ex["output"])}]})
    contents.append({"role": "user", "parts": [{"text": message}]})

    api_config = types.GenerateContentConfig(
        system_instruction=cfg.get("system_instruction", ""),
        response_mime_type=cfg.get("output_mime_type", "application/json"),
    )
    resp = client.models.generate_content(
        model="gemini-flash-latest",
        contents=contents,
        config=api_config,
    )
    try:
        data = json.loads(resp.text)
        return data.get("sentence") or message
    except Exception:
        return message

def determine_question_type(client: genai.Client, message: str) -> dict:
    msg = normalize_input(client, message)
    cfg = _load_config_from_pkg("app.utils", "questionTypeChecker.json")
    contents = []
    for ex in cfg.get("examples", []):
        contents.append({"role": "user", "parts": [{"text": ex["input"]}]})
        contents.append({"role": "model", "parts": [{"text": json.dumps(ex["output"])}]})
    contents.append({"role": "user", "parts": [{"text": msg}]})

    api_config = types.GenerateContentConfig(
        system_instruction=cfg.get("system_instruction", ""),
        response_mime_type=cfg.get("output_mime_type", "application/json"),
    )
    resp = client.models.generate_content(
        model="gemini-flash-latest",
        contents=contents,
        config=api_config,
    )
    try:
        return json.loads(resp.text)
    except Exception:
        return {"type": "unknown", "raw": resp.text}
"""
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
    
"""
