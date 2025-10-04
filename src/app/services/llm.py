from google import genai

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
