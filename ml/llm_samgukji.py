# 1. 라이브러리 가져오기
import google.generativeai as genai
import re
import os

# 2. API 키 설정 (환경변수 사용 권장)
# os.environ['GOOGLE_API_KEY'] = "AIza...여기에_발급받은_API_키를_붙여넣으세요"
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# 3. 사용할 모델 설정
model = genai.GenerativeModel('gemini-flash-latest') # 빠르고 저렴한 최신 모델

# 1. AI에게 '선택지 제시' 규칙을 추가합니다.
game_master_prompt = """
너는 지금부터 삼국지 시대 배경의 텍스트 어드벤처 게임을 진행하는 유능한 게임 마스터(GM)다.
- 게임의 배경: 후한 말기, 황건적의 난으로 혼란스러운 중국.
- 플레이어의 역할: 너와 대화하는 나는 '관우'다.
- 너의 임무: 내가 '관우'로서 겪는 상황을 실감 나고 흥미진진하게 묘사하고, 내 행동에 따른 결과를 알려줘야 한다.
- **매우 중요한 규칙: 너의 모든 답변 끝에는, 내가 다음에 할 수 있는 행동 3가지를 반드시 번호를 붙여서 제시해야 한다. 예시) 1. 유비의 제안을 받아들인다. 2. 술을 더 달라고 한다. 3. 아무 말 없이 자리를 뜬다.**
- 시작 상황: 게임은 관우가 형주 양양의 한 주점에서 천하의 혼란을 지켜보는 장면에서 시작된다.
"""

# 2. 'history'에 역할 부여 프롬프트를 첫 대화로 미리 넣어줍니다.
chat_history = [
    {
        "role": "user",
        "parts": game_master_prompt,
    },
    {
        "role": "model",
        "parts": "알겠습니다. 지금부터 저는 삼국지 게임 마스터입니다. 플레이어는 관우이며, 매 턴마다 3개의 선택지를 제공하겠습니다. 게임을 시작하겠습니다."
    }
]

# 3. 미리 설정된 history로 채팅 세션을 시작합니다.
chat = model.start_chat(history=chat_history)

print("🎲 삼국지 게임을 시작합니다. '종료'를 입력하면 게임이 끝납니다.")
print("-" * 50)

# 4. 게임 시작을 알리는 첫 번째 메시지를 보냅니다.
initial_response = chat.send_message("게임 시작")
print(f"🤖 GM: {initial_response.text.strip()}\n")

# 선택지를 저장할 리스트
current_choices = []

while True:
    # 5. AI의 답변에서 선택지만 추출하여 저장합니다.
    # 정규 표현식을 사용해 '1.', '2.', '3.'으로 시작하는 문장을 찾습니다.
    full_response_text = chat.history[-1].parts[0].text # 마지막 AI 답변 가져오기
    choices_text = re.findall(r'^\d\.\s.*', full_response_text, re.MULTILINE)
    
    # 6. 사용자에게 행동을 입력받습니다.
    user_action = input("🙂 관우: ")
    
    if user_action.lower() == '종료':
        print("🤖 GM: 게임을 종료합니다.")
        break
    
    # 7. 사용자가 숫자를 입력하면 해당 선택지로 변환합니다.
    final_action = user_action
    if user_action.isdigit() and choices_text:
        choice_index = int(user_action) - 1
        if 0 <= choice_index < len(choices_text):
            # "1. 유비의 제안을 받아들인다." 에서 "유비의 제안을 받아들인다." 부분만 추출
            final_action = re.sub(r'^\d\.\s', '', choices_text[choice_index])
            print(f"(선택: {final_action})")

    try:
        response = chat.send_message(final_action)
        print(f"🤖 GM: {response.text.strip()}\n")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        break