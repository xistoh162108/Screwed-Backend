import google.generativeai as genai
import os

# --- 설정 파일 경로 (실제 파일은 프로젝트 내에 있어야 합니다) ---

from pathlib import Path
import json

# 현재 파일: .../src/app/services/XXX.py
HERE = Path(__file__).resolve()
APP_DIR = HERE.parents[1]          # .../src/app
UTILS_DIR = APP_DIR / "utils"      # .../src/app/utils

# 파일 경로 정의 (services/utils 가 아니라 utils)
questionTypeDeterminerPath = str(UTILS_DIR / "questionTypeChecker.json")
normalizeUserinputPath    = str(UTILS_DIR / "normalizeUserinput.json")
procedureAnalyzerPath     = str(UTILS_DIR / "procedureAnalyzer.json")
feedbackGeneratorPath     = str(UTILS_DIR / "feedbackGenerator.json")

def load_config(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {path}")
        return None

def _call_gemini_model(path, contents, model_name="gemini-flash-latest"):
    """Gemini API 호출을 위한 헬퍼 함수."""
    config_data = load_config(path)
    if not config_data:
        return None

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=config_data["system_instruction"]
    )

    generation_config = genai.GenerationConfig(
        response_mime_type=config_data.get("output_mime_type", "text/plain")
    )
    
    # 예제 추가
    full_contents = []
    for ex in config_data.get("examples", []):
        full_contents.append({"role": "user", "parts": [{"text": ex["input"]}]})
        full_contents.append({"role": "model", "parts": [{"text": json.dumps(ex["output"])}]})
    
    # 사용자 입력 추가
    full_contents.extend(contents)

    try:
        response = model.generate_content(
            contents=full_contents,
            generation_config=generation_config
        )
        # JSON 응답을 가정하고 로드 (text/plain이면 그냥 텍스트가 될 수 있음)
        if config_data.get("output_mime_type") == "application/json":
            return json.loads(response.text)
        return response.text
    except Exception as e:
        print(f"Gemini API call failed for {path}: {e}")
        return None

# --- 핵심 함수 구현 ---

def normalizeInput(message):
    """
    1. 오탈자, 별명 등을 정규화합니다.
    2. 문장 내의 조치 개수(action_count)를 추출하여 반환합니다.
    """
    # 템플릿 콘텐츠: 사용자 입력만 포함
    contents = [{"role": "user", "parts": [{"text": message}]}]
    
    # 'normalizeUserinputPath' 모델 호출
    normalized_data = _call_gemini_model(normalizeUserinputPath, contents)
    
    # 반환 구조 예시: {"normalized_text": "옥수수 심어.", "action_count": 1}
    return normalized_data if isinstance(normalized_data, dict) else {"normalized_text": message, "action_count": 0}

def determineQuestionType(normalizedData):
    """
    정규화된 입력 텍스트를 Q(질문), I(명령), O(기타) 중 하나로 분류합니다.
    """
    message_text = normalizedData.get("normalized_text", "")
    contents = [{"role": "user", "parts": [{"text": message_text}]}]
    
    # 'questionTypeDeterminerPath' 모델 호출
    classification = _call_gemini_model(questionTypeDeterminerPath, contents)
    
    # 반환 구조 예시: {"type": "I"}
    return classification if isinstance(classification, dict) else {"type": "O"}

def procedureAnalyzer(sentence):
    """
    단일 명령을 받아 게임 엔진용 JSON 객체(의도/매개변수)로 변환합니다.
    과도한 위임 요청(예: '알아서 해줘')은 '위임_불가' 의도로 반환됩니다.
    """
    contents = [{"role": "user", "parts": [{"text": sentence}]}]
    
    # 'procedureAnalyzerPath' 모델 호출
    structured_command = _call_gemini_model(procedureAnalyzerPath, contents)
    
    # 반환 구조 예시: {"intent": "심기", "crop": "옥수수", "target_area": "A-5"}
    return structured_command if isinstance(structured_command, dict) else {"intent": "무효", "error": "분석 실패"}

def questionHandler(question_text):
    """
    질문 텍스트를 분석하여 필요한 정보를 조회하고 자연어 답변을 생성합니다.
    """
    # 이 함수는 직접적으로 게임 데이터베이스나 로직을 조회해야 합니다.
    # 여기서는 답변을 생성하는 GPT 모델 호출을 시뮬레이션합니다.
    contents = [{"role": "user", "parts": [{"text": f"질문에 답변해주세요: {question_text}"}]}]
    
    # 'feedbackGeneratorPath' 모델 호출 (질문 응답 모드로 가정)
    response_data = _call_gemini_model(feedbackGeneratorPath, contents)

    if isinstance(response_data, str):
        return {"final_response": response_data, "status": "ANSWERED"}
        
    return {"final_response": "정보를 조회하는 데 실패했습니다. 다시 질문해 주세요.", "status": "ERROR"}


def isCompleted(structured_command):
    """
    유효한 구조화 명령을 시뮬레이션 엔진에 전달하고 최종 완료 피드백을 생성합니다.
    """
    
    # 1. 시뮬레이션 엔진 실행 (외부 시스템 함수 호출을 가정)
    try:
        # 이 함수는 실제 게임 로직(크레딧 차감, 토양 상태 변경 등)을 실행합니다.
        # 시뮬레이션 결과 데이터를 반환한다고 가정합니다.
        simulation_result = _run_game_simulation(structured_command)
    except Exception as e:
        return {"final_response": f"명령 실행 중 오류가 발생했습니다: {e}", "status": "ERROR"}

    # 2. 피드백 생성 (Reasoning/Feedback Generator 역할)
    # 실행 결과 데이터를 바탕으로 사용자에게 친절하고 교육적인 피드백을 생성합니다.
    contents = [{"role": "user", "parts": [{"text": f"실행된 명령: {structured_command}, 시뮬레이션 결과: {simulation_result}"}]}]
    
    # 'feedbackGeneratorPath' 모델 호출 (실행 피드백 모드로 가정)
    final_feedback = _call_gemini_model(feedbackGeneratorPath, contents)

    return {"final_response": final_feedback, "status": "COMPLETED"}

def _run_game_simulation(command):
    """
    실제 게임 엔진 로직을 시뮬레이션하는 더미 함수입니다.
    이 부분은 AI/데이터 분석 팀이 개발해야 할 영역입니다.
    """
    # 예시: 작물 심기가 성공했고, 다음 달 예상 물 소비량 증가를 시뮬레이션했다고 가정
    return {
        "command_status": "SUCCESS",
        "cost": 50,
        "credit_balance": 950,
        "next_month_water_stress_index": 0.25,
        "current_crop": command.get("crop")
    }


def eventHandler(user_input):
    """
    사용자 입력을 받아 단일 처리를 수행하고 최종 응답을 반환하는 메인 흐름 제어 함수입니다.
    """
    
    # 1. 입력 정규화 및 조치 개수 확인
    normalizedData = normalizeInput(user_input)
    action_count = normalizedData.get("action_count", 0)
    
    # 2. 질문 타입 확인
    sentenceType = determineQuestionType( normalizedData)
    user_type = sentenceType.get("type")
    
    normalized_text = normalizedData.get("normalized_text", user_input)
    
    # --- I. 명령 처리 ('I') ---
    if user_type == 'I':
        # 3. 단일 조치 제약 검토
        if action_count != 1:
            # 복수 조치 명령 -> 거부 응답
            return {
                "final_response": "잠깐만요. 한 턴에 수행할 수 있는 조치는 하나뿐이에요. 하나만 선택해서 다시 말씀해 주시겠어요?",
                "status": "VIOLATION"
            }
        
        # 4. 단일 조치 명령 분석 (procedureAnalyzer)
        structured_command = procedureAnalyzer(normalized_text)
        
        # 5. 과도한 위임/무효 요청 검토
        if structured_command and structured_command.get("intent") in ["위임_불가", "무효"]:
            # 위임 불가 명령에 대한 거부 응답
            return {
                "final_response": "죄송하지만, 제가 임의로 중요한 결정을 내릴 수 없어요. 구체적인 작물, 지역, 관개 방식 등을 지정해 주셔야 해요.",
                "status": "VIOLATION"
            }
        
        # 6. 유효한 단일 명령 실행 (isCompleted)
        return isCompleted(structured_command)
        
    # --- II. 질문 처리 ('Q') ---
    elif user_type == 'Q':
        # 7. 질문 처리 (questionHandler)
        return questionHandler(normalized_text)
        
    # --- III. 기타 발언 처리 ('O') ---
    else: # user_type == 'O'
        # 8. 사교적/무시 응답
        return {
            "final_response": "이해할 수 없습니다. 농업 시스템 관련해서 도움이 필요할 때 언제든 말씀해 주세요!",
            "status": "IGNORED"
        }

# --- 테스트 함수 ---

def test_event_handler():
    """테스트 문자열을 사용하여 eventHandler의 흐름을 확인하는 더미 테스트 함수."""
    print("--- Event Handler Test Simulation ---")
    
    # GEMINI_API_KEY 환경 변수가 설정되어 있어야 합니다.
    api_key = "AIzaSyCSCpQlWCX3ThJTNLzjDYPTBy8gWzqCBTQ"#os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    try:
        genai.configure(api_key=api_key)
        print("Gemini API configured successfully.")
    except Exception as e:
        print(f"Failed to configure Gemini API: {e}")
        return

    # 테스트 데이터 (action_count 및 type을 가정한 응답)
    test_inputs = [
        "비가 좀 더 오게 해",                       # 1. 'I' + 조치 1개 (비현실적 명령) -> procedureAnalyzer에서 필터링 가능
        "지금 여기서 뭐해야됨",                       # 2. 'Q'
        "과거 시점 A로 돌아가서 물 좀 더 대고 쌀도 심어줘", # 3. 'I' + 조치 2개 -> VIOLATION
        "이전 시점으로 좀 돌아가자",                 # 4. 'O'
        "강냉이 심어",                            # 5. 'I' + 조치 1개 -> COMPLETED
        "쌀이랑 옥수수 심어"                         # 6. 'I' + 조치 2개 -> VIOLATION
    ]
    
    print("\n[주의: 이 테스트는 JSON 설정 파일의 유효성과 실제 API 응답에 따라 결과가 달라집니다.]")

    for i, item in enumerate(test_inputs):
        print(f"\n--- Test {i+1}: Input: '{item}' ---")
        # 실제 API 호출이 포함되므로, 시간이 소요될 수 있습니다.
        result = eventHandler(item)
        print(f"Final Result: {json.dumps(result, indent=4, ensure_ascii=False)}")

def start_interactive_mode():
    """사용자와 직접 상호작용하는 메인 함수."""
    """
    print("--- 농업 시뮬레이션 게임 AI 비서 ---")
    print("안녕하세요! 무엇을 도와드릴까요?")
    print("(게임을 종료하려면 '종료' 또는 'exit'를 입력하세요)")
    """

    api_key = "AIzaSyCPbFbFjbAeAbFWkCIeLTnOPy8DQ4YxAvc"
    if not api_key:
        print("\n[오류] GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("게임을 시작하기 전에 API 키를 설정해주세요.")
        return
    
    try:
        genai.configure(api_key=api_key)
        print("\nGemini API가 성공적으로 연결되었습니다.")
    except Exception as e:
        print(f"\n[오류] Gemini API 연결에 실패했습니다: {e}")
        return

    while True:
        try:
            # 1. 사용자 입력 받기
            user_input = input("\n> 나: ")
            
            # 2. 종료 조건 확인
            if user_input.lower() in ["종료", "exit", "quit"]:
                print("게임을 종료합니다. 이용해주셔서 감사합니다.")
                break
            
            # 3. eventHandler로 입력 처리
            result = eventHandler(user_input)
            
            # 4. AI 응답 출력
            ai_response = result.get("final_response", "오류가 발생했습니다.")
            try:
                # 먼저 유니코드 이스케이프 형식이라고 가정하고 디코딩을 시도합니다.
                decoded_response = ai_response.encode('latin1').decode('unicode_escape')
            except UnicodeEncodeError:
                # 만약 위에서 오류가 발생하면 (이미 정상적인 한글이라는 뜻이므로),
                # 아무 처리 없이 원본 응답을 그대로 사용합니다.
                decoded_response = ai_response

            print(f"💬 AI 비서: {decoded_response}")

        except KeyboardInterrupt: # Ctrl+C 입력 시 종료
            print("\n게임을 강제 종료합니다.")
            break
        except Exception as e:
            print(f"[알 수 없는 오류 발생]: {e}")

if __name__ == "__main__":
    # 로컬에서만 인터랙티브 테스트할 때 수동으로 켜세요
    start_interactive_mode()
    # 또는 test_event_handler()
