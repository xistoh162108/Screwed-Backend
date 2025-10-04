using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Serialization;

// ---------- 요청/응답 DTO ----------
[Serializable]
public class TurnCreate {
    public string parent_id;   // 루트 만들면 null/빈 문자열
    public string branch_id;   // null이면 서버 기본(b_main) 혹은 상속
    public string month;       // "YYYY-MM"
    public string state;       // "DRAFT" | "SIMULATED" 등
    public Stats stats;        // 자유 JSON 구조
}

[Serializable]
public class Stats {
    public Dictionary<string, object> climate;
    [JsonProperty("yield")] public Dictionary<string, object> yieldDict; // "yield" 키 그대로
    public Dictionary<string, object> env;
    public Dictionary<string, object> money;
    public List<string> notes;
}

[Serializable]
public class TurnOut {
    public string id;
    public string parent_id;
    public string branch_id;
    public string month;
    public string state;
    public string created_at;
    public string updated_at;
    public List<string> children;
}

[Serializable]
public class CommandCreate {
    public string text;
}

[Serializable]
public class CommandOut {
    public string id;
    public string turn_id;
    public string text;
    public object validity;
    public object cost;
    public string created_at;
}

// ---------- 메인 컴포넌트 ----------
public class DataManager : MonoBehaviour
{
    [Header("UI")]
    public Text outputText;
    public InputField inputField;

    [Header("Server")]
    // 예: FastAPI: app.main에서 docs가 /api/v1/docs 이면 baseUrl은 /api/v1
    public string baseUrl = "http://34.50.58.237/api/v1";

    // 현재 작업 중인 Turn id
    public string currentTurnId = "";

    void Start()
    {
        // 필요하면 초기화 출력
        LogLine("Ready. Q=Create Turn, Space=Send Command");
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Q))
        {
            // 새 Turn 생성(루트)
            StartCoroutine(CreateTurn());
        }

        if (Input.GetKeyDown(KeyCode.Space))
        {
            if (string.IsNullOrEmpty(currentTurnId))
            {
                LogLine("No currentTurnId. Press Q to create a Turn first.");
                return;
            }
            var txt = inputField != null ? inputField.text : "";
            StartCoroutine(PostCommand(currentTurnId, txt));
        }
    }

    // ---------- Turn 생성 ----------
    IEnumerator CreateTurn()
    {
        // 요청 바디 구성
        var body = new TurnCreate {
            parent_id = null,          // 루트 자동 보장(우리 서버 로직대로)
            branch_id = null,          // null => b_main (또는 상속)
            month = DateTime.UtcNow.ToString("yyyy-MM"),
            state = "DRAFT",
            stats = new Stats {
                climate = new Dictionary<string, object> {
                    {"rainfall_mm", 83f}, {"temp_c", 25.4f}
                },
                yieldDict = new Dictionary<string, object> {
                    {"wheat_ton", 12.3f}, {"corn_ton", 9.8f}
                },
                env = new Dictionary<string, object> {
                    {"co2_ton", 1.2f}, {"soil_quality", 0.87f}
                },
                money = new Dictionary<string, object> {
                    {"balance", 200000}, {"currency", "KRW"}
                },
                notes = new List<string> { "pest risk low" }
            }
        };

        string json = JsonConvert.SerializeObject(body, Formatting.None,
            new JsonSerializerSettings {
                NullValueHandling = NullValueHandling.Ignore,
                ContractResolver = new CamelCasePropertyNamesContractResolver()
            });

        var url = baseUrl.TrimEnd('/') + "/turns";
        yield return PostJson(url, json,
            onSuccess: (resp) => {
                LogLine($"CreateTurn OK: {resp}");
                var turn = JsonConvert.DeserializeObject<TurnOut>(resp);
                if (turn != null && !string.IsNullOrEmpty(turn.id)) {
                    currentTurnId = turn.id;
                    LogLine($"currentTurnId = {currentTurnId}");
                } else {
                    LogLine("CreateTurn: cannot parse id from response");
                }
            },
            onError: (msg, code, bodyText) => {
                LogError($"CreateTurn FAIL ({code}): {msg}\n{bodyText}");
            }
        );
    }

    // ---------- Command 전송 ----------
    IEnumerator PostCommand(string turnId, string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            LogLine("Empty command text.");
            yield break;
        }

        var body = new CommandCreate { text = text };
        string json = JsonConvert.SerializeObject(body, Formatting.None,
            new JsonSerializerSettings {
                NullValueHandling = NullValueHandling.Ignore,
                ContractResolver = new CamelCasePropertyNamesContractResolver()
            });

        var url = $"{baseUrl.TrimEnd('/')}/turns/{turnId}/commands";
        yield return PostJson(url, json,
            onSuccess: (resp) => {
                LogLine($"PostCommand OK: {resp}");
                // 필요 시 응답 파싱
                // var cmd = JsonConvert.DeserializeObject<CommandOut>(resp);
            },
            onError: (msg, code, bodyText) => {
                LogError($"PostCommand FAIL ({code}): {msg}\n{bodyText}");
            }
        );
    }

    // ---------- 공통 POST(JSON) ----------
    IEnumerator PostJson(string url, string json,
        Action<string> onSuccess,
        Action<string, long, string> onError)
    {
        byte[] payload = Encoding.UTF8.GetBytes(json);
        using (var req = new UnityWebRequest(url, UnityWebRequest.kHttpVerbPOST))
        {
            req.uploadHandler = new UploadHandlerRaw(payload);
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");
            // 토큰이 있으면:
            // req.SetRequestHeader("Authorization", "Bearer " + token);
            req.timeout = 15;

            LogLine($"POST {url}\n{json}");
            yield return req.SendWebRequest();

            if (req.result == UnityWebRequest.Result.Success ||
                (req.responseCode >= 200 && req.responseCode < 300))
            {
                onSuccess?.Invoke(req.downloadHandler.text);
            }
            else
            {
                onError?.Invoke(req.error, req.responseCode, req.downloadHandler.text);
            }
        }
    }

    // ---------- 로그 도우미 ----------
    void LogLine(string msg)
    {
        Debug.Log(msg);
        if (outputText != null)
        {
            outputText.text = (outputText.text + "\n" + msg);
        }
    }

    void LogError(string msg)
    {
        Debug.LogError(msg);
        if (outputText != null)
        {
            outputText.text = (outputText.text + "\n<color=red>" + msg + "</color>");
        }
    }
}