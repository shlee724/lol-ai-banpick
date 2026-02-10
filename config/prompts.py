# config/prompts.py
import json

PICKED_CHAMPS_WITH_ROLES_PROMPT = """
You are analyzing League of Legends champion select.
In this image, my team has FIXED roles (TOP, JUNGLE, MID, ADC, SUPPORT).
The enemy team roles are NOT fixed / unknown.

Task:
1) Identify my team picked champions AND map them to the correct role.
2) Identify enemy team picked champions as a list of up to 5 champions (order does not matter).
3) If a slot is empty or uncertain, use null.

Return ONLY valid JSON (no markdown, no extra text).

Output JSON schema:
{
  "my_team": {
    "top": string|null,
    "jungle": string|null,
    "mid": string|null,
    "adc": string|null,
    "support": string|null
  },
  "enemy_team": [string|null, string|null, string|null, string|null, string|null],
  "notes": string
}

Rules:
- Use official champion English names if possible (e.g., "Lee Sin", "Ahri").
- Only output JSON.
""".strip()



BANNED_CHAMPS_PROMPT = """
You are analyzing League of Legends champion select.
Identify banned champions for both teams from this image.

Output JSON schema:
{
  "my_team_bans": [string|null, string|null, string|null, string|null, string|null],
  "enemy_team_bans": [string|null, string|null, string|null, string|null, string|null],
  "notes": string
}

Rules:
- Use official champion English names if possible.
- If unsure, use null.
""".strip()

BANNED_CHAMPS_10_PROMPT = """
You are analyzing League of Legends champion select.
This image shows 10 banned champion portraits in a single row (combined list).
Do NOT try to split by team. Just identify the banned champions in order from left to right.

Return ONLY valid JSON (no markdown, no extra text).

Output JSON schema:
{
  "bans": [string|null, string|null, string|null, string|null, string|null,
           string|null, string|null, string|null, string|null, string|null],
  "notes": string
}

Rules:
- Use official champion English names if possible (e.g., "Lee Sin", "Ahri").
- If a slot is empty or uncertain, use null.
- Keep the order left → right.
""".strip()

DRAFT_RECOMMEND_PROMPT_TEMPLATE = """
You are a League of Legends draft assistant.

User profile:
- My fixed role: {my_role}
- My tier: {my_tier}
- My champion pool (preferred picks for my role): {my_champ_pool}

Current draft state:
- My team picks (fixed roles):
  TOP: {top}
  JUNGLE: {jungle}
  MID: {mid}
  ADC: {adc}
  SUPPORT: {support}

- Enemy team picks (roles unknown): {enemy_picks}

- Banned champions (10 combined, left to right): {bans_10}

Task:
1) Recommend the next pick for ME (my fixed role) based on the current draft state and bans.
2) Only suggest champions from my champion pool if possible. If none are good/available, suggest outside pool but explain briefly.
3) Do NOT suggest bans.

Return ONLY valid JSON (no markdown, no extra text).

Output JSON schema:
{
  "my_role": string,
  "recommendations": [
    {
      "champion": string,
      "confidence": number,
      "reason": string
    },
    {
      "champion": string,
      "confidence": number,
      "reason": string
    },
    {
      "champion": string,
      "confidence": number,
      "reason": string
    }
  ],
  "notes": string
}
""".strip()


def build_draft_recommend_prompt(
    *,
    my_role: str,
    my_tier: str,
    my_champ_pool: list[str],
    my_team: dict,
    enemy_picks: list,
    bans_10: list,
) -> str:
    my_team_json = json.dumps(my_team, ensure_ascii=False)
    enemy_json = json.dumps(enemy_picks, ensure_ascii=False)
    bans_json = json.dumps(bans_10, ensure_ascii=False)
    pool_json = json.dumps(my_champ_pool, ensure_ascii=False)

    return f"""
너는 **리그 오브 레전드 한국 서버(KR) 기준** 밴픽 도우미야.
아래 정보들을 보고, **내 고정 포지션**에서 다음 픽 추천을 해줘.

[유저 정보]
- 내 고정 포지션: {my_role}
- 내 티어(한국 서버 기준): {my_tier}
- 내 챔프폭(가능하면 여기서 우선 추천): {pool_json}

[현재 밴픽 상황]
- 우리팀 픽(포지션 고정): {my_team_json}
- 상대팀 픽(포지션 미확정): {enemy_json}
- 밴된 챔피언 10개(왼→오, 팀 구분 없음): {bans_json}

[요구사항]
1) **추천은 내 포지션({my_role})의 챔피언**만 해줘.
2) 가능한 한 **내 챔프폭**에서 추천해줘. (없거나 밴/픽으로 불가능하면 챔프폭 밖도 가능)
3) **밴 추천은 절대 하지 마.**
4) 설명(reason, notes)은 **반드시 한글**로 작성해.
5) 챔피언 이름은 데이터 처리 편의를 위해 **공식 영문명**으로 출력해.

반환은 JSON만. 마크다운 금지. 추가 텍스트 금지.

출력 JSON 스키마:
{{
  "my_role": string,
  "recommendations": [
    {{
      "champion": string,
      "confidence": number,
      "reason_kr": string
    }},
    {{
      "champion": string,
      "confidence": number,
      "reason_kr": string
    }},
    {{
      "champion": string,
      "confidence": number,
      "reason_kr": string
    }}
  ],
  "notes_kr": string
}}
""".strip()
