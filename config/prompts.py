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

DRAFT_FROM_IMAGE_PROMPT_LITE = """
LoL KR draft from image. Output ONLY JSON.

Step1: Read picks from image:
- my_team roles fixed: top,jungle,mid,adc,support (use null if empty)
- enemy_team: up to 5 champs list (null if empty)

Step2: Recommend my next pick for role={my_role}, tier={my_tier}
- prefer pool: {pool_json}
- practice allowed: ["Akali","Sion","Tryndamere","Ornn"]
- no ban advice
- champion must be official English
- reason_kr Korean <= 80 chars

Return JSON:
{{
  "my_team": {{"top":null,"jungle":null,"mid":null,"adc":null,"support":null}},
  "enemy_team": [null,null,null,null,null],
  "reco": [{{"c":"","r":""}},{{"c":"","r":""}},{{"c":"","r":""}}]
}}
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
- 연습하고 있는 챔피언: ["Akali", "Sion", "Tryndamere", "Ornn"]

[현재 밴픽 상황]
- 우리팀 픽(포지션 고정): {my_team_json}
- 상대팀 픽(포지션 미확정): {enemy_json}
- 밴된 챔피언 10개(왼→오, 팀 구분 없음): {bans_json}

[요구사항]
1) 추천은 **내 챔프폭**에서 우선적으로 해주되, 상대 라이너와의 상성이나 전체적인 조합을 고려하여
**연습하고 있는 챔피언** 풀에 있는 챔피언이 더 좋다면 그것을 해줘. (밴/픽으로 불가능하면 챔프폭 밖도 가능)
2) 가능하면 유력한 상대 라이너와의 라인전 상성을 우선 고려해주고,
내가 미드 라이너라면 두번째로 우리팀과 상대팀 미드정글 조합을 고려, 세번째로 전체 조합을 고려
4) **밴 추천은 절대 하지 마.**
5) 설명(reason, notes)은 **반드시 한글**로 작성해.
6) 챔피언 이름은 데이터 처리 편의를 위해 **공식 영문명**으로 출력해.

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


def build_draft_recommend_prompt_lite(
    *,
    my_role: str,
    my_tier: str,
    my_champ_pool: list[str],
    my_team: dict,
    enemy_picks: list,
    bans_10: list,
) -> str:
    return (
        "KR LoL draft. ONLY JSON.\n"
        f"role={my_role} tier={my_tier}\n"
        f"pool={json.dumps(my_champ_pool, ensure_ascii=False)} practice=[Akali,Sion,Tryndamere,Ornn]\n"
        f"our={json.dumps(my_team, ensure_ascii=False)} enemy={json.dumps(enemy_picks, ensure_ascii=False)} bans={json.dumps(bans_10, ensure_ascii=False)}\n"
        "Pick for my role. Prefer pool, else practice, else any available.\n"
        "No ban advice. champion must be official English. reason_kr Korean <= 80 chars.\n"
        'Return: {"my_role":str,"reco":[{"c":str,"r":str},{"c":str,"r":str},{"c":str,"r":str}]}'
    )
