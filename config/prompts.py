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
    # 안전하게 문자열화 (따옴표/특수문자/None 등)
    my_team_json = json.dumps(my_team, ensure_ascii=False)
    enemy_json = json.dumps(enemy_picks, ensure_ascii=False)
    bans_json = json.dumps(bans_10, ensure_ascii=False)
    pool_json = json.dumps(my_champ_pool, ensure_ascii=False)

    return f"""
You are a League of Legends draft assistant.

User profile:
- My fixed role: {my_role}
- My tier: {my_tier}
- My champion pool (preferred picks for my role): {pool_json}

Current draft state:
- My team picks (fixed roles): {my_team_json}
- Enemy team picks (roles unknown): {enemy_json}
- Banned champions (10 combined, left to right): {bans_json}

Task:
1) Recommend the next pick for ME (my fixed role) based on the current draft state and bans.
2) Prefer champions from my champion pool if possible. If none are good/available, you may suggest outside the pool but explain briefly.
3) Do NOT suggest bans.

Return ONLY valid JSON (no markdown, no extra text).

Output JSON schema:
{{
  "my_role": string,
  "recommendations": [
    {{
      "champion": string,
      "confidence": number,
      "reason": string
    }},
    {{
      "champion": string,
      "confidence": number,
      "reason": string
    }},
    {{
      "champion": string,
      "confidence": number,
      "reason": string
    }}
  ],
  "notes": string
}}
""".strip()