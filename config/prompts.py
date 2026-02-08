# config/prompts.py

# config/prompts.py

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


DRAFT_ADVICE_PROMPT_TEMPLATE = """
We are in LoL draft.

My team (fixed roles):
- TOP: {top}
- JUNGLE: {jungle}
- MID: {mid}
- ADC: {adc}
- SUPPORT: {support}

Enemy team picks (roles unknown): {enemy_picks}

My team bans: {my_bans}
Enemy team bans: {enemy_bans}

Give me draft advice:
- suggested next pick (top 3) with role recommendation
- suggested ban (top 3)
- short reasoning

Return ONLY valid JSON. No markdown.
""".strip()



def build_draft_advice_prompt(
    *,
    my_picks,
    enemy_picks,
    my_bans=None,
    enemy_bans=None,
) -> str:
    return DRAFT_ADVICE_PROMPT_TEMPLATE.format(
        my_picks=my_picks,
        enemy_picks=enemy_picks,
        my_bans=my_bans or [],
        enemy_bans=enemy_bans or [],
    )
