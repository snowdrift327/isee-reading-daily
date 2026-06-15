"""
ISEE Reading Daily — Batch Generator
Generates N articles at once, saves to articles.json, supports resume on interruption.

Usage:
  uv run main.py        → generate 50 articles (default)
  uv run main.py 30     → generate 30 articles
  uv run main.py reset  → clear articles.json and start over
"""

import os
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

# ============ Configuration ============
load_dotenv()
client = Anthropic()

TOPICS_FILE = Path("topics.json")
ARTICLES_FILE = Path("articles.json")
INDEX_FILE = Path("index.html")
HISTORY_PAGE = Path("history.html")
MISTAKES_FILE_HTML = Path("mistakes.html")

DEFAULT_BATCH_SIZE = 50

GENRES = [
    {
        "name": "narrative",
        "label": "Narrative Nonfiction",
        "guidance": "Tell the story like a writer — use scenes, a sense of time passing, and a specific moment or turning point."
    },
    {
        "name": "informational",
        "label": "Informational",
        "guidance": "Explain clearly with facts, examples, and structure. Use cause-and-effect, compare/contrast, or sequence."
    },
    {
        "name": "argumentative",
        "label": "Argumentative",
        "guidance": "Present a clear position or claim, then support it with 2-3 specific reasons grounded in facts."
    },
    {
        "name": "descriptive",
        "label": "Descriptive",
        "guidance": "Use vivid sensory details (sight, sound, smell, texture, motion) to paint the topic."
    },
]

Q5_TYPES = ["organization", "tone", "advanced_detail"]

Q5_INSTRUCTIONS = {
    "organization": """Q5 (Organization): MUST use one of these formats:
   - "Which best describes how the passage is organized?"
   - "How does the second paragraph relate to the first?"
   - "The author begins the passage by..."

   Options MUST describe structural patterns like:
   - "presenting a problem, then a solution"
   - "describing events in chronological order"
   - "comparing two related events"
   - "stating a claim, then supporting it with examples"
   - "introducing a person, then describing their key achievement" """,

    "tone": """Q5 (Tone/Attitude): MUST use one of these formats:
   - "The author's attitude toward [X] can best be described as..."
   - "The tone of the passage is mainly..."
   - "Which word best describes how the author feels about [X]?"

   Options MUST be specific tone adjectives. Choose 4 from:
   admiring, neutral, skeptical, informative, cautionary, reverent,
   sympathetic, critical, enthusiastic, contemplative, ironic,
   curious, appreciative, concerned, matter-of-fact """,

    "advanced_detail": """Q5 (Advanced Detail / Synthesis): MUST use one of these formats:
   - "Which statement best summarizes the relationship between X and Y?"
   - "The author's description of [thing] emphasizes its..."
   - "Which of the following BEST captures the significance of [X]?"
   Requires connecting multiple pieces of information across the passage."""
}


# ============ Helpers ============
def parse_ai_json(prompt, max_tokens=4096, max_retries=3, label=""):
    """Call Claude and parse JSON, with retries on failure."""
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            return json.loads(text)
        except json.JSONDecodeError as e:
            last_error = e
            print(f"   ⚠️  {label} attempt {attempt+1}/{max_retries}: JSON parse failed (line {e.lineno}, col {e.colno})")
            if attempt < max_retries - 1:
                print(f"   🔄 Retrying...")
            continue
    raise RuntimeError(f"{label} failed after {max_retries} attempts. Last error: {last_error}")


def parse_vocab(passage):
    """Extract <vocab>word|def</vocab> markers; return HTML-rendered passage and vocab list."""
    pattern = r'<vocab>(.*?)\|(.*?)</vocab>'
    matches = re.findall(pattern, passage)
    vocab_list = [{"word": m[0].strip(), "definition": m[1].strip()} for m in matches]

    def replace_func(match):
        word, definition = match.group(1).strip(), match.group(2).strip()
        safe_def = definition.replace('"', '&quot;')
        return f'<span class="vocab" data-def="{safe_def}">{word}</span>'

    html_passage = re.sub(pattern, replace_func, passage)
    return html_passage, vocab_list


def format_passage_html(passage_text):
    """Wrap paragraphs in <p> tags."""
    paragraphs = re.split(r'\n\n+', passage_text.strip())
    if len(paragraphs) == 1:
        paragraphs = re.split(r'\n', passage_text.strip())
    return "\n".join(f"<p>{p.strip()}</p>" for p in paragraphs if p.strip())


# ============ Core: Generate One Article ============
def generate_one_article(topic, genre, length_label, word_range, q5_type, used_vocab_recent):
    """Generate a single article (two-step) and return as dict."""

    # === Step 1: Research ===
    research_prompt = f"""You are a meticulous research assistant. Provide 6-8 verifiable, specific facts about this topic, suitable as the factual basis for an educational reading passage for advanced 5th-6th grade students.

TOPIC: {topic['title']}

REQUIREMENTS:
- Each fact must include specific elements: names, dates, places, numbers, or measurable details
- Avoid vague qualifiers ("many", "some", "often")
- Avoid invented quotations; only widely documented ones
- Each fact: 1-2 sentences

Output STRICT JSON only:
{{
  "facts": ["Fact 1 with specifics", "Fact 2 with specifics", "..."]
}}

Generate 6-8 facts. If uncertain about a detail, omit it."""

    research_result = parse_ai_json(research_prompt, max_tokens=2048, label="Research")
    facts = research_result["facts"]

    # === Step 2: Writing ===
    q5_instructions = Q5_INSTRUCTIONS[q5_type]

    avoid_vocab_text = ""
    if used_vocab_recent:
        avoid_vocab_text = f"""

VOCABULARY EXCLUSION LIST (DO NOT use in <vocab> markers):
{', '.join(used_vocab_recent)}
Choose DIFFERENT challenging words."""

    writing_prompt = f"""You are a senior editor at a publication like Newsela or National Geographic Kids, writing for advanced 10-11 year old readers (ISEE Lower Level preparation).

═══════════════════════════════════════
TOPIC: {topic['title']}
GENRE: {genre['label']}
GENRE GUIDANCE: {genre['guidance']}
LENGTH TARGET: {word_range} words
READING LEVEL: Lexile 1000-1100
═══════════════════════════════════════
{avoid_vocab_text}

VERIFIED FACTS:
{json.dumps(facts, indent=2)}

═══════════════════════════════════════
QUALITY REQUIREMENTS — strict:

1. STRUCTURE
   - Open with a SPECIFIC concrete hook (a moment, image, detail)
   - End with a thought-provoking conclusion
   - FORBIDDEN openings: "Have you ever wondered...", "Long ago...", "Once upon..."
   - FORBIDDEN closings: "In conclusion...", "And that is why..."

2. CONTENT
   - Anchor every paragraph in at least one fact above
   - Include 3+ concrete details: names, dates, numbers, places
   - Include 2+ sensory details (sight, sound, motion)
   - For narrative: include a turning point
   - For argumentative: claim + 2 supporting reasons
   - For descriptive: vivid sensory language
   - For informational: clear structure

3. STYLE
   - Vary sentence length
   - Use precise verbs (not "did", "made", "got")
   - Avoid filler ("very", "really", "a lot of")
   - Show, don't tell

4. VOCABULARY EMBEDDING (CRITICAL)
   - Include 5-7 challenging vocabulary words at ISEE Lower Level
   - Mark with: <vocab>WORD|simple English definition</vocab>
   - Example: "The team faced <vocab>perilous|extremely dangerous</vocab> weather."
   - 5-7 markers, no more, no less

5. FORBIDDEN
   - Forbidden phrases listed above
   - Vague qualifiers ("many", "some", "often")
   - Made-up quotations not in the facts

═══════════════════════════════════════
QUESTIONS — 5 multiple choice:

Q1 (Main Idea): One of:
   - "Which sentence best states the central idea of the passage?"
   - "The primary purpose of this passage is to..."
   - "Which would be the BEST title for this passage?"

Q2 (Detail - High Rigor): MUST use one of:
   - "Which detail from the passage BEST supports the idea that..."
   - "The author mentions [X] in order to..."
   - "Which of the following is NOT mentioned in the passage as..."
   - "According to the passage, which is true about..."
   AVOID simple lookup like "What year did X happen?"

Q3 (Inference): "The passage suggests that..." or "We can conclude that..."

Q4 (Vocabulary in Context): "In the passage, the word \\"___\\" most nearly means..."

{q5_instructions}

For each question:
- 4 options
- Exactly 1 correct
- 3 plausible distractors following these RIGOR rules:
  * Type A: TRUE in real world but NOT stated in passage
  * Type B: PARTIALLY true (correct premise, wrong conclusion/scope)
  * Type C: USES passage words but rearranges meaning
- Each question MUST include at least 2 of the 3 distractor types
- VARY correct answer positions across the 5 questions
- Include 1-sentence explanation

═══════════════════════════════════════
OUTPUT FORMAT — STRICT JSON, no markdown wrappers:

{{
  "title": "Engaging 5-8 word title",
  "passage": "Passage with <vocab>WORD|def</vocab> markers inline.",
  "questions": [
    {{"type": "main_idea", "stem": "...", "options": ["...","...","...","..."], "correct_index": 0, "explanation": "..."}},
    {{"type": "detail", "stem": "...", "options": [...], "correct_index": 1, "explanation": "..."}},
    {{"type": "inference", "stem": "...", "options": [...], "correct_index": 2, "explanation": "..."}},
    {{"type": "vocab_in_context", "stem": "...", "options": [...], "correct_index": 3, "explanation": "..."}},
    {{"type": "{q5_type}", "stem": "...", "options": [...], "correct_index": 0, "explanation": "..."}}
  ]
}}"""

    result = parse_ai_json(writing_prompt, max_tokens=8192, label="Writing")
    html_passage, vocab_list = parse_vocab(result["passage"])

    return {
        "id": f"article_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
        "topic_id": topic["id"],
        "topic_title": topic["title"],
        "category": topic["category"],
        "is_china": topic.get("china", False),
        "is_challenge": topic["challenge"],
        "genre": genre["label"],
        "length": length_label,
        "q5_type": q5_type,
        "title": result["title"],
        "passage_html": html_passage,
        "passage_raw": result["passage"],
        "vocab": vocab_list,
        "questions": result["questions"]
    }


# ============ Load/Save Articles Database ============
def load_articles():
    if ARTICLES_FILE.exists():
        with open(ARTICLES_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"generated_at": None, "total": 0, "articles": []}


def save_articles(data):
    data["total"] = len(data["articles"])
    data["last_updated"] = datetime.now().isoformat()
    with open(ARTICLES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============ Main Batch Generation ============
def run_batch(target_count):
    # Load topics
    with open(TOPICS_FILE, encoding="utf-8") as f:
        all_topics = json.load(f)["topics"]

    # Load existing
    db = load_articles()
    existing_articles = db.get("articles", [])
    used_topic_ids = [a["topic_id"] for a in existing_articles]
    used_vocab = []
    for a in existing_articles[-10:]:  # recent 10 articles' vocab
        used_vocab.extend([v["word"].lower() for v in a.get("vocab", [])])

    already_have = len(existing_articles)
    to_generate = target_count - already_have

    if to_generate <= 0:
        print(f"✅ Already have {already_have} articles. To start over, run: uv run main.py reset")
        return existing_articles

    # Available topics
    available_topics = [t for t in all_topics if t["id"] not in used_topic_ids]
    if len(available_topics) < to_generate:
        print(f"⚠️  Only {len(available_topics)} unused topics left; will generate that many.")
        to_generate = len(available_topics)

    random.shuffle(available_topics)
    chosen_topics = available_topics[:to_generate]

    # Build balanced distributions
    genres_cycle = []
    while len(genres_cycle) < to_generate:
        genres_cycle.extend(GENRES)
    genres_cycle = genres_cycle[:to_generate]
    random.shuffle(genres_cycle)

    lengths_cycle = (["short"] * (to_generate // 2 + 1) + ["long"] * (to_generate // 2 + 1))[:to_generate]
    random.shuffle(lengths_cycle)

    q5_cycle = []
    while len(q5_cycle) < to_generate:
        q5_cycle.extend(Q5_TYPES)
    q5_cycle = q5_cycle[:to_generate]
    random.shuffle(q5_cycle)

    print(f"\n📚 Generating {to_generate} new articles (currently have {already_have})")
    print(f"⏱  Estimated time: {to_generate * 35 // 60}-{to_generate * 60 // 60} minutes")
    print(f"💰 Estimated cost: ${to_generate * 0.08:.2f}-${to_generate * 0.12:.2f}\n")

    if db.get("generated_at") is None:
        db["generated_at"] = datetime.now().isoformat()

    failed = []
    for i, topic in enumerate(chosen_topics):
        genre = genres_cycle[i]
        length_choice = lengths_cycle[i]
        if length_choice == "short":
            length_label, word_range = "Short", "200-250"
        else:
            length_label, word_range = "Long", "350-450"
        q5_type = q5_cycle[i]

        progress = f"[{already_have + i + 1}/{target_count}]"
        china_flag = " 🇨🇳" if topic.get("china") else ""
        challenge_flag = " ⭐" if topic["challenge"] else ""
        print(f"{progress} {topic['title'][:50]}{china_flag}{challenge_flag}")
        print(f"    {genre['label']} | {length_label} | Q5={q5_type}")

        try:
            article = generate_one_article(
                topic, genre, length_label, word_range, q5_type,
                used_vocab_recent=used_vocab[-40:]  # avoid recent 40 vocab words
            )
            existing_articles.append(article)
            used_vocab.extend([v["word"].lower() for v in article["vocab"]])
            db["articles"] = existing_articles
            save_articles(db)
            print(f"    ✓ {article['title']}\n")
        except Exception as e:
            failed.append((topic["title"], str(e)))
            print(f"    ❌ Failed: {e}\n")
            continue

    print(f"\n{'=' * 60}")
    print(f"✅ Batch complete!")
    print(f"   Total articles: {len(existing_articles)}")
    if failed:
        print(f"   Failed: {len(failed)}")
        for title, err in failed:
            print(f"     - {title[:50]}: {err[:80]}")
    print(f"{'=' * 60}\n")

    return existing_articles


# ============ HTML Generators ============
def render_index_html(articles_count):
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Daily Reading — ISEE Practice</title>
<style>
* {{ box-sizing: border-box; }}
body {{
    font-family: "Georgia", "Times New Roman", serif;
    max-width: 760px;
    margin: 0 auto;
    padding: 24px 22px 80px;
    background: #fafaf7;
    color: #1a1a1a;
    line-height: 1.75;
    font-size: 17px;
}}
.start-overlay {{
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(250, 250, 247, 0.98); z-index: 9999;
    display: flex; align-items: center; justify-content: center;
    backdrop-filter: blur(8px);
}}
.start-overlay.hidden {{ display: none; }}
.start-card {{
    background: white; max-width: 500px; width: 90%;
    padding: 40px 32px; border: 3px double #1a4d8f;
    border-radius: 8px; text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.15);
}}
.start-card h2 {{ font-size: 26px; color: #1a4d8f; margin: 0 0 8px; }}
.reading-title {{ color: #333; font-size: 17px; font-style: italic; margin: 16px 0 8px; line-height: 1.4; }}
.badges {{ margin: 16px 0 24px; }}
.badge {{
    display: inline-block; font-size: 11px; padding: 3px 10px;
    border-radius: 12px; margin: 0 4px;
    font-family: "Helvetica Neue", Arial, sans-serif; letter-spacing: 0.5px;
}}
.badge.category {{ background: #e3edf8; color: #1a4d8f; }}
.badge.genre {{ background: #f0e6d8; color: #8b6914; }}
.badge.length {{ background: #e8f0e3; color: #2d5a1f; }}
.badge.china {{ background: #fde8e8; color: #b71c1c; }}
.badge.challenge {{ background: #fff4e0; color: #e67e22; }}
.instructions {{
    text-align: left; background: #f5f5f0; padding: 16px 20px;
    border-radius: 6px; margin: 20px 0 28px;
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 14px; line-height: 1.7;
}}
.instructions strong {{
    color: #1a4d8f; display: block; margin-bottom: 6px;
    font-size: 13px; text-transform: uppercase; letter-spacing: 1px;
}}
.instructions ul {{ margin: 0; padding-left: 20px; }}
.start-btn {{
    background: #1a4d8f; color: white; border: none;
    padding: 16px 56px; font-size: 17px; font-weight: bold;
    border-radius: 4px; cursor: pointer;
    font-family: "Helvetica Neue", Arial, sans-serif;
    letter-spacing: 2px;
}}
.start-btn:hover {{ background: #143a6b; }}
.reading-header {{ border-bottom: 3px double #333; padding-bottom: 20px; margin-bottom: 28px; }}
.reading-header h1 {{ font-size: 28px; margin: 0 0 12px; line-height: 1.3; }}
.meta-bar {{
    display: flex; justify-content: space-between; align-items: center;
    margin-top: 14px; padding: 10px 16px; background: white;
    border: 1px solid #ddd; border-radius: 4px;
    font-family: "Helvetica Neue", Arial, sans-serif; font-size: 14px;
}}
.timer {{ font-family: "Courier New", monospace; font-size: 17px; font-weight: bold; color: #1a4d8f; }}
.passage-section {{
    background: white; padding: 28px 32px; border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 32px;
    border-left: 4px solid #1a4d8f;
}}
.passage-section p {{ margin: 0 0 1em; text-align: justify; }}
.vocab {{
    border-bottom: 2px dotted #1a4d8f; cursor: help;
    color: #1a4d8f; font-weight: 500; position: relative;
}}
.vocab:hover::after {{
    content: attr(data-def); position: absolute;
    bottom: 100%; left: 50%; transform: translateX(-50%);
    background: #1a4d8f; color: white; padding: 8px 12px;
    border-radius: 4px; font-size: 13px; font-weight: normal;
    font-family: "Helvetica Neue", Arial, sans-serif;
    max-width: 260px; width: max-content; z-index: 1000;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2); margin-bottom: 6px;
    pointer-events: none;
}}
.vocab:hover::before {{
    content: ""; position: absolute; bottom: 100%; left: 50%;
    transform: translateX(-50%); border: 6px solid transparent;
    border-top-color: #1a4d8f; z-index: 1000; pointer-events: none;
}}
h2.section-title {{
    font-size: 20px; margin: 36px 0 18px; padding-left: 14px;
    border-left: 4px solid #1a4d8f;
}}
.question {{
    background: white; padding: 22px 24px; border-radius: 8px;
    margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
.q-header {{ margin-bottom: 12px; font-family: "Helvetica Neue", Arial, sans-serif; }}
.q-number {{ font-size: 18px; font-weight: bold; margin-right: 12px; }}
.q-type {{
    font-size: 11px; color: white; background: #1a4d8f;
    padding: 2px 8px; border-radius: 3px; text-transform: uppercase;
    letter-spacing: 0.5px; font-weight: 600;
}}
.q-stem {{ font-size: 16px; margin-bottom: 14px; line-height: 1.6; }}
.q-options {{ display: flex; flex-direction: column; gap: 6px; margin-left: 12px; font-family: "Helvetica Neue", Arial, sans-serif; }}
.option {{
    display: flex; align-items: flex-start; padding: 10px 14px;
    border: 1.5px solid #e0e0e0; border-radius: 5px;
    cursor: pointer; font-size: 15px;
}}
.option:hover {{ border-color: #1a4d8f; background: #f0f5fb; }}
.letter {{ font-weight: bold; color: #555; margin-right: 8px; min-width: 20px; }}
.option.user-correct {{ background: #e8f5e9; border-color: #4caf50; }}
.option.user-wrong {{ background: #ffebee; border-color: #f44336; }}
.option.show-correct {{ background: #e8f5e9; border-color: #4caf50; border-style: dashed; }}
.submit-section {{ margin-top: 32px; text-align: center; padding-top: 20px; border-top: 3px double #333; }}
#submit-btn {{
    background: #1a4d8f; color: white; border: none;
    padding: 14px 52px; font-size: 16px; font-weight: bold;
    border-radius: 4px; cursor: pointer;
    font-family: "Helvetica Neue", Arial, sans-serif;
    letter-spacing: 1px;
}}
#submit-btn:hover {{ background: #143a6b; }}
#submit-btn:disabled {{ background: #aaa; cursor: not-allowed; }}
#results {{
    display: none; margin-top: 30px; padding: 28px;
    background: #f9f9f7; border: 2px solid #1a4d8f;
    border-radius: 8px; font-family: "Helvetica Neue", Arial, sans-serif;
}}
#results.show {{ display: block; }}
.new-test-section {{ text-align: center; margin-bottom: 20px; padding-bottom: 18px; border-bottom: 1px solid #ddd; }}
.start-new-btn {{
    background: #2e7d32; color: white; border: none;
    padding: 12px 40px; font-size: 15px; font-weight: bold;
    border-radius: 6px; cursor: pointer; letter-spacing: 1px;
}}
.start-new-btn:hover {{ background: #1b5e20; }}
.score-summary {{
    display: flex; justify-content: space-around; text-align: center;
    margin-bottom: 24px; flex-wrap: wrap; gap: 14px;
}}
.score-item {{ flex: 1; min-width: 120px; }}
.score-value {{ font-size: 32px; font-weight: bold; color: #1a4d8f; display: block; }}
.score-label {{ font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }}
.explanation-block {{ background: white; padding: 14px 18px; margin-bottom: 12px; border-radius: 6px; border-left: 4px solid #4caf50; }}
.explanation-block.wrong {{ border-left-color: #f44336; }}
.exp-q {{ font-weight: 600; margin-bottom: 6px; font-size: 14px; }}
.exp-result {{ font-size: 13px; color: #555; margin-bottom: 8px; }}
.exp-text {{ font-size: 14px; color: #333; line-height: 1.6; }}
.action-links {{
    margin: 20px 0; padding: 16px 0;
    border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;
    display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
}}
.action-links a {{
    display: block; text-align: center; padding: 12px 16px;
    background: #1a4d8f; color: white; text-decoration: none;
    border-radius: 4px; font-size: 14px; font-weight: 500;
}}
.action-links a:hover {{ background: #143a6b; }}
.progress-pill {{
    background: white; padding: 6px 12px; border-radius: 12px;
    font-size: 12px; color: #555; border: 1px solid #ddd;
    font-family: Arial, sans-serif;
}}
footer {{ text-align: center; margin-top: 50px; color: #999; font-size: 13px; font-family: "Helvetica Neue", Arial, sans-serif; }}
footer a {{ color: #888; text-decoration: none; }}
.all-done {{ text-align: center; padding: 60px 20px; }}
.all-done h1 {{ color: #1a4d8f; font-size: 32px; }}
.all-done .actions {{ margin-top: 32px; display: flex; gap: 14px; justify-content: center; flex-wrap: wrap; }}
.all-done button {{
    background: #1a4d8f; color: white; border: none;
    padding: 14px 28px; font-size: 14px; font-weight: bold;
    border-radius: 6px; cursor: pointer;
    font-family: "Helvetica Neue", Arial, sans-serif;
}}
.all-done button.secondary {{ background: #6c757d; }}
@media (max-width: 600px) {{
    body {{ padding: 18px 14px 60px; font-size: 16px; }}
    .reading-header h1 {{ font-size: 22px; }}
    .passage-section {{ padding: 20px 18px; }}
    .question {{ padding: 18px 16px; }}
    .meta-bar {{ flex-direction: column; gap: 6px; align-items: flex-start; }}
    .action-links {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>

<div id="start-overlay" class="start-overlay">
    <div class="start-card" id="start-card">
        <div style="text-align:center;color:#999;font-size:14px;">Loading article library...</div>
    </div>
</div>

<div class="reading-header" id="reading-header" style="display:none;">
    <h1 id="title-display"></h1>
    <div class="badges" id="badges-display"></div>
    <div class="meta-bar">
        <span class="timer">Time: <span id="timer-display">00:00</span></span>
        <span>Questions: <span id="progress-count">0</span> / 5</span>
        <span class="progress-pill" id="library-progress">Loading...</span>
    </div>
</div>

<div class="passage-section" id="passage-section" style="display:none;">
    <div id="passage-content"></div>
</div>

<h2 class="section-title" id="questions-title" style="display:none;">📝 Comprehension Questions</h2>

<form id="quiz-form" onsubmit="return false;"></form>

<div class="submit-section" id="submit-section" style="display:none;">
    <button id="submit-btn" onclick="submitQuiz()">SUBMIT</button>
</div>

<div id="results"></div>

<footer style="display:none;" id="footer">
    <a href="history.html">📊 View History</a>
    &nbsp;|&nbsp;
    <a href="mistakes.html">📖 All Vocabulary</a>
    &nbsp;|&nbsp;
    <span id="library-count-footer"></span>
</footer>

<script>
let ALL_ARTICLES = [];
let CURRENT = null;
let CURRENT_ID = null;
let startTime = null;
let submitted = false;
let testStarted = false;

async function init() {{
    try {{
        const res = await fetch('articles.json?v=' + Date.now());
        const data = await res.json();
        ALL_ARTICLES = data.articles || [];

        if (ALL_ARTICLES.length === 0) {{
            showError("No articles in library. Run 'uv run main.py' on your computer to generate articles.");
            return;
        }}

        const savedId = sessionStorage.getItem('current_article_id');
        if (savedId) {{
            CURRENT = ALL_ARTICLES.find(a => a.id === savedId);
            CURRENT_ID = savedId;
        }}

        if (!CURRENT) {{
            const done = JSON.parse(localStorage.getItem('done_article_ids') || '[]');
            const available = ALL_ARTICLES.filter(a => !done.includes(a.id));

            if (available.length === 0) {{
                showAllDone();
                return;
            }}

            CURRENT = available[Math.floor(Math.random() * available.length)];
            CURRENT_ID = CURRENT.id;
            sessionStorage.setItem('current_article_id', CURRENT_ID);
        }}

        renderArticle();

        const savedState = sessionStorage.getItem('reading_submitted_state_' + CURRENT_ID);
        if (savedState) {{
            const state = JSON.parse(savedState);
            document.getElementById('start-overlay').classList.add('hidden');
            document.getElementById('quiz-form').innerHTML = state.formHTML;
            document.getElementById('results').innerHTML = state.resultsHTML;
            document.getElementById('results').classList.add('show');
            document.getElementById('timer-display').textContent = state.timer;
            document.getElementById('progress-count').textContent = state.progress;
            document.getElementById('submit-btn').disabled = true;
            document.getElementById('submit-btn').textContent = 'SUBMITTED';
            submitted = true;
        }}
    }} catch (err) {{
        showError("Failed to load articles: " + err.message);
    }}
}}

function renderArticle() {{
    const done = JSON.parse(localStorage.getItem('done_article_ids') || '[]');
    const remaining = ALL_ARTICLES.length - done.length;

    document.getElementById('start-card').innerHTML = `
        <h2>DAILY READING</h2>
        <div class="reading-title">"${{CURRENT.title}}"</div>
        <div class="badges">
            <span class="badge category">${{CURRENT.category}}</span>
            <span class="badge genre">${{CURRENT.genre}}</span>
            <span class="badge length">${{CURRENT.length}}</span>
            ${{CURRENT.is_china ? '<span class="badge china">🇨🇳 China</span>' : ''}}
            ${{CURRENT.is_challenge ? '<span class="badge challenge">⭐ Challenge</span>' : ''}}
        </div>
        <div class="instructions">
            <strong>Instructions</strong>
            <ul>
                <li>Read the passage carefully</li>
                <li>Hover over <span style="color:#1a4d8f;border-bottom:2px dotted #1a4d8f;">highlighted words</span> for definitions</li>
                <li>Answer 5 comprehension questions</li>
                <li>Timer starts when you click START</li>
            </ul>
        </div>
        <button class="start-btn" onclick="startTest()">START</button>
        <div style="margin-top:16px;font-size:12px;color:#888;font-family:Arial;">
            📚 ${{ALL_ARTICLES.length - remaining}}/${{ALL_ARTICLES.length}} articles completed
        </div>
    `;

    document.getElementById('title-display').textContent = CURRENT.title;
    document.getElementById('badges-display').innerHTML = `
        <span class="badge category">${{CURRENT.category}}</span>
        <span class="badge genre">${{CURRENT.genre}}</span>
        <span class="badge length">${{CURRENT.length}}</span>
        ${{CURRENT.is_china ? '<span class="badge china">🇨🇳 China</span>' : ''}}
        ${{CURRENT.is_challenge ? '<span class="badge challenge">⭐ Challenge</span>' : ''}}
    `;
    document.getElementById('library-progress').textContent = `${{ALL_ARTICLES.length - remaining}}/${{ALL_ARTICLES.length}} done`;
    document.getElementById('library-count-footer').textContent = `Library: ${{ALL_ARTICLES.length}} articles`;

    const paragraphs = CURRENT.passage_html.split(/<\\/p>\\s*<p>/);
    let passageContent = CURRENT.passage_html;
    if (!passageContent.includes('<p>')) {{
        passageContent = '<p>' + passageContent.split('\\n\\n').join('</p><p>') + '</p>';
    }}
    document.getElementById('passage-content').innerHTML = passageContent;

    const TYPE_LABELS = {{
        main_idea: "Main Idea", detail: "Detail",
        inference: "Inference", vocab_in_context: "Vocabulary",
        organization: "Organization", tone: "Tone/Attitude",
        advanced_detail: "Synthesis"
    }};

    let questionsHTML = "";
    CURRENT.questions.forEach((q, i) => {{
        const typeLabel = TYPE_LABELS[q.type] || q.type;
        let optionsHTML = "";
        const letters = ["A", "B", "C", "D"];
        q.options.forEach((opt, j) => {{
            optionsHTML += `
                <label class="option" data-q="${{i}}" data-opt="${{j}}">
                    <input type="radio" name="q${{i}}" value="${{j}}" style="margin-right:10px;margin-top:4px;">
                    <span class="letter">${{letters[j]}}.</span>
                    <span>${{opt}}</span>
                </label>`;
        }});
        questionsHTML += `
            <div class="question" data-q-index="${{i}}">
                <div class="q-header">
                    <span class="q-number">${{i+1}}.</span>
                    <span class="q-type">${{typeLabel}}</span>
                </div>
                <div class="q-stem">${{q.stem}}</div>
                <div class="q-options">${{optionsHTML}}</div>
            </div>`;
    }});
    document.getElementById('quiz-form').innerHTML = questionsHTML;

    document.querySelectorAll('input[type="radio"]').forEach(input => {{
        input.addEventListener('change', () => {{
            const answered = new Set();
            document.querySelectorAll('input[type="radio"]:checked').forEach(r => answered.add(r.name));
            document.getElementById('progress-count').textContent = answered.size;
        }});
    }});

    document.getElementById('reading-header').style.display = 'block';
    document.getElementById('passage-section').style.display = 'block';
    document.getElementById('questions-title').style.display = 'block';
    document.getElementById('submit-section').style.display = 'block';
    document.getElementById('footer').style.display = 'block';
}}

function startTest() {{
    testStarted = true;
    startTime = Date.now();
    document.getElementById('start-overlay').classList.add('hidden');
}}

setInterval(() => {{
    if (!testStarted || submitted) return;
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    const mm = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const ss = String(elapsed % 60).padStart(2, '0');
    document.getElementById('timer-display').textContent = mm + ':' + ss;
}}, 1000);

function submitQuiz() {{
    if (submitted) return;
    const userAnswers = {{}};
    let answeredCount = 0;
    CURRENT.questions.forEach((q, i) => {{
        const sel = document.querySelector('input[name="q' + i + '"]:checked');
        if (sel) {{ userAnswers[i] = parseInt(sel.value); answeredCount++; }}
        else {{ userAnswers[i] = -1; }}
    }});

    if (answeredCount < CURRENT.questions.length) {{
        if (!confirm('You have ' + (CURRENT.questions.length - answeredCount) + ' unanswered. Submit?')) return;
    }}

    submitted = true;
    const totalTime = startTime ? Math.floor((Date.now() - startTime) / 1000) : 0;

    let correctCount = 0;
    CURRENT.questions.forEach((q, i) => {{
        const userAns = userAnswers[i];
        const correctAns = q.correct_index;
        const opts = document.querySelectorAll('label.option[data-q="' + i + '"]');
        opts.forEach(opt => {{
            const idx = parseInt(opt.dataset.opt);
            opt.style.pointerEvents = 'none';
            opt.querySelector('input').disabled = true;
            if (idx === correctAns) {{
                opt.classList.add(userAns === correctAns ? 'user-correct' : 'show-correct');
            }} else if (idx === userAns) {{
                opt.classList.add('user-wrong');
            }}
        }});
        if (userAns === correctAns) correctCount++;
    }});

    const accuracy = Math.round((correctCount / CURRENT.questions.length) * 100);
    const mm = String(Math.floor(totalTime / 60)).padStart(2, '0');
    const ss = String(totalTime % 60).padStart(2, '0');

    const sessionRecord = {{
        date: new Date().toISOString().split('T')[0],
        timestamp: new Date().toISOString(),
        article_id: CURRENT.id,
        title: CURRENT.title,
        topic: CURRENT.topic_title,
        category: CURRENT.category,
        genre: CURRENT.genre,
        length: CURRENT.length,
        correct: correctCount,
        total: CURRENT.questions.length,
        accuracy: accuracy,
        duration_sec: totalTime
    }};
    const allSessions = JSON.parse(localStorage.getItem('reading_sessions') || '[]');
    allSessions.push(sessionRecord);
    localStorage.setItem('reading_sessions', JSON.stringify(allSessions));

    let explanationsHtml = '<h3 style="margin-top:20px;color:#1a1a1a;font-size:16px;border-bottom:1px solid #ddd;padding-bottom:8px;">📝 Question Review</h3>';
    CURRENT.questions.forEach((q, i) => {{
        const userAns = userAnswers[i];
        const isCorrect = userAns === q.correct_index;
        const userText = userAns >= 0 ? q.options[userAns] : '(unanswered)';
        explanationsHtml += '<div class="explanation-block' + (isCorrect ? '' : ' wrong') + '">' +
            '<div class="exp-q">' + (i+1) + '. ' + q.stem + '</div>' +
            '<div class="exp-result">' + (isCorrect ? '✓ Correct: ' : '✗ You answered: ') +
            '<strong>' + userText + '</strong>' +
            (isCorrect ? '' : ' &nbsp; Correct: <strong>' + q.options[q.correct_index] + '</strong>') +
            '</div>' +
            '<div class="exp-text"><strong>Why:</strong> ' + q.explanation + '</div>' +
            '</div>';
    }});

    const resultsHtml =
        '<div class="new-test-section">' +
        '<button class="start-new-btn" onclick="nextArticle()">📖 Next Article</button>' +
        '<p style="color:#666;font-size:12px;margin-top:8px;">Marks this article as complete, randomly picks next</p>' +
        '</div>' +
        '<div class="score-summary">' +
        '<div class="score-item"><span class="score-value">' + correctCount + '/' + CURRENT.questions.length + '</span><span class="score-label">Score</span></div>' +
        '<div class="score-item"><span class="score-value">' + accuracy + '%</span><span class="score-label">Accuracy</span></div>' +
        '<div class="score-item"><span class="score-value">' + mm + ':' + ss + '</span><span class="score-label">Time</span></div>' +
        '</div>' +
        '<div class="action-links">' +
        '<a href="history.html">📊 View Progress</a>' +
        '<a href="mistakes.html">📖 Review Words</a>' +
        '</div>' +
        explanationsHtml;

    document.getElementById('results').innerHTML = resultsHtml;
    document.getElementById('results').classList.add('show');
    document.getElementById('submit-btn').disabled = true;
    document.getElementById('submit-btn').textContent = 'SUBMITTED';
    document.getElementById('results').scrollIntoView({{behavior: 'smooth'}});

    sessionStorage.setItem('reading_submitted_state_' + CURRENT_ID, JSON.stringify({{
        formHTML: document.getElementById('quiz-form').innerHTML,
        resultsHTML: document.getElementById('results').innerHTML,
        timer: document.getElementById('timer-display').textContent,
        progress: document.getElementById('progress-count').textContent
    }}));
}}

function nextArticle() {{
    const done = JSON.parse(localStorage.getItem('done_article_ids') || '[]');
    if (CURRENT && !done.includes(CURRENT.id)) {{
        done.push(CURRENT.id);
        localStorage.setItem('done_article_ids', JSON.stringify(done));
    }}
    sessionStorage.removeItem('current_article_id');
    sessionStorage.removeItem('reading_submitted_state_' + CURRENT_ID);
    location.reload();
}}

function showAllDone() {{
    document.body.innerHTML = `
        <div class="all-done">
            <h1>🎉 All Articles Completed!</h1>
            <p style="color:#666;font-size:17px;margin:24px 0;">
                You've finished all ${{ALL_ARTICLES.length}} readings in the library.<br>
                Great work!
            </p>
            <div class="actions">
                <button onclick="resetAndContinue()">🔄 Reset and Practice Again</button>
                <button class="secondary" onclick="showGenerateMore()">📚 How to Generate More</button>
            </div>
            <div style="margin-top:40px;font-size:13px;color:#888;">
                <a href="history.html" style="color:#1a4d8f;">📊 View Full History</a> &nbsp;|&nbsp;
                <a href="mistakes.html" style="color:#1a4d8f;">📖 Review All Vocabulary</a>
            </div>
        </div>`;
}}

function resetAndContinue() {{
    if (confirm('Reset all "done" records and re-randomize all articles?\\n\\nYour history and vocabulary are NOT affected.')) {{
        localStorage.removeItem('done_article_ids');
        location.reload();
    }}
}}

function showGenerateMore() {{
    alert(
        "To generate more articles, run on your computer:\\n\\n" +
        "cd ~/code/isee-reading-daily\\n" +
        "uv run main.py 50\\n\\n" +
        "Then push to GitHub:\\n" +
        "git add . && git commit -m 'More articles' && git push\\n\\n" +
        "Refresh this page to see new articles."
    );
}}

function showError(msg) {{
    document.getElementById('start-card').innerHTML = `
        <h2 style="color:#c62828;">⚠️ Error</h2>
        <p style="color:#555;font-family:Arial;font-size:14px;">${{msg}}</p>`;
}}

init();
</script>

</body>
</html>'''


def render_history_html():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Reading Progress History</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
* { box-sizing: border-box; }
body { font-family: "Georgia", serif; max-width: 860px; margin: 0 auto; padding: 24px 20px 60px; background: #fafaf7; }
header { border-bottom: 3px double #333; padding-bottom: 16px; margin-bottom: 28px; }
h1 { font-size: 24px; margin: 0 0 6px; }
h2 { font-size: 18px; margin: 32px 0 16px; padding-left: 12px; border-left: 4px solid #1a4d8f; }
.subtitle { font-size: 13px; color: #555; font-style: italic; }
.back-link { display: inline-block; margin-bottom: 20px; color: #1a4d8f; text-decoration: none; font-family: Arial, sans-serif; font-size: 14px; }
.stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
.stat-card { background: white; padding: 18px 16px; border: 1px solid #ddd; border-radius: 6px; text-align: center; font-family: Arial, sans-serif; }
.stat-value { font-size: 28px; font-weight: bold; color: #1a4d8f; }
.stat-label { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 1px; margin-top: 6px; }
.chart-section { background: white; padding: 20px; border: 1px solid #ddd; border-radius: 6px; margin-top: 12px; }
.chart-container { position: relative; height: 320px; }
table { width: 100%; border-collapse: collapse; background: white; font-family: Arial, sans-serif; font-size: 13px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); border-radius: 6px; overflow: hidden; }
th { background: #1a4d8f; color: white; padding: 10px 12px; text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
td { padding: 10px 12px; border-bottom: 1px solid #eee; }
tr:hover td { background: #fafafa; }
button { background: #1a4d8f; color: white; border: none; padding: 8px 18px; font-size: 13px; border-radius: 4px; cursor: pointer; font-family: Arial, sans-serif; }
button.danger { background: #c62828; }
.clear-section { margin-top: 20px; text-align: right; }
.empty-state { text-align: center; padding: 60px 20px; color: #888; }
@media (max-width: 600px) { .stats-grid { grid-template-columns: repeat(2, 1fr); } table { font-size: 11px; } th, td { padding: 8px 6px; } }
</style>
</head>
<body>
<a href="index.html" class="back-link">← Back to Reading</a>
<header><h1>READING PROGRESS</h1><div class="subtitle">Practice history</div></header>
<div id="content"></div>
<script>
function render() {
    const sessions = JSON.parse(localStorage.getItem('reading_sessions') || '[]');
    const content = document.getElementById('content');
    if (sessions.length === 0) {
        content.innerHTML = '<div class="empty-state"><h2 style="border:none;padding:0;">📊 No sessions yet</h2><p>Complete a reading to start tracking.</p></div>';
        return;
    }
    const totalQ = sessions.reduce((s, x) => s + x.total, 0);
    const totalCorrect = sessions.reduce((s, x) => s + x.correct, 0);
    const avgAcc = Math.round(totalCorrect / totalQ * 100);
    let trend = '—', trendColor = '#888';
    if (sessions.length >= 4) {
        const recent = sessions.slice(-3).reduce((s, x) => s + x.accuracy, 0) / 3;
        const earlier = sessions.slice(0, 3).reduce((s, x) => s + x.accuracy, 0) / 3;
        const diff = Math.round(recent - earlier);
        if (diff > 0) { trend = '↑ +' + diff + '%'; trendColor = '#27ae60'; }
        else if (diff < 0) { trend = '↓ ' + diff + '%'; trendColor = '#c62828'; }
        else trend = '→ 0%';
    }
    let html = '<div class="stats-grid">' +
        '<div class="stat-card"><div class="stat-value">' + sessions.length + '</div><div class="stat-label">Readings</div></div>' +
        '<div class="stat-card"><div class="stat-value">' + totalQ + '</div><div class="stat-label">Questions</div></div>' +
        '<div class="stat-card"><div class="stat-value">' + avgAcc + '%</div><div class="stat-label">Avg Accuracy</div></div>' +
        '<div class="stat-card"><div class="stat-value" style="color:' + trendColor + '">' + trend + '</div><div class="stat-label">Trend</div></div>' +
        '</div>' +
        '<h2>📈 Accuracy Over Time</h2>' +
        '<div class="chart-section"><div class="chart-container"><canvas id="acc-chart"></canvas></div></div>' +
        '<h2>📋 All Sessions</h2>' +
        '<table><thead><tr><th>Date</th><th>Title</th><th>Genre</th><th>Score</th><th>Acc.</th><th>Time</th></tr></thead><tbody>';
    sessions.slice().reverse().forEach(s => {
        const time = new Date(s.timestamp).toLocaleTimeString('en-US', {hour:'2-digit',minute:'2-digit'});
        const dur = Math.floor(s.duration_sec/60) + 'm' + (s.duration_sec%60) + 's';
        const accColor = s.accuracy >= 80 ? '#27ae60' : (s.accuracy >= 60 ? '#f57c00' : '#c62828');
        html += '<tr><td>' + s.date + '<br><span style="color:#999;font-size:10px;">' + time + '</span></td>' +
            '<td>' + (s.title || '-') + '</td><td>' + (s.genre || '-') + '</td>' +
            '<td>' + s.correct + '/' + s.total + '</td>' +
            '<td style="color:' + accColor + ';font-weight:bold;">' + s.accuracy + '%</td>' +
            '<td>' + dur + '</td></tr>';
    });
    html += '</tbody></table><div class="clear-section"><button class="danger" onclick="if(confirm(\\'Clear all?\\')){localStorage.removeItem(\\'reading_sessions\\');render();}">Clear All History</button></div>';
    content.innerHTML = html;
    const ctx = document.getElementById('acc-chart');
    new Chart(ctx, {
        type: 'line',
        data: { labels: sessions.map(s => s.date), datasets: [{ label: 'Accuracy', data: sessions.map(s => s.accuracy), borderColor: '#1a4d8f', backgroundColor: 'rgba(26,77,143,0.1)', borderWidth: 2.5, pointRadius: 5, tension: 0.3, fill: true }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, max: 100, ticks: { callback: v => v + '%' } } } }
    });
}
render();
</script>
</body>
</html>'''


def render_mistakes_html():
    """All vocabulary from articles.json"""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vocabulary Library</title>
<style>
* { box-sizing: border-box; }
body { font-family: "Georgia", serif; max-width: 760px; margin: 0 auto; padding: 24px 20px 60px; background: #fafaf7; }
header { border-bottom: 3px double #333; padding-bottom: 16px; margin-bottom: 28px; }
h1 { font-size: 24px; margin: 0 0 6px; }
.subtitle { font-size: 13px; color: #555; font-style: italic; }
.back-link { display: inline-block; margin-bottom: 20px; color: #1a4d8f; text-decoration: none; font-family: Arial, sans-serif; font-size: 14px; }
.stats { background: white; padding: 14px 18px; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 24px; font-family: Arial, sans-serif; font-size: 14px; }
.vocab-card { background: white; padding: 14px 18px; margin-bottom: 10px; border-left: 4px solid #1a4d8f; border-radius: 0 4px 4px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.vocab-word { font-size: 18px; font-weight: bold; color: #1a4d8f; }
.vocab-def { color: #333; font-size: 14px; margin-top: 4px; }
.vocab-source { font-size: 11px; color: #999; margin-top: 6px; font-family: Arial, sans-serif; font-style: italic; }
.empty-state { text-align: center; padding: 60px 20px; color: #888; }
</style>
</head>
<body>
<a href="index.html" class="back-link">← Back to Reading</a>
<header><h1>VOCABULARY LIBRARY</h1><div class="subtitle">All words from the article library</div></header>
<div id="content"></div>
<script>
async function load() {
    try {
        const res = await fetch('articles.json?v=' + Date.now());
        const data = await res.json();
        const articles = data.articles || [];
        const wordMap = {};
        articles.forEach(a => {
            (a.vocab || []).forEach(v => {
                if (!wordMap[v.word]) {
                    wordMap[v.word] = { word: v.word, definition: v.definition, source: a.title };
                }
            });
        });
        const VOCAB = Object.values(wordMap);
        const content = document.getElementById('content');
        if (VOCAB.length === 0) {
            content.innerHTML = '<div class="empty-state"><p>No vocabulary yet. Generate articles first.</p></div>';
            return;
        }
        VOCAB.sort((a, b) => a.word.toLowerCase().localeCompare(b.word.toLowerCase()));
        let html = '<div class="stats"><strong>' + VOCAB.length + '</strong> unique words across ' + articles.length + ' articles</div>';
        VOCAB.forEach(v => {
            html += '<div class="vocab-card"><div class="vocab-word">' + v.word + '</div>' +
                '<div class="vocab-def">' + v.definition + '</div>' +
                '<div class="vocab-source">From: ' + v.source + '</div></div>';
        });
        content.innerHTML = html;
    } catch (err) {
        document.getElementById('content').innerHTML = '<div class="empty-state"><p>Error loading vocabulary.</p></div>';
    }
}
load();
</script>
</body>
</html>'''


# ============ Entry Point ============
if __name__ == "__main__":
    # Parse argument
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "reset":
            confirm = input("⚠️  This will delete articles.json. Continue? (yes/no): ")
            if confirm.lower() == "yes":
                if ARTICLES_FILE.exists():
                    ARTICLES_FILE.unlink()
                    print("✅ articles.json deleted. Run 'uv run main.py' to regenerate.")
                else:
                    print("articles.json doesn't exist.")
            else:
                print("Cancelled.")
            sys.exit(0)
        try:
            batch_size = int(arg)
        except ValueError:
            print(f"⚠️  Invalid argument '{arg}'. Use a number or 'reset'.")
            sys.exit(1)
    else:
        batch_size = DEFAULT_BATCH_SIZE

    # Run batch
    articles = run_batch(batch_size)

    # Write HTML files
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        f.write(render_index_html(len(articles)))
    with open(HISTORY_PAGE, "w", encoding="utf-8") as f:
        f.write(render_history_html())
    with open(MISTAKES_FILE_HTML, "w", encoding="utf-8") as f:
        f.write(render_mistakes_html())

    print(f"📄 HTML files generated:")
    print(f"   - {INDEX_FILE}")
    print(f"   - {HISTORY_PAGE}")
    print(f"   - {MISTAKES_FILE_HTML}")
    print(f"   - {ARTICLES_FILE}  ({len(articles)} articles)")
    print(f"\n✅ Done! Open {INDEX_FILE} or push to GitHub.")