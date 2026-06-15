import os
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

# ============ 配置 ============
load_dotenv()
client = Anthropic()

TOPICS_FILE = Path("topics.json")
HISTORY_FILE = Path("topic_history.json")  # 已学主题记录
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
INDEX_FILE = Path("index.html")
HISTORY_PAGE = Path("history.html")
MISTAKES_FILE_HTML = Path("mistakes.html")

GENRES = [
    {
        "name": "narrative",
        "label": "Narrative Nonfiction",
        "guidance": "Tell the story like a writer — use scenes, a sense of time passing, and a specific moment or turning point. Make the reader feel they are there."
    },
    {
        "name": "informational",
        "label": "Informational",
        "guidance": "Explain clearly with facts, examples, and structure. Use cause-and-effect, compare/contrast, or sequence. This is closest to ISEE Reading test passages."
    },
    {
        "name": "argumentative",
        "label": "Argumentative",
        "guidance": "Present a clear position or claim, then support it with 2-3 specific reasons grounded in facts. Acknowledge an opposing view briefly."
    },
    {
        "name": "descriptive",
        "label": "Descriptive",
        "guidance": "Use vivid sensory details (sight, sound, smell, texture, motion) to paint the topic. Bring it to life through specifics, not generalities."
    },
]

# ============ 命令行参数（可选）============
# uv run main.py            → 随机长度
# uv run main.py short      → 强制 short
# uv run main.py long       → 强制 long
length_arg = sys.argv[1].lower() if len(sys.argv) > 1 else None
if length_arg not in (None, "short", "long"):
    print(f"⚠️  Unknown argument: {length_arg}. Using random length.")
    length_arg = None

# ============ 主题选择（带去重）============
def load_topics():
    with open(TOPICS_FILE, encoding="utf-8") as f:
        return json.load(f)["topics"]

def load_topic_history():
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"used_topic_ids": [], "session_count": 0, "used_vocab": []}

def get_recent_vocab(history, n=40):
    """返回最近 n 个用过的词"""
    return history.get("used_vocab", [])[-n:]

def save_topic_history(h):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(h, f, ensure_ascii=False, indent=2)

def pick_topic(all_topics, history):
    available = [t for t in all_topics if t["id"] not in history["used_topic_ids"]]
    if not available:
        print("📌 All 100 topics used! Resetting topic history.")
        history["used_topic_ids"] = []
        available = all_topics
    return random.choice(available)

topics = load_topics()
history = load_topic_history()
topic = pick_topic(topics, history)

genre = random.choice(GENRES)

# === Q5 题型随机选择（不让 AI 决定，避免偏向）===
q5_type = random.choice(["organization", "tone", "advanced_detail"])

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
   - "introducing a person, then describing their key achievement"

   The correct answer must accurately describe the passage's actual structure.""",

    "tone": """Q5 (Tone/Attitude): MUST use one of these formats:
   - "The author's attitude toward [X] can best be described as..."
   - "The tone of the passage is mainly..."
   - "Which word best describes how the author feels about [X]?"

   Options MUST be specific tone adjectives. Choose 4 from this list:
   admiring, neutral, skeptical, informative, cautionary, reverent,
   sympathetic, critical, enthusiastic, contemplative, ironic,
   curious, appreciative, concerned, matter-of-fact

   The correct answer must be supported by specific word choices in the
   passage. Distractors should be tones that ALMOST fit but miss the nuance.""",

    "advanced_detail": """Q5 (Advanced Detail / Synthesis): MUST use one of these formats
   (DIFFERENT from Q2's format):
   - "Which statement best summarizes the relationship between X and Y?"
   - "The author's description of [thing] emphasizes its..."
   - "Which of the following BEST captures the significance of [X]?"

   This question should require connecting multiple pieces of information
   from across the passage, not just locating a single fact."""
}

q5_instructions = Q5_INSTRUCTIONS[q5_type]
q5_label = {"organization": "Organization", "tone": "Tone/Attitude", "advanced_detail": "Advanced Detail"}[q5_type]

if length_arg == "short":
    length_label, word_range = "Short", "200-250"
elif length_arg == "long":
    length_label, word_range = "Long", "350-450"
else:
    if random.random() < 0.5:
        length_label, word_range = "Short", "200-250"
    else:
        length_label, word_range = "Long", "350-450"

print(f"📚 Session #{history['session_count'] + 1}")
print(f"📖 Topic: {topic['title']}")
print(f"   Category: {topic['category']}{'  🇨🇳' if topic.get('china') else ''}{'  ⭐ Challenge' if topic['challenge'] else ''}")
print(f"🎭 Genre: {genre['label']}")
print(f"🎯 Q5 Type: {q5_label} (random)")
print(f"📏 Length: {length_label} ({word_range} words)\n")

# ============ JSON 解析工具（带重试，防 AI 偶发性输出错误）============
def parse_ai_json(prompt, max_tokens=4096, max_retries=3, label=""):
    """调用 Claude 并解析 JSON 响应，失败时自动重试"""
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text.strip()
            # 清除可能的 markdown 代码块包装
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

# ============ Step 1: Research (generate verifiable facts) ============
print("🔍 Step 1/2: Researching verifiable facts...")

research_prompt = f"""You are a meticulous research assistant. Provide 6-8 verifiable, specific facts about this topic, suitable as the factual basis for an educational reading passage for advanced 5th-6th grade students.

TOPIC: {topic['title']}

REQUIREMENTS for each fact:
- Must include at least one specific element: a name, date, place, number, quotation, or measurable detail
- Avoid vague qualifiers ("many", "some", "often", "people believe")
- Avoid invented quotations; only include quotations if they are widely documented
- Each fact should be 1-2 sentences

Output STRICT JSON only (no markdown, no commentary):

{{
  "facts": [
    "Specific fact 1 with names/dates/numbers.",
    "Specific fact 2 with names/dates/numbers.",
    "..."
  ]
}}

Generate 6-8 facts. Be accurate. If you are uncertain about a specific detail, omit it rather than guess.
"""

research_result = parse_ai_json(research_prompt, max_tokens=2048, label="Step 1 (Research)")
facts = research_result["facts"]
print(f"   ✓ {len(facts)} facts gathered\n")

# ============ Step 2: Writing (based on facts) ============
print("✍️  Step 2/2: Writing passage and questions...")

writing_prompt = f"""You are a senior editor at a publication like Newsela or National Geographic Kids, writing for advanced 10-11 year old readers (ISEE Lower Level preparation). Write ONE reading passage based ON THE PROVIDED FACTS, plus 5 comprehension questions.

═══════════════════════════════════════
TOPIC: {topic['title']}
GENRE: {genre['label']}
GENRE GUIDANCE: {genre['guidance']}
LENGTH TARGET: {word_range} words
READING LEVEL: Lexile 1000-1100
═══════════════════════════════════════
VOCABULARY EXCLUSION LIST (DO NOT use these words in your <vocab> markers):
{', '.join(get_recent_vocab(history, 40)) if get_recent_vocab(history, 40) else '(none yet)'}

These words have been used in recent readings. Choose DIFFERENT challenging vocabulary.
═══════════════════════════════════════

VERIFIED FACTS TO USE (do not invent additional facts):
{json.dumps(facts, indent=2)}

═══════════════════════════════════════
QUALITY REQUIREMENTS — strictly enforce all of these:

1. STRUCTURE
   - Open with a SPECIFIC, concrete hook (a moment, image, detail) — NOT "Have you ever wondered...", NOT "Long ago...", NOT a general statement
   - End with a thought-provoking conclusion that resonates — NOT "In conclusion...", NOT "And that is why..."
   - Logical flow appropriate to the genre

2. CONTENT
   - Anchor every paragraph in at least one specific fact from the list above
   - Include at least 3 concrete details: names, dates, numbers, places, measurements
   - Include at least 2 sensory details (sound, sight, texture, motion)
   - For narrative: include a "turning point" or moment of change
   - For argumentative: present a clear claim and 2 supporting reasons
   - For descriptive: use varied sensory language
   - For informational: use clear structure (cause-effect, compare, sequence)

3. STYLE
   - Vary sentence length (mix short punch with longer flowing sentences)
   - Use precise verbs (not "did", "made", "got") — use "discovered", "constructed", "earned"
   - Avoid filler ("very", "really", "a lot of")
   - Show, don't tell — concrete images over abstract claims

4. VOCABULARY EMBEDDING (CRITICAL)
   - Naturally include 5-7 challenging vocabulary words at ISEE Lower Level
   - These should be words like: perilous, ingenious, eloquent, defy, eventually, prevail, tedious, persevere, monumental, conceal, contemplate, scrutinize, etc.
   - Mark each in the passage with this exact format: <vocab>WORD|simple English definition</vocab>
   - Example: "The team faced <vocab>perilous|extremely dangerous</vocab> weather."
   - Use 5-7 markers, no more, no less
   - Choose words that are CHALLENGING but readable in context

5. FORBIDDEN
   - "Once upon a time", "Long ago", "Have you ever", "In a world where"
   - "In conclusion", "To sum up", "All in all", "And that is the story of"
   - Vague qualifiers: "many", "some", "often", "people thought"
   - Climate change mentions unless directly relevant to topic
   - Made-up quotations or invented details not in the facts above

═══════════════════════════════════════
QUESTION DESIGN — 5 questions in this fixed order:

Q1 (Main Idea): One of:
   - "Which sentence best states the central idea of the passage?"
   - "The primary purpose of this passage is to..."
   - "Which would be the BEST title for this passage?"

Q2 (Detail - High Rigor): MUST use one of these formats (NOT simple fact lookup):
   - "Which detail from the passage BEST supports the idea that..."
   - "The author mentions [specific thing] in order to..."
   - "Which of the following is NOT mentioned in the passage as..."
   - "According to the passage, which is true about..." (with subtle distractors)
   AVOID: "What year did X happen?" or "Where was X born?"

Q3 (Inference): "The passage suggests that..." or "We can conclude from the passage that..." — requires reading between lines, not stated directly.

Q4 (Vocabulary in Context): "In the passage, the word \\"___\\" most nearly means..." — choose a word from your <vocab> markers.

{q5_instructions}

For each question:
- 4 options
- Exactly 1 correct
- 3 plausible distractors that follow these RIGOR rules:
  * Distractor type A: TRUE in real world but NOT stated in the passage (forces close reading)
  * Distractor type B: PARTIALLY true (correct premise, wrong conclusion or scope)
  * Distractor type C: USES words from passage but rearranges meaning incorrectly
- Each question MUST include at least 2 of the 3 distractor types above
- AVOID distractors that are obviously absurd or unrelated to the topic
- Correct answer should require synthesis, not just word-matching to the passage
- Include a 1-sentence explanation of why the correct answer is right (and key wrong answers are wrong)
- VARY correct answer position across the 5 questions (do NOT cluster at any single index)
- Reading level for distractors should match passage difficulty (not simpler)

═══════════════════════════════════════
OUTPUT FORMAT — STRICT JSON, no markdown wrappers, no commentary:

{{
  "title": "Engaging title (5-8 words)",
  "passage": "Full passage text with <vocab>WORD|definition</vocab> markers embedded inline.",
  "questions": [
    {{
      "type": "main_idea",
      "stem": "...",
      "options": ["...", "...", "...", "..."],
      "correct_index": 0,
      "explanation": "..."
    }},
    {{
      "type": "detail",
      "stem": "...",
      "options": ["...", "...", "...", "..."],
      "correct_index": 1,
      "explanation": "..."
    }},
    {{
      "type": "detail",
      "stem": "...",
      "options": ["...", "...", "...", "..."],
      "correct_index": 2,
      "explanation": "..."
    }},
    {{
      "type": "inference",
      "stem": "...",
      "options": ["...", "...", "...", "..."],
      "correct_index": 3,
      "explanation": "..."
    }},
    {{
      "type": "vocab_in_context",
      "stem": "In the passage, the word \\"___\\" most nearly means",
      "options": ["...", "...", "...", "..."],
      "correct_index": 0,
      "explanation": "..."
    }}
  ]
}}

Final check before output:
- Did you use 5-7 <vocab> markers?
- Did you avoid all forbidden phrases?
- Did you anchor paragraphs in facts?
- Are correct answer positions varied (not all index 1)?
- Is the passage truly {word_range} words?
"""

result = parse_ai_json(writing_prompt, max_tokens=8192, label="Step 2 (Writing)")
print(f"   ✓ Title: {result['title']}")
print(f"   ✓ {len(result['questions'])} questions generated\n")

# ============ 解析 vocab 标记 ============
def parse_vocab(passage):
    """从 passage 中提取 <vocab>word|def</vocab>，返回 (HTML 版 passage, vocab 列表)"""
    pattern = r'<vocab>(.*?)\|(.*?)</vocab>'
    matches = re.findall(pattern, passage)
    vocab_list = [{"word": m[0].strip(), "definition": m[1].strip()} for m in matches]

    # 替换为 HTML span
    def replace_func(match):
        word, definition = match.group(1).strip(), match.group(2).strip()
        # 用 data-def 存释义；HTML 转义引号
        safe_def = definition.replace('"', '&quot;')
        return f'<span class="vocab" data-def="{safe_def}">{word}</span>'

    html_passage = re.sub(pattern, replace_func, passage)
    return html_passage, vocab_list

html_passage, vocab_list = parse_vocab(result["passage"])
print(f"   ✓ {len(vocab_list)} vocabulary words highlighted\n")

# ============ 保存数据 ============
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
session_data = {
    "timestamp": timestamp,
    "topic_id": topic["id"],
    "topic_title": topic["title"],
    "category": topic["category"],
    "is_china": topic.get("china", False),
    "is_challenge": topic["challenge"],
    "genre": genre["label"],
    "length": length_label,
    "facts": facts,
    "title": result["title"],
    "passage": result["passage"],
    "vocab": vocab_list,
    "questions": result["questions"]
}
data_file = DATA_DIR / f"{timestamp}.json"
with open(data_file, "w", encoding="utf-8") as f:
    json.dump(session_data, f, ensure_ascii=False, indent=2)

# 更新历史
history["used_topic_ids"].append(topic["id"])
history["session_count"] += 1
if "used_vocab" not in history:
    history["used_vocab"] = []
# 记录本次新词到历史
history["used_vocab"].extend([v["word"].lower() for v in vocab_list])
save_topic_history(history)

# ============ 生成 index.html ============
def render_index_html():
    questions_json = json.dumps(result["questions"], ensure_ascii=False)
    questions_data_json = json.dumps({
        "title": result["title"],
        "topic": topic["title"],
        "category": topic["category"],
        "genre": genre["label"],
        "length": length_label,
        "is_china": topic.get("china", False),
        "is_challenge": topic["challenge"]
    }, ensure_ascii=False)

    # 渲染题目 HTML
    question_blocks = ""
    for i, q in enumerate(result["questions"]):
        type_labels = {
            "main_idea": "Main Idea",
            "detail": "Detail",
            "inference": "Inference",
            "vocab_in_context": "Vocabulary in Context"
        }
        type_label = type_labels.get(q["type"], q["type"])

        options_html = ""
        letters = ["A", "B", "C", "D"]
        for j, opt in enumerate(q["options"]):
            options_html += f'''
            <label class="option" data-q="{i}" data-opt="{j}">
                <input type="radio" name="q{i}" value="{j}">
                <span class="letter">{letters[j]}.</span>
                <span class="opt-text">{opt}</span>
            </label>'''

        question_blocks += f'''
        <div class="question" data-q-index="{i}">
            <div class="q-header">
                <span class="q-number">{i+1}.</span>
                <span class="q-type">{type_label}</span>
            </div>
            <div class="q-stem">{q["stem"]}</div>
            <div class="q-options">{options_html}</div>
        </div>'''

    china_badge = '<span class="badge china">🇨🇳 China</span>' if topic.get("china") else ''
    challenge_badge = '<span class="badge challenge">⭐ Challenge</span>' if topic["challenge"] else ''

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{result["title"]} — Daily Reading</title>
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

/* Start overlay */
.start-overlay {{
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(250, 250, 247, 0.98);
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
    backdrop-filter: blur(8px);
}}
.start-overlay.hidden {{ display: none; }}
.start-card {{
    background: white;
    max-width: 500px;
    width: 90%;
    padding: 40px 32px;
    border: 3px double #1a4d8f;
    border-radius: 8px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.15);
}}
.start-card h2 {{
    font-size: 26px;
    color: #1a4d8f;
    margin: 0 0 8px;
    letter-spacing: 0.5px;
    font-family: "Georgia", serif;
}}
.start-card .reading-title {{
    color: #333;
    font-size: 17px;
    font-style: italic;
    margin: 16px 0 8px;
    line-height: 1.4;
}}
.start-card .badges {{
    margin: 16px 0 24px;
}}
.start-card .instructions {{
    text-align: left;
    background: #f5f5f0;
    padding: 16px 20px;
    border-radius: 6px;
    margin: 20px 0 28px;
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 14px;
    line-height: 1.7;
    color: #333;
}}
.start-card .instructions strong {{
    color: #1a4d8f;
    display: block;
    margin-bottom: 6px;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1px;
}}
.start-card .instructions ul {{ margin: 0; padding-left: 20px; }}
.start-card .instructions li {{ margin: 4px 0; }}
.start-btn {{
    background: #1a4d8f;
    color: white;
    border: none;
    padding: 16px 56px;
    font-size: 17px;
    font-weight: bold;
    border-radius: 4px;
    cursor: pointer;
    font-family: "Helvetica Neue", Arial, sans-serif;
    letter-spacing: 2px;
    transition: background 0.2s;
}}
.start-btn:hover {{ background: #143a6b; }}

.badge {{
    display: inline-block;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 12px;
    margin: 0 4px;
    font-family: "Helvetica Neue", Arial, sans-serif;
    letter-spacing: 0.5px;
    vertical-align: middle;
}}
.badge.category {{ background: #e3edf8; color: #1a4d8f; }}
.badge.genre {{ background: #f0e6d8; color: #8b6914; }}
.badge.length {{ background: #e8f0e3; color: #2d5a1f; }}
.badge.china {{ background: #fde8e8; color: #b71c1c; }}
.badge.challenge {{ background: #fff4e0; color: #e67e22; }}

/* Reading header */
.reading-header {{
    border-bottom: 3px double #333;
    padding-bottom: 20px;
    margin-bottom: 28px;
}}
.reading-header h1 {{
    font-size: 28px;
    margin: 0 0 12px;
    color: #1a1a1a;
    line-height: 1.3;
    font-family: "Georgia", serif;
}}
.reading-meta {{
    font-family: "Helvetica Neue", Arial, sans-serif;
    margin-top: 14px;
}}
.meta-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 14px;
    padding: 10px 16px;
    background: white;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 14px;
}}
.timer {{
    font-family: "Courier New", monospace;
    font-size: 17px;
    font-weight: bold;
    color: #1a4d8f;
}}
.progress {{ color: #555; }}

/* Passage */
.passage-section {{
    background: white;
    padding: 28px 32px;
    border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    margin-bottom: 32px;
    border-left: 4px solid #1a4d8f;
}}
.passage-section p {{
    margin: 0 0 1em;
    text-align: justify;
}}
.passage-section p:last-child {{ margin-bottom: 0; }}

/* Vocab highlight */
.vocab {{
    border-bottom: 2px dotted #1a4d8f;
    cursor: help;
    color: #1a4d8f;
    font-weight: 500;
    position: relative;
}}
.vocab:hover::after {{
    content: attr(data-def);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: #1a4d8f;
    color: white;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 13px;
    font-weight: normal;
    font-family: "Helvetica Neue", Arial, sans-serif;
    white-space: normal;
    max-width: 260px;
    width: max-content;
    z-index: 1000;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    margin-bottom: 6px;
    pointer-events: none;
}}
.vocab:hover::before {{
    content: "";
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 6px solid transparent;
    border-top-color: #1a4d8f;
    z-index: 1000;
    pointer-events: none;
}}

/* Questions section */
h2.section-title {{
    font-size: 20px;
    color: #1a1a1a;
    margin: 36px 0 18px;
    padding-left: 14px;
    border-left: 4px solid #1a4d8f;
    font-family: "Georgia", serif;
}}

.question {{
    background: white;
    padding: 22px 24px;
    border-radius: 8px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
.q-header {{
    margin-bottom: 12px;
    font-family: "Helvetica Neue", Arial, sans-serif;
}}
.q-number {{
    font-size: 18px;
    font-weight: bold;
    color: #1a1a1a;
    margin-right: 12px;
}}
.q-type {{
    font-size: 11px;
    color: white;
    background: #1a4d8f;
    padding: 2px 8px;
    border-radius: 3px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}}
.q-stem {{
    font-size: 16px;
    margin-bottom: 14px;
    line-height: 1.6;
    font-family: "Georgia", serif;
}}
.q-options {{
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-left: 12px;
    font-family: "Helvetica Neue", Arial, sans-serif;
}}
.option {{
    display: flex;
    align-items: flex-start;
    padding: 10px 14px;
    border: 1.5px solid #e0e0e0;
    border-radius: 5px;
    cursor: pointer;
    font-size: 15px;
    transition: all 0.15s;
}}
.option:hover {{ border-color: #1a4d8f; background: #f0f5fb; }}
.option input[type="radio"] {{ margin-right: 10px; margin-top: 4px; }}
.letter {{ font-weight: bold; color: #555; margin-right: 8px; min-width: 20px; }}
.option.user-correct {{ background: #e8f5e9; border-color: #4caf50; }}
.option.user-wrong {{ background: #ffebee; border-color: #f44336; }}
.option.show-correct {{ background: #e8f5e9; border-color: #4caf50; border-style: dashed; }}

.submit-section {{
    margin-top: 32px;
    text-align: center;
    padding-top: 20px;
    border-top: 3px double #333;
}}
#submit-btn {{
    background: #1a4d8f;
    color: white;
    border: none;
    padding: 14px 52px;
    font-size: 16px;
    font-weight: bold;
    border-radius: 4px;
    cursor: pointer;
    font-family: "Helvetica Neue", Arial, sans-serif;
    letter-spacing: 1px;
}}
#submit-btn:hover {{ background: #143a6b; }}
#submit-btn:disabled {{ background: #aaa; cursor: not-allowed; }}

/* Results */
#results {{
    display: none;
    margin-top: 30px;
    padding: 28px;
    background: #f9f9f7;
    border: 2px solid #1a4d8f;
    border-radius: 8px;
    font-family: "Helvetica Neue", Arial, sans-serif;
}}
#results.show {{ display: block; }}
.new-test-section {{
    text-align: center;
    margin-bottom: 20px;
    padding-bottom: 18px;
    border-bottom: 1px solid #ddd;
}}
.start-new-btn {{
    background: #2e7d32;
    color: white;
    border: none;
    padding: 12px 40px;
    font-size: 15px;
    font-weight: bold;
    border-radius: 6px;
    cursor: pointer;
    letter-spacing: 1px;
}}
.start-new-btn:hover {{ background: #1b5e20; }}
.score-summary {{
    display: flex;
    justify-content: space-around;
    text-align: center;
    margin-bottom: 24px;
    flex-wrap: wrap;
    gap: 14px;
}}
.score-item {{ flex: 1; min-width: 120px; }}
.score-value {{ font-size: 32px; font-weight: bold; color: #1a4d8f; display: block; }}
.score-label {{
    font-size: 12px;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}}

.explanation-block {{
    background: white;
    padding: 14px 18px;
    margin-bottom: 12px;
    border-radius: 6px;
    border-left: 4px solid #4caf50;
}}
.explanation-block.wrong {{ border-left-color: #f44336; }}
.explanation-block .exp-q {{ font-weight: 600; margin-bottom: 6px; font-size: 14px; }}
.explanation-block .exp-result {{ font-size: 13px; color: #555; margin-bottom: 8px; }}
.explanation-block .exp-text {{ font-size: 14px; color: #333; line-height: 1.6; }}

.action-links {{
    margin: 20px 0;
    padding: 16px 0;
    border-top: 1px solid #ddd;
    border-bottom: 1px solid #ddd;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
}}
.action-links a {{
    display: block;
    text-align: center;
    padding: 12px 16px;
    background: #1a4d8f;
    color: white;
    text-decoration: none;
    border-radius: 4px;
    font-size: 14px;
    font-weight: 500;
}}
.action-links a:hover {{ background: #143a6b; }}
@media (max-width: 480px) {{ .action-links {{ grid-template-columns: 1fr; }} }}

footer {{
    text-align: center;
    margin-top: 50px;
    color: #999;
    font-size: 13px;
    font-family: "Helvetica Neue", Arial, sans-serif;
}}
footer a {{ color: #888; text-decoration: none; }}
footer a:hover {{ text-decoration: underline; }}

@media (max-width: 600px) {{
    body {{ padding: 18px 14px 60px; font-size: 16px; }}
    .reading-header h1 {{ font-size: 22px; }}
    .passage-section {{ padding: 20px 18px; }}
    .question {{ padding: 18px 16px; }}
    .meta-bar {{ flex-direction: column; gap: 6px; align-items: flex-start; }}
    .vocab:hover::after {{ max-width: 200px; font-size: 12px; }}
    .score-value {{ font-size: 26px; }}
}}
</style>
</head>
<body>

<div id="start-overlay" class="start-overlay">
    <div class="start-card">
        <h2>DAILY READING</h2>
        <div class="reading-title">"{result["title"]}"</div>
        <div class="badges">
            <span class="badge category">{topic["category"]}</span>
            <span class="badge genre">{genre["label"]}</span>
            <span class="badge length">{length_label}</span>
            {china_badge}
            {challenge_badge}
        </div>
        <div class="instructions">
            <strong>Instructions</strong>
            <ul>
                <li>Read the passage carefully</li>
                <li>Hover over <span style="color:#1a4d8f;border-bottom:2px dotted #1a4d8f;font-weight:500;">highlighted words</span> to see definitions</li>
                <li>Answer 5 comprehension questions</li>
                <li>Timer starts when you click START</li>
            </ul>
        </div>
        <button class="start-btn" onclick="startTest()">START</button>
    </div>
</div>

<div class="reading-header">
    <h1>{result["title"]}</h1>
    <div class="badges">
        <span class="badge category">{topic["category"]}</span>
        <span class="badge genre">{genre["label"]}</span>
        <span class="badge length">{length_label}</span>
        {china_badge}
        {challenge_badge}
    </div>
    <div class="meta-bar">
        <span class="timer">Time: <span id="timer-display">00:00</span></span>
        <span class="progress">Questions: <span id="progress-count">0</span> / 5</span>
        <span style="color:#888;font-size:12px;">{timestamp}</span>
    </div>
</div>

<div class="passage-section" id="passage">
{format_passage_html(html_passage)}
</div>

<h2 class="section-title">📝 Comprehension Questions</h2>

<form id="quiz-form" onsubmit="return false;">
{question_blocks}
</form>

<div class="submit-section">
    <button id="submit-btn" onclick="submitQuiz()">SUBMIT</button>
</div>

<div id="results"></div>

<footer>
    <a href="history.html">📊 View History</a>
    &nbsp;|&nbsp;
    <a href="mistakes.html">📖 Review Words</a>
    &nbsp;|&nbsp;
    Generated by Claude
</footer>

<script>
const QUESTIONS_DATA = {questions_json};
const META = {questions_data_json};

let startTime = null;
let submitted = false;
let testStarted = false;

// 状态恢复
const savedState = sessionStorage.getItem('reading_submitted_state');
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

function startTest() {{
    testStarted = true;
    startTime = Date.now();
    document.getElementById('start-overlay').classList.add('hidden');
}}

function clearSubmittedState() {{
    sessionStorage.removeItem('reading_submitted_state');
}}

function startNewTest() {{
    clearSubmittedState();
    location.reload();
}}

document.querySelectorAll('input[type="radio"]').forEach(input => {{
    input.addEventListener('change', () => {{
        const answered = new Set();
        document.querySelectorAll('input[type="radio"]:checked').forEach(r => answered.add(r.name));
        document.getElementById('progress-count').textContent = answered.size;
    }});
}});

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
    QUESTIONS_DATA.forEach((q, i) => {{
        const selected = document.querySelector('input[name="q' + i + '"]:checked');
        if (selected) {{
            userAnswers[i] = parseInt(selected.value);
            answeredCount++;
        }} else {{
            userAnswers[i] = -1;
        }}
    }});

    if (answeredCount < QUESTIONS_DATA.length) {{
        if (!confirm('You have ' + (QUESTIONS_DATA.length - answeredCount) + ' unanswered. Submit?')) return;
    }}

    submitted = true;
    const totalTime = startTime ? Math.floor((Date.now() - startTime) / 1000) : 0;

    let correctCount = 0;
    const wrongDetails = [];

    QUESTIONS_DATA.forEach((q, i) => {{
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

        if (userAns === correctAns) {{
            correctCount++;
        }} else {{
            wrongDetails.push({{
                num: i + 1,
                stem: q.stem,
                correct: q.options[correctAns],
                user: userAns >= 0 ? q.options[userAns] : '(unanswered)',
                explanation: q.explanation,
                type: q.type
            }});
        }}
    }});

    const accuracy = Math.round((correctCount / QUESTIONS_DATA.length) * 100);
    const mm = String(Math.floor(totalTime / 60)).padStart(2, '0');
    const ss = String(totalTime % 60).padStart(2, '0');

    // 保存 session 到 history
    const sessionRecord = {{
        date: new Date().toISOString().split('T')[0],
        timestamp: new Date().toISOString(),
        title: META.title,
        topic: META.topic,
        category: META.category,
        genre: META.genre,
        length: META.length,
        correct: correctCount,
        total: QUESTIONS_DATA.length,
        accuracy: accuracy,
        duration_sec: totalTime
    }};
    const allSessions = JSON.parse(localStorage.getItem('reading_sessions') || '[]');
    allSessions.push(sessionRecord);
    localStorage.setItem('reading_sessions', JSON.stringify(allSessions));

    // 累积生词到 mistakes（用于回顾）
    if (wrongDetails.some(w => w.type === 'vocab_in_context')) {{
        // 简化：错的 vocab 题已经在 wrongDetails 里
    }}

    let explanationsHtml = '<h3 style="margin-top:20px;color:#1a1a1a;font-size:16px;border-bottom:1px solid #ddd;padding-bottom:8px;">📝 Question Review</h3>';
    QUESTIONS_DATA.forEach((q, i) => {{
        const userAns = userAnswers[i];
        const correctAns = q.correct_index;
        const isCorrect = userAns === correctAns;
        const userText = userAns >= 0 ? q.options[userAns] : '(unanswered)';
        explanationsHtml += '<div class="explanation-block' + (isCorrect ? '' : ' wrong') + '">' +
            '<div class="exp-q">' + (i+1) + '. ' + q.stem + '</div>' +
            '<div class="exp-result">' + (isCorrect ? '✓ Correct: ' : '✗ You answered: ') +
            '<strong>' + userText + '</strong>' +
            (isCorrect ? '' : ' &nbsp; Correct answer: <strong>' + q.options[correctAns] + '</strong>') +
            '</div>' +
            '<div class="exp-text"><strong>Why:</strong> ' + q.explanation + '</div>' +
            '</div>';
    }});

    const resultsHtml =
        '<div class="new-test-section">' +
        '<button class="start-new-btn" onclick="startNewTest()">🔄 Generate New Reading</button>' +
        '<p style="color:#666;font-size:12px;margin-top:8px;font-style:italic;">(Run <code>uv run main.py</code> on your computer first)</p>' +
        '</div>' +
        '<div class="score-summary">' +
        '<div class="score-item"><span class="score-value">' + correctCount + '/' + QUESTIONS_DATA.length + '</span><span class="score-label">Score</span></div>' +
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

    // 保存提交状态
    sessionStorage.setItem('reading_submitted_state', JSON.stringify({{
        formHTML: document.getElementById('quiz-form').innerHTML,
        resultsHTML: document.getElementById('results').innerHTML,
        timer: document.getElementById('timer-display').textContent,
        progress: document.getElementById('progress-count').textContent
    }}));
}}
</script>

</body>
</html>'''


def format_passage_html(passage_text):
    """把 passage 文本按段落分（双换行或单换行），包装成 <p>"""
    # 优先按双换行分段；如果没有双换行，按单换行
    paragraphs = re.split(r'\n\n+', passage_text.strip())
    if len(paragraphs) == 1:
        paragraphs = re.split(r'\n', passage_text.strip())
    return "\n".join(f"<p>{p.strip()}</p>" for p in paragraphs if p.strip())


# ============ 生成 history.html ============
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
body {
    font-family: "Georgia", serif;
    max-width: 860px;
    margin: 0 auto;
    padding: 24px 20px 60px;
    background: #fafaf7;
}
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
<header>
    <h1>READING PROGRESS</h1>
    <div class="subtitle">Daily reading practice history</div>
</header>
<div id="content"></div>
<script>
function render() {
    const sessions = JSON.parse(localStorage.getItem('reading_sessions') || '[]');
    const content = document.getElementById('content');
    if (sessions.length === 0) {
        content.innerHTML = '<div class="empty-state"><h2 style="border:none;padding:0;">📊 No sessions yet</h2><p>Complete a reading session to start tracking progress.</p></div>';
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
        html += '<tr>' +
            '<td>' + s.date + '<br><span style="color:#999;font-size:10px;">' + time + '</span></td>' +
            '<td>' + (s.title || '-') + '</td>' +
            '<td>' + (s.genre || '-') + '</td>' +
            '<td>' + s.correct + '/' + s.total + '</td>' +
            '<td style="color:' + accColor + ';font-weight:bold;">' + s.accuracy + '%</td>' +
            '<td>' + dur + '</td>' +
            '</tr>';
    });
    html += '</tbody></table><div class="clear-section"><button class="danger" onclick="if(confirm(\\'Clear all?\\')){localStorage.removeItem(\\'reading_sessions\\');render();}">Clear All History</button></div>';
    content.innerHTML = html;

    const ctx = document.getElementById('acc-chart');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: sessions.map(s => s.date),
            datasets: [{
                label: 'Accuracy',
                data: sessions.map(s => s.accuracy),
                borderColor: '#1a4d8f',
                backgroundColor: 'rgba(26,77,143,0.1)',
                borderWidth: 2.5, pointRadius: 5, tension: 0.3, fill: true
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true, max: 100, ticks: { callback: v => v + '%' } } }
        }
    });
}
render();
</script>
</body>
</html>'''


# ============ 生成 mistakes.html（生词回顾页）============
def render_mistakes_html():
    """这个版本只显示所有 data/ 里出现过的 vocab 词，作为复习页"""
    # 我们读所有 data/*.json，汇总所有 vocab
    all_vocab = {}
    for jf in DATA_DIR.glob("*.json"):
        try:
            with open(jf, encoding="utf-8") as f:
                d = json.load(f)
            for v in d.get("vocab", []):
                word = v["word"]
                if word not in all_vocab:
                    all_vocab[word] = {
                        "word": word,
                        "definition": v["definition"],
                        "first_seen": d.get("timestamp", ""),
                        "title": d.get("title", "")
                    }
        except Exception:
            continue

    vocab_data_json = json.dumps(list(all_vocab.values()), ensure_ascii=False)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Reading Vocabulary</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: "Georgia", serif; max-width: 760px; margin: 0 auto; padding: 24px 20px 60px; background: #fafaf7; }}
header {{ border-bottom: 3px double #333; padding-bottom: 16px; margin-bottom: 28px; }}
h1 {{ font-size: 24px; margin: 0 0 6px; }}
.subtitle {{ font-size: 13px; color: #555; font-style: italic; }}
.back-link {{ display: inline-block; margin-bottom: 20px; color: #1a4d8f; text-decoration: none; font-family: Arial, sans-serif; font-size: 14px; }}
.stats {{ background: white; padding: 14px 18px; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 24px; font-family: Arial, sans-serif; font-size: 14px; }}
.vocab-card {{ background: white; padding: 14px 18px; margin-bottom: 10px; border-left: 4px solid #1a4d8f; border-radius: 0 4px 4px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
.vocab-word {{ font-size: 18px; font-weight: bold; color: #1a4d8f; }}
.vocab-def {{ color: #333; font-size: 14px; margin-top: 4px; }}
.vocab-source {{ font-size: 11px; color: #999; margin-top: 6px; font-family: Arial, sans-serif; font-style: italic; }}
.empty-state {{ text-align: center; padding: 60px 20px; color: #888; }}
</style>
</head>
<body>
<a href="index.html" class="back-link">← Back to Reading</a>
<header>
    <h1>VOCABULARY COLLECTED</h1>
    <div class="subtitle">All words you've encountered in daily readings</div>
</header>
<div id="content"></div>
<script>
const VOCAB = {vocab_data_json};
const content = document.getElementById('content');
if (VOCAB.length === 0) {{
    content.innerHTML = '<div class="empty-state"><p>No words yet. Complete some readings first.</p></div>';
}} else {{
    let html = '<div class="stats"><strong>' + VOCAB.length + '</strong> words collected from daily readings</div>';
    VOCAB.sort((a, b) => a.word.localeCompare(b.word));
    VOCAB.forEach(v => {{
        html += '<div class="vocab-card">' +
            '<div class="vocab-word">' + v.word + '</div>' +
            '<div class="vocab-def">' + v.definition + '</div>' +
            (v.title ? '<div class="vocab-source">From: ' + v.title + '</div>' : '') +
            '</div>';
    }});
    content.innerHTML = html;
}}
</script>
</body>
</html>'''


# ============ 写入所有 HTML ============
with open(INDEX_FILE, "w", encoding="utf-8") as f:
    f.write(render_index_html())

with open(HISTORY_PAGE, "w", encoding="utf-8") as f:
    f.write(render_history_html())

with open(MISTAKES_FILE_HTML, "w", encoding="utf-8") as f:
    f.write(render_mistakes_html())

# ============ 总结输出 ============
print(f"📄 Files generated:")
print(f"   - {INDEX_FILE}     (today's reading)")
print(f"   - {HISTORY_PAGE}   (progress + chart)")
print(f"   - {MISTAKES_FILE_HTML}  (vocab collected)")
print(f"   - {data_file}      (session data)")
print(f"\n📊 Total readings completed: {history['session_count']}")
print(f"   Topics remaining: {len(topics) - len(history['used_topic_ids'])} / {len(topics)}")
print(f"\n✅ Done! Open index.html to start reading.")