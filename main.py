#main.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
from datetime import datetime
from calendar import monthrange
from models import call_llama
from vectorstore import (
    semantic_search,
    get_messages_by_date,
    load_memory,
    save_memory,
    get_conn,
    ingest_chat,
    store_full_chat_data
)

# CONTEXT BUILDER
def build_context(rows):
    return "\n".join(
        f"{u} ({t}): {txt}" for u, t, txt in rows
    )

# TIME FILTER EXTRACTION
def extract_time_filter(question: str):
    q = question.lower()

    m = re.search(
        r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\s+(to|-)\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})",
        q
    )

    if m:
        return {
            "start_month": m.group(1),
            "start_year": int(m.group(2)),
            "end_month": m.group(4),
            "end_year": int(m.group(5)),
        }

    return None

def resolve_date_range(f):
    sm = datetime.strptime(f["start_month"].title(), "%B").month
    em = datetime.strptime(f["end_month"].title(), "%B").month

    sy, ey = f["start_year"], f["end_year"]
    last_day = monthrange(ey, em)[1]

    return (
        f"{sy}-{sm:02d}-01 00:00:00",
        f"{ey}-{em:02d}-{last_day} 23:59:59"
    )
IMAGE_KEYWORDS = [
    "image", "images", "photo", "photos",
    "picture", "pictures",
    "show", "display", "angiogram"
]

def is_image_query(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in IMAGE_KEYWORDS)

# LLM ANSWER
def generate_answer(question: str, context: str):
    system_prompt = (
        "You are an AI assistant designed to extract factual information from group chat messages.\n\n"
        "RULES:\n"
        "1. Use ONLY the chat context. Do not guess.\n"
        "2. Apply the 'not discussed' rule ONLY when the user asks about a topic."
        "DO NOT apply this rule for date-based summaries.\n"
        "3. If user asks your opinion → respond: "
        "\"I am not trained to provide personal opinions or subjective viewpoints.\"\n"
        "4. If user greets (e.g., 'how are you') → reply neutrally.\n"
        "5. Keep responses short and factual."
    )

    user_prompt = f"Chat Context:\n{context}\n\nQuestion: {question}"
    return call_llama(system_prompt, user_prompt).strip()

# MAIN QA LOGIC
def chat_qa(question: str, group_id="101"):
    q = question.lower().strip()
    memory = load_memory()

    #  TIMELINE QUERY
    tf = extract_time_filter(question)
    if tf:
        start, end = resolve_date_range(tf)
        rows = get_messages_by_date(group_id, start, end)

        if not rows:
            return "No messages were found for the specified time period."

        context = build_context(rows)
        return generate_answer(question, context)

    #  FIRST MESSAGE
    if "who texted first" in q or "first message" in q:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT faiss_id, userName, createdOn, text
            FROM embeddings
            WHERE groupId=%s AND userName!='system'
            ORDER BY faiss_id ASC LIMIT 1
        """, (group_id,))
        r = cur.fetchone()
        conn.close()

        if not r:
            return "The group has no messages."

        save_memory(r[0], r[3], "first_message")
        return f"{r[1]} sent the first message:\n\"{r[3]}\""

    #  FOLLOW-UPS
    if any(x in q for x in ["who replied", "what happened next", "continue", "and then"]):
        if memory["last_faiss_id"] is None:
            return "I don't know which message you are referring to."

        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT faiss_id, userName, createdOn, text
            FROM embeddings
            WHERE faiss_id > %s
            ORDER BY faiss_id ASC LIMIT 1
        """, (memory["last_faiss_id"],))
        r = cur.fetchone()
        conn.close()

        if not r:
            return "There are no more replies after that message."

        save_memory(r[0], r[3], memory["last_topic"])
        return f"{r[1]} ({r[2]}): {r[3]}"

    matches = semantic_search(question, group_id)
    if not matches:
        return "No relevant messages found."

    image_intent = is_image_query(question)

    #  CASE 1: USER DID NOT ASK FOR IMAGES
    if not image_intent:
        text_only = [
            m for m in matches
            if not m["metadata"].get("image_url")
        ]

        if not text_only:
            return "The group discussed this topic, but no textual explanation is available."

        context = "\n".join(
            f'{m["metadata"]["userName"]} ({m["metadata"]["createdOn"]}): {m["metadata"]["text"]}'
            for m in text_only
        )

        first = text_only[0]
        save_memory(first["faiss_id"], first["metadata"]["text"], None)
        return generate_answer(question, context)


    #  CASE 2: USER ASKED FOR IMAGES
    SIMILARITY_THRESHOLD = 0.35   # tune if needed
    TOP_K_IMAGES = 5

    images = [
        {
            "url": m["metadata"]["image_url"],
            # "context": m["metadata"].get("image_context"),
            "postedBy": m["metadata"]["userName"],
            "time": m["metadata"]["createdOn"]
        }
        for m in matches
        if (
            m["metadata"].get("image_url")
            and m["score"] >= SIMILARITY_THRESHOLD
        )
    ]

    images = images[:TOP_K_IMAGES]

    if not images:
        return {
            "answer": "No relevant images were found for this query.",
            "images": []
        }

    return {
        "answer": "Relevant images from the discussion:",
        "images": images
    }

# FLASK APP
app = Flask(__name__)
CORS(app)

@app.post("/chat")
def chat():
    try:
        data = request.get_json()
        result = chat_qa(
            data.get("question", ""),
            str(data.get("group_id", "101"))
        )
        return jsonify(result)   
    except Exception as e:
        print("SERVER ERROR:", e)
        return jsonify({"answer": "Server error"}), 500


# START SERVER
if __name__ == "__main__":
    print("📥 Loading chat data & generating FAISS index...")
    store_full_chat_data("cases.json")
    total = ingest_chat("cases.json")
    print(f"✔ Indexed {total} messages.")

    print("🚀 Flask AI Chat Server running on port 5000")
    app.run(host="0.0.0.0", port=5000)

