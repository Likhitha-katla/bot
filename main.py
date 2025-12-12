from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from models import call_llama
from vectorstore import (
    semantic_search,
    store_full_chat_data,
    ingest_chat,
    
    get_all_users,
    get_replies_after,
    load_memory,
    save_memory,
    get_topic_start,
    
    get_conn
)

# -------------------------------------------------------
# Build Context
# -------------------------------------------------------
def build_context(matches):
    ctx = ""
    for m in matches:
        md = m["metadata"]
        ctx += f"{md['userName']} ({md['createdOn']}): {md['text']}\n"
    return ctx

# -------------------------------------------------------
# LLM Response Generator
# -------------------------------------------------------
def generate_answer(question: str, context: str):

    system_prompt = (
        "You are an AI assistant designed to extract factual information from group chat messages.\n\n"
        "RULES:\n"
        "1. Use ONLY the chat context. Do not guess.\n"
        "2. If the topic is not in chat: \"The group has not discussed anything related to this topic.\"\n"
        "3. If user asks your opinion → respond: "
        "\"I am not trained to provide personal opinions or subjective viewpoints.\"\n"
        "4. If user greets (e.g., 'how are you') → reply neutrally.\n"
        "5. Keep responses short and factual."
    )

    user_prompt = f"Chat Context:\n{context}\n\nQuestion: {question}"

    answer = call_llama(system_prompt, user_prompt)
    return answer.strip()


# -------------------------------------------------------
# MAIN QA LOGIC
# -------------------------------------------------------
def chat_qa(question: str, group_id="101"):

    q = question.lower().strip()
    memory = load_memory()

    # 1️⃣ Who texted first
    if "who texted first" in q or "first message" in q:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT faiss_id, userName, createdOn, text
            FROM embeddings
            WHERE groupId=%s AND userName!='system'
            ORDER BY faiss_id ASC LIMIT 1
        """, (group_id,))
        row = cur.fetchone()
        conn.close()

        if not row:
            return "The group has no messages."

        faiss_id, user, t, text = row
        save_memory(last_faiss_id=faiss_id, last_message_text=text, last_topic="first_message")
        return f"{user} sent the first message:\n\"{text}\""

    # 2️⃣ User list
    if "how many users" in q or "users in group" in q:
        data = get_all_users(group_id)
        reply = f"There are {data['count']} users:\n"
        for i, u in enumerate(data["users"], start=1):
            reply += f"{i}. {u}\n"
        return reply

    # 3️⃣ Who started topic
    if "who started" in q and "about" in q:
        topic = q.split("about", 1)[1].strip()
        starter = get_topic_start(group_id, topic)

        if not starter:
            return "The group has not discussed anything related to this topic."

        save_memory(last_faiss_id=starter["faiss_id"], last_message_text=starter["text"], last_topic=topic)

        response = (
            f"{starter['user']} ({starter['time']}) started talking about {topic}:\n"
            f"\"{starter['text']}\"\n\n"
        )

        replies = get_replies_after(starter["faiss_id"])
        if replies:
            response += "Next messages:\n"
            for r in replies:
                response += f"{r['user']} ({r['time']}): {r['text']}\n"

            save_memory(
                last_faiss_id=replies[-1]["faiss_id"],
                last_message_text=replies[-1]["text"],
                last_topic=topic,
            )

        return response

    # 4️⃣ Follow-up questions
    follow = ["who replied", "reply for that", "what happened next", "continue", "and then"]
    if any(x in q for x in follow):
        if memory["last_faiss_id"] is None:
            return "I don't know which message you are referring to."

        replies = get_replies_after(memory["last_faiss_id"])
        if not replies:
            return "There are no more replies after that message."

        response = "Here are the next messages:\n\n"
        for r in replies:
            response += f"{r['user']} ({r['time']}): {r['text']}\n"

        save_memory(
            last_faiss_id=replies[-1]["faiss_id"],
            last_message_text=replies[-1]["text"],
            last_topic=memory["last_topic"],
        )
        return response

    # 5️⃣ Otherwise → semantic search
    matches = semantic_search(question, group_id)

    if not matches:
        return "No relevant messages found."

    first = matches[0]
    save_memory(
        last_faiss_id=first["faiss_id"],
        last_message_text=first["metadata"]["text"],
        last_topic=None,
    )

    context = build_context(matches)
    answer = generate_answer(question, context)
    return answer


# -------------------------------------------------------
# FLASK SERVER (Merged)
# -------------------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})   # Full CORS enabled
@app.get("/welcome")
def guest_api():
    return jsonify({"message": "Welcome to IR4U chatbot"})

@app.post("/chat")
def chat():
    try:
        data = request.get_json()
        question = data.get("question")
        group_id = str(data.get("group_id", "101"))

        answer = chat_qa(question, group_id)
        return jsonify({"answer": answer})

    except Exception as e:
        print("SERVER ERROR:", e)
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------------
# INIT + START SERVER
# -------------------------------------------------------
if __name__ == "__main__":
    print("📥 Loading chat data & generating FAISS index...")
    store_full_chat_data("cases.json")
    total = ingest_chat("cases.json")
    print(f"✔ Indexed {total} messages.")

    print("🚀 Flask AI Chat Server running on port 5000")
    app.run(host="0.0.0.0", port=5000)


