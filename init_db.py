#init_db.py

from vectorstore import store_full_chat_data, ingest_chat

print("📥 Initializing DB and FAISS index...")
store_full_chat_data("cases.json")
total = ingest_chat("cases.json")
print("✔ Done. Indexed", total)
