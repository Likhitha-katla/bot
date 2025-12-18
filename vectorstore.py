# vectorstore.py
import json
import mysql.connector
import faiss
import numpy as np
import os
from datetime import datetime
from models import get_embedding

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")
MYSQL_PORT=os.getenv("MYSQL_PORT")

# print("DEBUG MYSQL:", MYSQL_HOST, MYSQL_USER, MYSQL_DATABASE,MYSQL_PORT)

def get_conn():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE"),
        port=os.getenv("MYSQL_PORT"),
        ssl_ca="ca.pem",
        ssl_disabled=False
    )


# --------------------------------------------------
# VECTOR HELPERS
# --------------------------------------------------
def to_blob(vec):
    return np.asarray(vec, dtype=np.float32).tobytes()

def from_blob(blob):
    return np.frombuffer(blob, dtype=np.float32)

def _normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    return vecs / norms

# --------------------------------------------------
# 🔥 AUTO MIGRATION (NO MANUAL SQL)
# --------------------------------------------------
def ensure_createdOn_dt_column():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT COUNT(*)
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA=%s
          AND TABLE_NAME='embeddings'
          AND COLUMN_NAME='createdOn_dt'
    """, (MYSQL_DATABASE,))

    exists = cur.fetchone()[0]

    if not exists:
        print("🛠 Adding createdOn_dt column...")
        cur.execute("ALTER TABLE embeddings ADD COLUMN createdOn_dt DATETIME")

        print("🛠 Backfilling createdOn_dt...")
        cur.execute("""
            UPDATE embeddings
            SET createdOn_dt = STR_TO_DATE(
                REPLACE(REPLACE(createdOn, 'T', ' '), 'Z', ''),
                '%Y-%m-%d %H:%i:%s'
            )
        """)

        conn.commit()
        print(" createdOn_dt ready")

    conn.close()

# MEMORY TABLE (FOLLOW-UPS)
def ensure_memory_table():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversation_memory (
            id INT PRIMARY KEY,
            last_faiss_id INT NULL,
            last_message_text TEXT NULL,
            last_topic VARCHAR(255) NULL
        )
    """)

    cur.execute("SELECT COUNT(*) FROM conversation_memory WHERE id=1")
    if cur.fetchone()[0] == 0:
        cur.execute("""
            INSERT INTO conversation_memory
            VALUES (1, NULL, NULL, NULL)
        """)

    conn.commit()
    conn.close()

def load_memory():
    ensure_memory_table()
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT last_faiss_id, last_message_text, last_topic
        FROM conversation_memory WHERE id=1
    """)

    r = cur.fetchone()
    conn.close()

    return {
        "last_faiss_id": r[0],
        "last_message_text": r[1],
        "last_topic": r[2]
    }

def save_memory(last_faiss_id=None, last_message_text=None, last_topic=None):
    ensure_memory_table()
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        UPDATE conversation_memory
        SET last_faiss_id=%s,
            last_message_text=%s,
            last_topic=%s
        WHERE id=1
    """, (last_faiss_id, last_message_text, last_topic))

    conn.commit()
    conn.close()



def get_all_users(group_id: str):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT DISTINCT userName 
        FROM embeddings 
        WHERE groupId = %s AND userName != 'system'
    """, (str(group_id),))
    
    rows = cur.fetchall()
    conn.close()

    users = [r[0] for r in rows]
    return {"count": len(users), "users": users}

def get_replies_after(faiss_id: int):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT faiss_id, userName, createdOn, text
        FROM embeddings
        WHERE faiss_id > %s
        ORDER BY faiss_id ASC
        LIMIT 1
    """, (faiss_id,))

    row = cur.fetchone()
    conn.close()

    if not row:
        return []   

    # Return as list of a single reply
    return [{
        "faiss_id": row[0],
        "user": row[1],
        "time": row[2],
        "text": row[3]
    }]

def get_topic_start(group_id: str, keyword: str):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT faiss_id, userName, createdOn, text
        FROM embeddings
        WHERE LOWER(text) LIKE %s AND groupId=%s
        ORDER BY faiss_id ASC
        LIMIT 1
    """, (f"%{keyword.lower()}%", str(group_id)))

    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "faiss_id": row[0],
        "user": row[1],
        "time": row[2],
        "text": row[3]
    }

# TIMELINE QUERY (FIXED)
def get_messages_by_date(group_id: str, start: str, end: str):
    ensure_createdOn_dt_column()

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT userName, createdOn_dt, text
        FROM embeddings
        WHERE groupId=%s
          AND createdOn_dt BETWEEN %s AND %s
          AND userName!='system'
        ORDER BY createdOn_dt ASC
    """, (group_id, start, end))

    rows = cur.fetchall()
    conn.close()
    return rows

# DB INIT (SAFE FOR RE-INGEST)
def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS embeddings")
    cur.execute("DROP TABLE IF EXISTS faiss_index")

    cur.execute("""
    CREATE TABLE embeddings (
        faiss_id INT PRIMARY KEY,
        chatId VARCHAR(50),
        groupId VARCHAR(50),
        userName VARCHAR(255),
        createdOn VARCHAR(255),
        createdOn_dt DATETIME,
        text TEXT,
        image_url TEXT,
        image_context TEXT,
        message_type VARCHAR(50),
        embedding LONGBLOB
        )
    """)

    cur.execute("""
        CREATE TABLE faiss_index (
            id INT PRIMARY KEY,
            index_data LONGBLOB
        )
    """)

    conn.commit()
    conn.close()
    ensure_memory_table()
# STORE FULL CHAT JSON (RAW BACKUP)
def store_full_chat_data(json_path="cases.json"):
    """
    Stores the complete raw chat JSON into MySQL once.
    Useful for backup, re-ingestion, and debugging.
    """

    if not os.path.exists(json_path):
        print(f" Chat file not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        raw_json = f.read()

    conn = get_conn()
    cur = conn.cursor()

    # Create table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS full_chat_data (
            id INT PRIMARY KEY,
            data LONGTEXT
        )
    """)

    # Check if data already stored
    cur.execute("SELECT COUNT(*) FROM full_chat_data WHERE id=1")
    exists = cur.fetchone()[0]

    if exists == 0:
        cur.execute(
            "INSERT INTO full_chat_data (id, data) VALUES (1, %s)",
            (raw_json,)
        )
        conn.commit()
        print(" Full chat JSON stored in database")
    else:
        print("ℹFull chat JSON already exists (skipped)")

    conn.close()

# INGEST + FAISS

def ingest_chat(json_path="cases.json"):
    # Reset DB + tables
    init_db()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_msgs = []
    last_text = None

    # STEP 1: FLATTEN CHAT DATA
    for block in data:
        for msg in block["data"]:
            msg_type = msg.get("messageType")
            text = msg.get("message") or msg.get("question", {}).get("message")
            image_url = msg.get("images")

            image_context = None

            if text:
                last_text = text

            if msg_type == "image":
                image_context = msg.get("clinicalNotes") or last_text

            if not text and not image_url:
                continue

            all_msgs.append({
                "chatId": msg["chatId"],
                "groupId": msg["groupId"],
                "userName": msg["userName"],
                "createdOn": msg["createdOn"],
                "text": text,
                "image_url": image_url,
                "image_context": image_context,
                "message_type": msg_type
            })

    # STEP 2: INSERT + EMBED
    conn = get_conn()
    cur = conn.cursor()
    vectors = []

    for i, m in enumerate(all_msgs):

        # Decide what to embed
        content_for_embedding = (
            m["text"]
            or m["image_context"]
            or "image"
        )

        emb = get_embedding(content_for_embedding)
        vectors.append(emb)

        #  Parse createdOn safely in Python
        created_on_dt = None
        if m["createdOn"]:
            created_on_dt = datetime.strptime(
                m["createdOn"].replace("T", " ").replace("Z", ""),
                "%Y-%m-%d %H:%M:%S"
            )

        cur.execute("""
            INSERT INTO embeddings
            (
              faiss_id,
              chatId,
              groupId,
              userName,
              createdOn,
              createdOn_dt,
              text,
              image_url,
              image_context,
              message_type,
              embedding
            )
            VALUES (
              %s,%s,%s,%s,
              %s,%s,
              %s,%s,%s,%s,%s
            )
        """, (
            i,
            m["chatId"],
            m["groupId"],
            m["userName"],
            m["createdOn"],
            created_on_dt,
            m["text"],
            m["image_url"],
            m["image_context"],
            m["message_type"],
            to_blob(emb)
        ))

    conn.commit()

    # STEP 3: BUILD FAISS INDEX
    emb_array = _normalize(np.array(vectors, dtype="float32"))
    index = faiss.IndexFlatIP(emb_array.shape[1])
    index.add(emb_array)

    cur.execute(
        "INSERT INTO faiss_index (id, index_data) VALUES (1, %s)",
        (faiss.serialize_index(index).tobytes(),)
    )

    conn.commit()
    conn.close()

    print(f"✔ Ingested {len(vectors)} messages")
    return len(vectors)

# FAISS SEARCH
def load_faiss_index():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT index_data FROM faiss_index WHERE id=1")
    blob = cur.fetchone()[0]
    conn.close()

    return faiss.deserialize_index(np.frombuffer(blob, dtype=np.uint8))

def semantic_search(question: str, group_id: str):
    index = load_faiss_index()
    q = _normalize(np.array(get_embedding(question), dtype="float32").reshape(1, -1))

    scores, ids = index.search(q, index.ntotal)

    conn = get_conn()
    cur = conn.cursor()
    results = []

    for score, idx in zip(scores[0], ids[0]):
        cur.execute("""
            SELECT faiss_id,userName,createdOn,text,image_url,image_context
            FROM embeddings
            WHERE faiss_id=%s AND groupId=%s
        """, (int(idx), group_id))

        r = cur.fetchone()
        if r:
            results.append({
                "faiss_id": r[0],
                "score": float(score),   # 🔥 IMPORTANT
                "metadata": {
                    "userName": r[1],
                    "createdOn": r[2],
                    "text": r[3],
                    "image_url": r[4],
                    "image_context": r[5]
                }
            })

    conn.close()
    return results

