# vectorstore.py
import json
import mysql.connector
import faiss
import numpy as np
from models import get_embedding
import os
# from dotenv import load_dotenv

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")
# print("DEBUG MYSQL:", MYSQL_HOST, MYSQL_USER, MYSQL_DATABASE)

def get_conn():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )


# VECTOR HELPERS
def to_blob(vec):
    return np.asarray(vec, dtype=np.float32).tobytes()

def from_blob(blob):
    return np.frombuffer(blob, dtype=np.float32)

def _normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    return vecs / norms


# MEMORY TABLE
def ensure_memory_table():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversation_memory (
            id INT PRIMARY KEY,
            last_faiss_id INT NULL,
            last_message_text TEXT NULL,
            last_topic VARCHAR(255) NULL
        );
    """)

    cur.execute("SELECT COUNT(*) FROM conversation_memory WHERE id = 1")
    if cur.fetchone()[0] == 0:
        cur.execute("""
            INSERT INTO conversation_memory (id, last_faiss_id, last_message_text, last_topic)
            VALUES (1, NULL, NULL, NULL)
        """)

    conn.commit()
    conn.close()


def save_memory(last_faiss_id=None, last_message_text=None, last_topic=None):
    ensure_memory_table()
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        UPDATE conversation_memory
        SET last_faiss_id = %s,
            last_message_text = %s,
            last_topic = %s
        WHERE id = 1
    """, (last_faiss_id, last_message_text, last_topic))

    conn.commit()
    conn.close()


def load_memory():
    ensure_memory_table()
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT last_faiss_id, last_message_text, last_topic
        FROM conversation_memory
        WHERE id = 1
    """)
    row = cur.fetchone()
    conn.close()

    return {
        "last_faiss_id": row[0],
        "last_message_text": row[1],
        "last_topic": row[2]
    }


# CREATE RAW DATA TABLES
def init_chat_tables():
    conn = get_conn()
    cur = conn.cursor()

    # GROUP TABLE
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_groups (
            id VARCHAR(50) PRIMARY KEY,
            name VARCHAR(255),
            description TEXT,
            category VARCHAR(255),
            subcategory VARCHAR(255),
            image VARCHAR(500),
            createdOn VARCHAR(255),
            updatedOn VARCHAR(255),
            createdBy VARCHAR(255)
        );
    """)

    # USERS TABLE
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_users (
            userName VARCHAR(255) PRIMARY KEY,
            firstName VARCHAR(255),
            lastName VARCHAR(255),
            isMute BOOLEAN
        );
    """)

    # RAW MESSAGES TABLE
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            chatId VARCHAR(50) PRIMARY KEY,
            groupId VARCHAR(50),
            userName VARCHAR(255),
            messageType VARCHAR(20),
            text TEXT,
            createdOn VARCHAR(255)
        );
    """)

    conn.commit()
    conn.close()


# STORE RAW JSON DATA
def store_full_chat_data(json_path="cases.json"):
    init_chat_tables()

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    conn = get_conn()
    cur = conn.cursor()

    for block in json_data:  
        group = block["chatGroupDetails"]
        messages = block.get("data", [])
        users = group.get("users", [])

        # ---------------- GROUP ----------------
        cur.execute("""
            REPLACE INTO chat_groups
            (id, name, description, category, subcategory, image, createdOn, updatedOn, createdBy)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            group["id"], group["name"], group["description"],
            group.get("category"), group.get("subcategory"),
            group["image"], group["CreatedOn"], group["UpdatedOn"],
            group["createdBy"]
        ))

        # ---------------- USERS ----------------
        for u in users:
            cur.execute("""
                REPLACE INTO chat_users (userName, firstName, lastName, isMute)
                VALUES (%s, %s, %s, %s)
            """, (u["userName"], u["firstName"], u["lastName"], u["isMute"]))

        # ---------------- MESSAGES ----------------
        for msg in messages:
            # extract correct text
            if msg["messageType"] == "message":
                txt = msg.get("message")
            else:
                txt = msg.get("question", {}).get("message")

            cur.execute("""
                REPLACE INTO chat_messages
                (chatId, groupId, userName, messageType, text, createdOn)
                VALUES (%s,%s,%s,%s,%s,%s)
            """, (
                msg["chatId"], msg["groupId"], msg["userName"],
                msg["messageType"], txt, msg["createdOn"]
            ))

    conn.commit()
    conn.close()
    print("✔ All chat groups + messages stored successfully!")

# RESET EMBEDDING TABLES
def init_db():
    print("\n🗄️ Resetting MySQL schema for embeddings/FAISS...")

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
            text TEXT,
            embedding LONGBLOB
        );
    """)

    cur.execute("""
        CREATE TABLE faiss_index (
            id INT PRIMARY KEY,
            index_data LONGBLOB
        );
    """)

    conn.commit()
    conn.close()

    ensure_memory_table()


# INGEST + EMBEDDINGS + FAISS
def ingest_chat(json_path="cases.json") -> int:
    init_db()

    print("\n📥 Loading chat.json...")
    with open(json_path, "r", encoding="utf-8") as f:
        json_list = json.load(f)

    all_messages = []

    for block in json_list:
        group = block["chatGroupDetails"]
        group_id = group["id"]
        messages = block["data"]
        users = group.get("users", [])

        # METADATA messages for each group
        virtual = []

        # virtual.append({
        #     "chatId": f"meta_creator_{group_id}",
        #     "groupId": group_id,
        #     "userName": "system",
        #     "messageType": "message",
        #     "message": f"The group was created by {group['createdBy']}.",
        #     "createdOn": group["CreatedOn"]
        # })

        # virtual.append({
        #     "chatId": f"meta_groupname_{group_id}",
        #     "groupId": group_id,
        #     "userName": "system",
        #     "messageType": "message",
        #     "message": f"The group name is {group['name']}.",
        #     "createdOn": group["CreatedOn"]
        # })

        # if group.get("description"):
        #     virtual.append({
        #         "chatId": f"meta_description_{group_id}",
        #         "groupId": group_id,
        #         "userName": "system",
        #         "messageType": "message",
        #         "message": f"Group description: {group['description']}",
        #         "createdOn": group["CreatedOn"]
        #     })

        user_list = ", ".join([u["userName"] for u in users])
        virtual.append({
            "chatId": f"meta_users_{group_id}",
            "groupId": group_id,
            "userName": "system",
            "messageType": "message",
            "message": f"Users in this group: {user_list}",
            "createdOn": group["CreatedOn"]
        })

        # merge
        all_messages.extend(virtual)

        # REAL chat messages
        for msg in messages:
            if msg["messageType"] == "message":
                text = msg.get("message")
            else:
                text = msg.get("question", {}).get("message")

            if not text:
                continue

            all_messages.append({
                "chatId": msg["chatId"],
                "groupId": msg["groupId"],
                "userName": msg["userName"],
                "messageType": msg["messageType"],
                "message": text,
                "createdOn": msg["createdOn"]
            })

    # Now we index ALL messages from all groups
    conn = get_conn()
    cur = conn.cursor()

    vectors = []
    faiss_id = 0

    for msg in all_messages:
        text = msg["message"]
        emb = get_embedding(text)

        cur.execute("""
            INSERT INTO embeddings
            (faiss_id, chatId, groupId, userName, createdOn, text, embedding)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
        """, (
            faiss_id, msg["chatId"], msg["groupId"], msg["userName"],
            msg["createdOn"], text, to_blob(emb)
        ))

        vectors.append(emb)
        faiss_id += 1

    conn.commit()

    emb_array = np.array(vectors).astype("float32")
    emb_array = _normalize(emb_array)

    dim = emb_array.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_array)

    index_bytes = faiss.serialize_index(index).tobytes()
    cur.execute("INSERT INTO faiss_index (id, index_data) VALUES (1, %s)", (index_bytes,))

    conn.commit()
    conn.close()

    print(f"✔ Ingestion finished. Indexed {len(vectors)} messages.")
    return len(vectors)

# # LOAD FAISS INDEX
def load_faiss_index():
    print("\n📦 Loading FAISS index...")

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT index_data FROM faiss_index WHERE id = 1")
    row = cur.fetchone()
    conn.close()

    if not row:
        raise RuntimeError("❌ No FAISS index found!")

    index_blob = row[0]
    index = faiss.deserialize_index(np.frombuffer(index_blob, dtype=np.uint8))

    print("✔ FAISS index loaded.")
    return index

# @st.cache_resource
# def load_faiss_index():
#     print("📦 Loading FAISS index (cached)...")

#     conn = get_conn()
#     cur = conn.cursor()
#     cur.execute("SELECT index_data FROM faiss_index WHERE id = 1")
#     row = cur.fetchone()
#     conn.close()

#     if not row:
#         raise RuntimeError("❌ No FAISS index found in DB!")

#     index_bytes = row[0]
#     index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8))

#     print("✔ FAISS index loaded")
#     return index


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


# SEMANTIC SEARCH
def semantic_search(question: str, group_id: str, top_k: int = None):
    print("\n🔍 Running semantic search...")
    index = load_faiss_index()

    q_emb = np.array(get_embedding(question)).astype("float32").reshape(1, -1)
    q_emb = _normalize(q_emb)

    k = index.ntotal if top_k is None else min(top_k, index.ntotal)

    distances, indices = index.search(q_emb, k)

    conn = get_conn()
    cur = conn.cursor()

    results = []

    for score, idx in zip(distances[0], indices[0]):
        idx = int(idx)
        if idx == -1:
            continue

        cur.execute("""
            SELECT faiss_id, chatId, groupId, userName, createdOn, text
            FROM embeddings
            WHERE faiss_id=%s
        """, (idx,))

        row = cur.fetchone()
        if row and row[2] == str(group_id):
            results.append({
                "faiss_id": row[0],
                "metadata": {
                    "userName": row[3],
                    "createdOn": row[4],
                    "text": row[5]
                }
            })

    conn.close()

    return sorted(results, key=lambda x: x["faiss_id"])


# COUNT MESSAGES
def count_messages_in_group(group_id: str) -> int:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM embeddings WHERE groupId=%s", (str(group_id),))
    count = cur.fetchone()[0]
    conn.close()

    return count
