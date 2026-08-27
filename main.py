from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

if os.environ.get("OPENROUTER_API_KEY"):
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    MODEL = os.environ.get("MODEL", "google/gemma-4-31b-it:free")
else:
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    MODEL = os.environ.get("MODEL", "gpt-4o-mini")

def get_db():
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "db"),
        database=os.environ.get("DB_NAME", "chatbot"),
        user=os.environ.get("DB_USER", "myuser"),
        password=os.environ.get("DB_PASSWORD", "1234")
    )

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            role VARCHAR(20),
            content TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

@app.on_event("startup")
def startup():
    init_db()

class Message(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message": "OpenAI 챗봇 API 실행 중"}

@app.post("/chat")
async def chat(msg: Message):
    if not msg.message.strip():
        raise HTTPException(status_code=400, detail="메시지를 입력해주세요")

    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT role, content FROM messages ORDER BY created_at")
    rows = cur.fetchall()
    history = [{"role": r[0], "content": r[1]} for r in rows]

    cur.execute("INSERT INTO messages (role, content) VALUES (%s, %s)", ("user", msg.message))
    conn.commit()

    history.append({"role": "user", "content": msg.message})
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "당신은 친절한 AI 어시스턴트입니다. 한국어로 답변하세요."},
            *history
        ]
    )
    reply = response.choices[0].message.content

    cur.execute("INSERT INTO messages (role, content) VALUES (%s, %s)", ("assistant", reply))
    conn.commit()
    cur.close()
    conn.close()

    return {"reply": reply, "history_length": len(history) + 1}

@app.get("/history")
def get_history():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, role, content, created_at FROM messages ORDER BY created_at")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"id": r[0], "role": r[1], "content": r[2], "created_at": str(r[3])} for r in rows]

@app.delete("/chat/reset")
def reset():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM messages")
    conn.commit()
    cur.close()
    conn.close()
    return {"message": "대화 히스토리 초기화 완료"}
