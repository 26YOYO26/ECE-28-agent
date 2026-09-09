import streamlit as st
import requests
import os
import tempfile
import hashlib
import io
from typing import List, Dict, Any

# Embeddings
from sentence_transformers import SentenceTransformer

# Document parsing
import pypdf
from docx import Document
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# Chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------- Page config ----------
st.set_page_config(page_title="Course AI Assistant | مساعد المقرر", page_icon="🎓", layout="wide")

# ---------- Translations ----------
translations = {
    "en": {
        "title": "Course AI Assistant",
        "sidebar_actions": "Actions",
        "upload_label": "Upload a file to add to knowledge base",
        "upload_button_save": "Save to knowledge base",
        "upload_button_temp": "Process temporarily",
        "chat_placeholder": "Ask a question about the course...",
        "thinking": "Thinking...",
        "no_context": "I couldn't find any relevant information in the course materials.",
        "processing": "Processing...",
        "file_processed_temp": "File processed. You can now ask questions about it (temporary).",
        "file_saved": "File saved to knowledge base successfully.",
        "no_text_extracted": "No clear text could be extracted. The file may be a video or image with unclear text.",
        "language_label": "Language / اللغة",
    },
    "ar": {
        "title": "مساعد المقرر الذكي",
        "sidebar_actions": "الإجراءات",
        "upload_label": "ارفع ملفًا لإضافته لقاعدة المعرفة",
        "upload_button_save": "حفظ في قاعدة المعرفة",
        "upload_button_temp": "معالجة مؤقتة",
        "chat_placeholder": "اسأل سؤالاً عن المقرر...",
        "thinking": "جارٍ التفكير...",
        "no_context": "لم أجد معلومات ذات صلة في مواد المقرر.",
        "processing": "جارٍ المعالجة...",
        "file_processed_temp": "تمت معالجة الملف. يمكنك الآن طرح أسئلة عنه (مؤقت).",
        "file_saved": "تم حفظ الملف في قاعدة المعرفة بنجاح.",
        "no_text_extracted": "تعذر استخراج نص واضح. قد يكون الملف فيديو أو صورة غير واضحة.",
        "language_label": "اللغة / Language",
    }
}

# Language setup (default English)
if "lang" not in st.session_state:
    st.session_state.lang = "en"

with st.sidebar:
    lang = st.radio(
        "Language / اللغة",
        options=["en", "ar"],
        index=0 if st.session_state.lang == "en" else 1,
        horizontal=True,
    )
    st.session_state.lang = lang

t = translations[lang]

# ---------- Secrets ----------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
GROQ_MODEL = "qwen/qwen3.8-27b"  # يمكن تغييره إلى allam-2-7b لتجنب الحدود

# ---------- Embeddings ----------
@st.cache_resource
def load_embedder():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embedder = load_embedder()

# ---------- Qdrant REST API helpers ----------
QDRANT_HEADERS = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}

def qdrant_collection_exists(name):
    url = f"{QDRANT_URL}/collections/{name}"
    resp = requests.get(url, headers=QDRANT_HEADERS)
    return resp.status_code == 200

def qdrant_create_collection(name, vector_size):
    url = f"{QDRANT_URL}/collections/{name}"
    payload = {"vectors": {"size": vector_size, "distance": "Cosine"}}
    resp = requests.put(url, headers=QDRANT_HEADERS, json=payload)
    if resp.status_code not in [200, 201]:
        st.error(f"Failed to create collection {name}: {resp.text}")

def qdrant_upsert_points(name, points):
    url = f"{QDRANT_URL}/collections/{name}/points?wait=true"
    payload = {"points": points}
    resp = requests.put(url, headers=QDRANT_HEADERS, json=payload)
    if resp.status_code not in [200, 201]:
        st.error(f"Failed to upsert points: {resp.text}")

def qdrant_delete_points_by_filter(name, file_id):
    url = f"{QDRANT_URL}/collections/{name}/points/delete?wait=true"
    payload = {"filter": {"must": [{"key": "file_id", "match": {"value": file_id}}]}}
    resp = requests.post(url, headers=QDRANT_HEADERS, json=payload)
    if resp.status_code not in [200, 201]:
        st.error(f"Failed to delete points: {resp.text}")

def qdrant_delete_points_by_ids(name, ids):
    url = f"{QDRANT_URL}/collections/{name}/points/delete?wait=true"
    payload = {"points": ids}
    resp = requests.post(url, headers=QDRANT_HEADERS, json=payload)
    if resp.status_code not in [200, 201]:
        st.error(f"Failed to delete points: {resp.text}")

def qdrant_search(name, vector, top_k=4):
    url = f"{QDRANT_URL}/collections/{name}/points/search"
    payload = {"vector": vector, "limit": top_k, "with_payload": True}
    resp = requests.post(url, headers=QDRANT_HEADERS, json=payload)
    if resp.status_code == 200:
        return resp.json()["result"]
    else:
        st.error(f"Search failed: {resp.text}")
        return []

def qdrant_scroll_all(name):
    url = f"{QDRANT_URL}/collections/{name}/points/scroll"
    payload = {"limit": 1000, "with_payload": True}
    resp = requests.post(url, headers=QDRANT_HEADERS, json=payload)
    if resp.status_code == 200:
        return resp.json()["result"]["points"]
    else:
        st.error(f"Scroll failed: {resp.text}")
        return []

# ---------- Collection names ----------
COLLECTION_NAME = "course_kb"
SYNC_COLLECTION = "sync_state"

# ---------- Helper functions ----------
def is_supported(mime_type: str) -> bool:
    supported = [
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/msword',
        'text/plain',
        'text/markdown',
        'application/rtf',
    ]
    return mime_type in supported

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    try:
        pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.warning(f"PDF text extraction failed, will try OCR: {e}")
    if len(text.strip()) < 100:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            images = convert_from_path(tmp_path)
            for img in images:
                text += pytesseract.image_to_string(img, lang='ara+eng') + "\n"
        except Exception as e:
            st.error(f"OCR failed: {e}")
        finally:
            os.unlink(tmp_path)
    return text

def extract_text_from_docx(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    doc = Document(tmp_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    os.unlink(tmp_path)
    return text

def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode('utf-8', errors='ignore')

def extract_text_from_image(file_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(image, lang='ara+eng')

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

def get_doctor_name(file_name: str) -> str:
    # نستخرج اسم الدكتور من اسم الملف (اختياري)
    base = os.path.splitext(file_name)[0]
    parts = base.replace('_', ' ').replace('-', ' ').split()
    if not parts:
        return "unknown"
    prefixes = ['د', 'dr', 'doctor']
    for i, p in enumerate(parts):
        if p.lower() in prefixes:
            if i + 1 < len(parts):
                return parts[i] + " " + parts[i+1]
            else:
                return parts[i]
    return parts[0]

def store_file_to_qdrant(file_bytes: bytes, file_name: str, mime_type: str):
    """حفظ الملف في قاعدة المعرفة Qdrant بشكل دائم."""
    # استخراج النص حسب النوع
    if mime_type == 'application/pdf':
        text = extract_text_from_pdf(file_bytes)
    elif 'word' in mime_type:
        text = extract_text_from_docx(file_bytes)
    elif mime_type.startswith('text'):
        text = extract_text_from_txt(file_bytes)
    elif mime_type.startswith('image'):
        text = extract_text_from_image(file_bytes)
    else:
        st.error("Unsupported file type.")
        return False

    if not text.strip():
        st.warning(t["no_text_extracted"])
        return False

    # إنشاء معرف فريد للملف (hash من الاسم + الوقت)
    unique_id = hashlib.md5(f"{file_name}_{datetime.now().isoformat()}".encode()).hexdigest()
    doctor = get_doctor_name(file_name)

    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()

    points_to_upsert = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        point_id = hashlib.md5(f"{unique_id}_{i}".encode()).hexdigest()
        points_to_upsert.append({
            "id": point_id,
            "vector": emb,
            "payload": {
                "file_id": unique_id,
                "file_name": file_name,
                "doctor_name": doctor,
                "chunk_index": i,
                "text": chunk
            }
        })

    qdrant_upsert_points(COLLECTION_NAME, points_to_upsert)

    # تحديث حالة المزامنة
    state_point_id = hashlib.md5(unique_id.encode()).hexdigest()
    qdrant_upsert_points(SYNC_COLLECTION, [{
        "id": state_point_id,
        "vector": [0.0],
        "payload": {"file_id": unique_id, "file_name": file_name, "modified_time": datetime.now().isoformat()}
    }])
    return True

# ---------- Retrieve context ----------
def retrieve_context(query: str, top_k: int = 4) -> str:
    query_embedding = embedder.encode([query]).tolist()[0]
    results = qdrant_search(COLLECTION_NAME, query_embedding, top_k)
    contexts = []
    for hit in results:
        payload = hit["payload"]
        contexts.append(f"From {payload['file_name']} (Dr. {payload['doctor_name']}):\n{payload['text']}")
    return "\n\n".join(contexts)

# ---------- Generate answer ----------
def generate_answer(question: str, context: str) -> str:
    if not context:
        return t["no_context"]

    language_instruction = ""
    if lang == "ar":
        language_instruction = "أجب باللغة العربية."
    else:
        language_instruction = "Answer in English."

    prompt = f"""You are an academic assistant for a course. Use ONLY the provided context to answer the question.
If the answer is not in the context, say: "{t['no_context']}"
Do not mention videos or any external content.
{language_instruction}

Context:
{context}

Question: {question}

Answer:"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 500   # تم تقليلها لتجنب خطأ 429
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error from Groq: {response.status_code} {response.text}"

# ---------- UI ----------
st.title(t["title"])

with st.sidebar:
    st.header(t["sidebar_actions"])

    st.markdown("---")
    st.write(t["upload_label"])
    uploaded_file = st.file_uploader(
        t["upload_label"],
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
        key="upload"
    )

    if uploaded_file:
        # نحتاج لقراءة الملف مرة واحدة، لكن يمكن تخزينه في session_state لتفادي إعادة القراءة
        if "uploaded_file_bytes" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.uploaded_file_bytes = uploaded_file.getvalue()
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.uploaded_file_type = uploaded_file.type

        col1, col2 = st.columns(2)
        with col1:
            if st.button(t["upload_button_save"], key="save_btn"):
                with st.spinner(t["processing"]):
                    success = store_file_to_qdrant(
                        st.session_state.uploaded_file_bytes,
                        st.session_state.uploaded_file_name,
                        st.session_state.uploaded_file_type
                    )
                    if success:
                        st.success(t["file_saved"])
        with col2:
            if st.button(t["upload_button_temp"], key="temp_btn"):
                with st.spinner(t["processing"]):
                    # معالجة مؤقتة فقط
                    mime = st.session_state.uploaded_file_type
                    if mime == 'application/pdf':
                        text = extract_text_from_pdf(st.session_state.uploaded_file_bytes)
                    elif 'word' in mime:
                        text = extract_text_from_docx(st.session_state.uploaded_file_bytes)
                    elif mime.startswith('image'):
                        text = extract_text_from_image(st.session_state.uploaded_file_bytes)
                    else:
                        text = extract_text_from_txt(st.session_state.uploaded_file_bytes)
                    if not text.strip():
                        st.error(t["no_text_extracted"])
                    else:
                        st.session_state.temp_context = text[:5000]
                        st.success(t["file_processed_temp"])

# منطقة المحادثة
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input(t["chat_placeholder"]):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(t["thinking"]):
            if "temp_context" in st.session_state:
                context = st.session_state.temp_context
                answer = generate_answer(prompt, context)
                # بعد الإجابة، يمكن مسح السياق المؤقت إذا أردت
                # لكن نتركه حتى يرفع ملف جديد
            else:
                context = retrieve_context(prompt)
                answer = generate_answer(prompt, context)
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
