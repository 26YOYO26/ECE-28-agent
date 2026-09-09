import streamlit as st
import requests
import os
import tempfile
import hashlib
import io
from typing import List, Dict, Any

# Google Drive
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

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
        "sync_button": "Sync with Google Drive",
        "upload_label": "Upload a file to ask about",
        "upload_button": "Process uploaded file",
        "chat_placeholder": "Ask a question about the course...",
        "thinking": "Thinking...",
        "no_context": "I couldn't find any relevant information in the course materials.",
        "sync_started": "Starting sync with Google Drive...",
        "sync_completed": "Sync completed.",
        "processing": "Processing...",
        "file_processed": "File processed. You can now ask questions about it.",
        "no_text_extracted": "No clear text could be extracted. The file may be a video or image with unclear text.",
        "language_label": "Language / اللغة",
    },
    "ar": {
        "title": "مساعد المقرر الذكي",
        "sidebar_actions": "الإجراءات",
        "sync_button": "مزامنة مع Google Drive",
        "upload_label": "ارفع ملفًا لتسأل عنه",
        "upload_button": "معالجة الملف المرفوع",
        "chat_placeholder": "اسأل سؤالاً عن المقرر...",
        "thinking": "جارٍ التفكير...",
        "no_context": "لم أجد معلومات ذات صلة في مواد المقرر.",
        "sync_started": "بدء المزامنة مع Google Drive...",
        "sync_completed": "اكتملت المزامنة.",
        "processing": "جارٍ المعالجة...",
        "file_processed": "تمت معالجة الملف. يمكنك الآن طرح أسئلة عنه.",
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
GOOGLE_DRIVE_API_KEY = st.secrets["GOOGLE_DRIVE_API_KEY"]
GOOGLE_DRIVE_FOLDER_ID = st.secrets["GOOGLE_DRIVE_FOLDER_ID"]
GROQ_MODEL = "qwen/qwen3.8-27b"  # تم التحديث

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
    payload = {
        "vectors": {
            "size": vector_size,
            "distance": "Cosine"
        }
    }
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
    payload = {
        "filter": {
            "must": [
                {"key": "file_id", "match": {"value": file_id}}
            ]
        }
    }
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
    payload = {
        "vector": vector,
        "limit": top_k,
        "with_payload": True
    }
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

# ---------- Global collection names ----------
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

# ---------- Sync from Google Drive ----------
def sync_drive():
    st.info(t["sync_started"])
    service = build('drive', 'v3', developerKey=GOOGLE_DRIVE_API_KEY)
    results = service.files().list(
        q=f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents and trashed=false",
        fields="files(id, name, mimeType, modifiedTime)",
        pageSize=1000
    ).execute()
    drive_files = results.get('files', [])
    st.write(f"Found {len(drive_files)} files in Drive folder.")

    # Ensure collections exist
    if not qdrant_collection_exists(COLLECTION_NAME):
        qdrant_create_collection(COLLECTION_NAME, 384)
    if not qdrant_collection_exists(SYNC_COLLECTION):
        qdrant_create_collection(SYNC_COLLECTION, 1)

    # Get current sync state
    db_files = {}
    points = qdrant_scroll_all(SYNC_COLLECTION)
    for p in points:
        payload = p.get("payload", {})
        db_files[payload.get("file_id")] = payload.get("modified_time")

    current_ids = set()

    for file in drive_files:
        file_id = file['id']
        name = file['name']
        mime = file['mimeType']
        modified = file['modifiedTime']
        current_ids.add(file_id)

        if not is_supported(mime):
            continue

        if file_id not in db_files or db_files[file_id] != modified:
            st.write(f"Processing {name}...")
            # Delete old vectors
            qdrant_delete_points_by_filter(COLLECTION_NAME, file_id)

            # Download file
            request = service.files().get_media(fileId=file_id)
            file_bytes = io.BytesIO()
            downloader = MediaIoBaseDownload(file_bytes, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()

            # Extract text
            if mime == 'application/pdf':
                text = extract_text_from_pdf(file_bytes.getvalue())
            elif 'word' in mime:
                text = extract_text_from_docx(file_bytes.getvalue())
            else:
                text = extract_text_from_txt(file_bytes.getvalue())

            if not text.strip():
                st.warning(f"{t['no_text_extracted']} ({name})")
                continue

            chunks = chunk_text(text)
            doctor = get_doctor_name(name)

            embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()

            points_to_upsert = []
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                point_id = hashlib.md5(f"{file_id}_{i}".encode()).hexdigest()
                points_to_upsert.append({
                    "id": point_id,
                    "vector": emb,
                    "payload": {
                        "file_id": file_id,
                        "file_name": name,
                        "doctor_name": doctor,
                        "chunk_index": i,
                        "text": chunk
                    }
                })
            qdrant_upsert_points(COLLECTION_NAME, points_to_upsert)

            # Update sync state
            state_point_id = hashlib.md5(file_id.encode()).hexdigest()
            qdrant_upsert_points(SYNC_COLLECTION, [{
                "id": state_point_id,
                "vector": [0.0],
                "payload": {"file_id": file_id, "modified_time": modified}
            }])
            st.success(f"Processed {name}")

    # Delete files no longer present
    for file_id in db_files:
        if file_id not in current_ids:
            st.write(f"Deleting {file_id} from vector DB...")
            qdrant_delete_points_by_filter(COLLECTION_NAME, file_id)
            state_point_id = hashlib.md5(file_id.encode()).hexdigest()
            qdrant_delete_points_by_ids(SYNC_COLLECTION, [state_point_id])
            st.success(f"Deleted {file_id}")

    st.success(t["sync_completed"])

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
        "max_tokens": 1024
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
    if st.button(t["sync_button"]):
        with st.spinner(t["sync_started"]):
            sync_drive()

    st.markdown("---")
    st.write(t["upload_label"])
    uploaded_file = st.file_uploader(
        t["upload_label"],
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
        key="upload"
    )
    if uploaded_file and st.button(t["upload_button"]):
        with st.spinner(t["processing"]):
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            mime = uploaded_file.type
            if mime == 'application/pdf':
                text = extract_text_from_pdf(open(tmp_path, 'rb').read())
            elif 'word' in mime:
                text = extract_text_from_docx(open(tmp_path, 'rb').read())
            elif mime.startswith('image'):
                text = extract_text_from_image(open(tmp_path, 'rb').read())
            else:
                text = extract_text_from_txt(open(tmp_path, 'rb').read())
            os.unlink(tmp_path)

            if not text.strip():
                st.error(t["no_text_extracted"])
            else:
                st.session_state["temp_context"] = text[:5000]
                st.success(t["file_processed"])

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
            else:
                context = retrieve_context(prompt)
                answer = generate_answer(prompt, context)
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
