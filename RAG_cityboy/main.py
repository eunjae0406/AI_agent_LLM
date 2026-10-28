import os
import tempfile
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub

# FastAPI 앱 초기화
app = FastAPI(title="Hyun Jin-geon RAG Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")

CHROMA_PERSIST_DIR = "./chroma_db"

# =========================================================================
# 💡 핵심 수정 부분: 구글 모델의 다차원 리스트 반환 버그를 고치는 래퍼 클래스
# =========================================================================
class SafeGoogleEmbeddings(GoogleGenerativeAIEmbeddings):
    def _flatten(self, vec):
        # 배열이 [[0.01...]] 처럼 여러 겹으로 쌓여 있다면 순수 숫자 리스트 [0.01...]가 나올 때까지 껍질을 벗김
        while isinstance(vec, list) and len(vec) > 0 and isinstance(vec[0], list):
            vec = vec[0]
        return vec

    def embed_documents(self, texts, **kwargs):
        vectors = super().embed_documents(texts, **kwargs)
        # 여러 문서의 벡터들의 껍질을 각각 벗겨서 반환
        return [self._flatten(v) for v in vectors]

    def embed_query(self, text, **kwargs):
        vector = super().embed_query(text, **kwargs)
        # 질문 벡터의 껍질을 벗겨서 반환
        return self._flatten(vector)

# 기존 임베딩 대신 껍질을 벗겨주는 커스텀 클래스 사용
embeddings = SafeGoogleEmbeddings(model="models/text-embedding-004")
# =========================================================================


class QueryRequest(BaseModel):
    query: str

@app.post("/upload")
async def upload_and_create_vectordb(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory=CHROMA_PERSIST_DIR
        )

        os.remove(tmp_path)
        return {"status": "success", "message": f"'{file.filename}' 벡터 DB 생성 완료"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_question(request: QueryRequest):
    if not os.path.exists(CHROMA_PERSIST_DIR):
        raise HTTPException(status_code=400, detail="벡터 DB가 아직 생성되지 않았습니다. PDF를 먼저 업로드하세요.")
        
    vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", streaming=True)
    
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        llm=llm
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    base_prompt = hub.pull("rlm/rag-prompt")
    system_message = (
        "당신은 소설 '운수 좋은 날'을 집필한 현진건 작가님입니다. "
        "근대 한국 문학의 현실주의 문체로 사용자의 질문에 답변하세요. "
        "다음 검색된 컨텍스트를 바탕으로 답변을 작성하시오.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{question}")
    ])

    rag_chain = (
        {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    retrieved_docs = retriever_from_llm.invoke(request.query)
    source_texts = [doc.page_content for doc in retrieved_docs]

    async def generate_response():
        yield json.dumps({"type": "source", "data": source_texts}) + "\n"
        
        async for chunk in rag_chain.astream(request.query):
            yield json.dumps({"type": "answer", "data": chunk}) + "\n"

    return StreamingResponse(generate_response(), media_type="application/x-ndjson")
