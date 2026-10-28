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

# CORS 설정 (GAS 클라이언트에서의 호출을 허용하기 위해 추가)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 환경변수에서 Google AI API 키 로드 (Cloud Run 환경 변수에 등록해야 함)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")

# Chroma DB 설정
CHROMA_PERSIST_DIR = "./chroma_db"
# 구글 임베딩 모델 (요청하신 text-embedding-004 사용)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
# 질의 요청 모델
class QueryRequest(BaseModel):
    query: str

@app.post("/upload")
async def upload_and_create_vectordb(file: UploadFile = File(...)):
    """PDF 파일을 업로드받아 텍스트 추출, 청크 분할 후 Chroma DB에 저장합니다."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
    
    try:
        # 1. 임시 파일로 PDF 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # 2. PDF 텍스트 추출
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        # 3. 청크 단위로 쪼개기
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        # 4. Chroma DB에 벡터 저장
        Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, 
            persist_directory=CHROMA_PERSIST_DIR
        )

        # 임시 파일 삭제
        os.remove(tmp_path)
        return {"status": "success", "message": f"'{file.filename}' 벡터 DB 생성 완료"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_question(request: QueryRequest):
    """사용자 질의를 받아 MultiQueryRetriever로 검색 후 스트리밍으로 답변을 반환합니다."""
    
    # 벡터 DB 로드
    if not os.path.exists(CHROMA_PERSIST_DIR):
        raise HTTPException(status_code=400, detail="벡터 DB가 아직 생성되지 않았습니다. PDF를 먼저 업로드하세요.")
        
    vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    
    # LLM 초기화 (스트리밍 지원을 위해 streaming=True 설정)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", streaming=True)
    
    # 1. 다중 질문 검색기 (MultiQueryRetriever) 생성
    # 사용자의 질문을 LLM을 이용해 여러 개로 확장하여 검색 품질을 높임
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        llm=llm
    )
    
    # 문서 포맷팅 함수
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 2. 프롬프트 허브 사용 및 시스템 역할 부여
    # LangChain Hub에서 기본 RAG 프롬프트를 가져온 후, 시스템 메시지를 덮어씌워 역할 부여
    base_prompt = hub.pull("rlm/rag-prompt")
    system_message = (
        "당신은 소설 '운수 좋은 날'을 집필한 현진건 작가님입니다. "
        "근대 한국 문학의 현실주의 문체로 사용자의 질문에 답변하세요. "
        "다음 검색된 컨텍스트를 바탕으로 답변을 작성하시오.\n\n"
        "{context}"
    )
    # 기존 프롬프트를 요청하신 페르소나로 재구성
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{question}")
    ])

    # 3. LCEL 체인 구성 (RunnablePassthrough를 활용한 질의 확장 및 전달)
    rag_chain = (
        {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 4. 검색된 출처(Source)를 먼저 가져오기
    retrieved_docs = retriever_from_llm.invoke(request.query)
    source_texts = [doc.page_content for doc in retrieved_docs]

    # 5. 스트리밍 핸들러 제너레이터 함수 구현
    # 서버의 실시간 스트리밍 처리를 위해 astream을 사용하여 청크 단위로 응답
    async def generate_response():
        # 프론트엔드에서 파싱하기 쉽게 출처 데이터를 먼저 JSON 형태로 전송
        yield json.dumps({"type": "source", "data": source_texts}) + "\n"
        
        # LLM 답변 스트리밍 전송
        async for chunk in rag_chain.astream(request.query):
            yield json.dumps({"type": "answer", "data": chunk}) + "\n"

    # StreamingResponse로 반환 (SSE 형식)
    return StreamingResponse(generate_response(), media_type="application/x-ndjson")
