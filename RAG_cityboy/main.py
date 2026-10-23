import os
import json
import tempfile
import pdfplumber
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase.client import create_client

app = FastAPI(title="도시 정책 전문가 RAG 서버")

# CORS 미들웨어 설정 (Apps Script 클라이언트 요청 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 환경 변수 로드
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# API 키 누락 검증
if not all([GOOGLE_API_KEY, OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
    raise ValueError("서버 구동에 필요한 환경변수가 누락되었습니다.")

# Supabase 클라이언트 및 임베딩 설정
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

# LLM 객체 생성 (실시간 스트리밍 지원, Gemini 3.5 Flash Lite 적용)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite", 
    google_api_key=GOOGLE_API_KEY, 
    temperature=0, 
    streaming=True
)

class ChatRequest(BaseModel):
    question: str
    chat_history: List[Dict[str, str]] = [] # [{"role": "user", "content": "..."}]

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """PDF 파일을 업로드받아 텍스트 변환, 청크 분할, 벡터DB 저장을 수행하는 API"""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
    
    try:
        # 1. 임시 파일로 PDF 저장 및 텍스트 추출
        text_content = ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text_content += extracted + "\n"
        os.remove(tmp_path) # 임시 파일 삭제

        if not text_content.strip():
            raise HTTPException(status_code=422, detail="PDF에서 텍스트를 추출하지 못했습니다.")

        # 2. 텍스트 청크 단위로 쪼개기
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200, 
            separators=["\n\n", "\n", " ", ""]
        )
        texts = text_splitter.split_text(text_content)
        metadatas = [{"source": file.filename, "chunk_index": i} for i in range(len(texts))]

        # 3. 오픈 AI 임베딩 사용하여 Supabase 벡터 DB에 적용
        SupabaseVectorStore.from_texts(
            texts, 
            embeddings, 
            client=supabase_client, 
            table_name="documents", 
            query_name="match_documents",
            metadatas=metadatas
        )

        return {"message": "RAG 벡터 DB 생성이 성공적으로 완료되었습니다.", "chunks_created": len(texts)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 처리 중 오류 발생: {str(e)}")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """스트리밍 방식으로 질문에 대한 검색(RAG) 및 답변을 반환하는 API"""
    
    # 1. 대화 기록(Chat History)을 Langchain 메시지 객체로 변환
    history_messages = []
    for msg in request.chat_history:
        if msg.get("role") == "user":
            history_messages.append(HumanMessage(content=msg.get("content")))
        else:
            history_messages.append(AIMessage(content=msg.get("content")))
            
    # 2. Query Augmentation Chain 구성 (질의 확장)
    condense_system_prompt = """기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. 대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라"""
    
    condense_prompt = ChatPromptTemplate.from_messages([
        ("system", condense_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    query_augmentation_chain = condense_prompt | llm | StrOutputParser()

    # 3. 최종 답변 생성 프롬프트 구성 (시스템 메시지 설계)
    qa_system_prompt = """너는 문서에 기반해 답변하는 도시 정책 전문가야.
    
    [검색된 문서 내용]
    {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    rag_chain = qa_prompt | llm | StrOutputParser()

    # 4. 벡터 DB 및 리트리버 초기화
    vector_store = SupabaseVectorStore(
        client=supabase_client,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents"
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    async def stream_generator():
        try:
            # 4. 체인 실행 (데이터 주입) - 증강된 질문 생성
            # 대화 기록이 없으면 원본 질문을 그대로 사용, 있으면 질의 확장
            if history_messages:
                augmented_query = await query_augmentation_chain.ainvoke({
                    "question": request.question,
                    "chat_history": history_messages
                })
            else:
                augmented_query = request.question
                
            # 5. 증강된 질의로 벡터 DB 검색
            docs = retriever.invoke(augmented_query)
            context = "\n\n".join([f"[출처: {d.metadata.get('source', '알수없음')}]\n{d.page_content}" for d in docs])
            
            # 클라이언트에 검색 내용(RAG)을 먼저 전송
            initial_data = {
                "type": "retrieval", 
                "context": context, 
                "augmented_query": augmented_query
            }
            yield f"data: {json.dumps(initial_data, ensure_ascii=False)}\n\n"
            
            # 6. 주어진 청크 기반하여 언어 모델로 답변 생성 (스트리밍)
            async for chunk in rag_chain.astream({
                "context": context,
                "chat_history": history_messages,
                "question": augmented_query
            }):
                chunk_data = {"type": "chunk", "content": chunk}
                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                
            # 스트리밍 종료 신호
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            error_data = {"type": "error", "message": f"답변 생성 중 오류가 발생했습니다: {str(e)}"}
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

    # 실시간 스트리밍 핸들러 적용 (Server-Sent Events)
    return StreamingResponse(stream_generator(), media_type="text/event-stream")
