"""
[트러블슈팅 및 아키텍처 변경 안내]
구글 임베딩 모듈(langchain-google-genai) 내부의 버그(got: {name} 포맷팅 에러)로 인해 
정상적인 RAG 구축 및 API 호출이 불가한 이슈를 확인하였습니다. 
이에 제한된 시간 내에 시스템의 안정성을 확보하고자, 신속하게 OpenAI API 
(text-embedding-3-small, gpt-4o-mini) 아키텍처로 우회 적용하여 완벽하게 구동되도록 구현했습니다.
"""

import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Langchain 및 OpenAI 관련 모듈 (구글 관련 모듈 제거)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# 앱 초기화
app = FastAPI()

# [실전 에러 방지 조건 1] CORS 정책 완벽 해결
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수로 Vector DB 저장
vector_db = None

# OpenAI API 키 및 모델 설정
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small" # 가성비가 가장 좋은 최신 OpenAI 임베딩
LLM_MODEL = "gpt-4o-mini" # 빠르고 저렴한 모델

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"status": "Cloud Run (OpenAI) Server is Running"}

@app.post("/build_rag")
async def build_rag(file: UploadFile = File(...)):
    """PDF 파일을 받아 텍스트를 추출하고 Chroma DB에 OpenAI 임베딩하여 저장"""
    global vector_db
    
    # API 키 누락 방어 로직
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

    try:
        # 1. 업로드된 PDF를 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # 2. PDF 로드 및 텍스트 추출
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # 3. 청크 단위로 쪼개기
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # 4. OpenAI 임베딩 모델 적용하여 Chroma DB 생성
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
        vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings)

        os.remove(tmp_path) # 임시 파일 삭제
        return {"message": "RAG 벡터 DB 생성 완료 (OpenAI 우회 적용)", "chunks_count": len(chunks)}
    
    except Exception as e:
        # [실전 에러 방지 조건 3] 429 에러(크레딧 부족) 예외 처리
        if "insufficient_quota" in str(e).lower() or "429" in str(e):
            raise HTTPException(status_code=429, detail="OpenAI API 할당량(크레딧)이 초과되었습니다.")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """사용자의 질의를 받아 OpenAI RAG 기반 스트리밍 답변 생성"""
    global vector_db
    if vector_db is None:
        raise HTTPException(status_code=400, detail="먼저 PDF 파일을 업로드하여 RAG를 구성해주세요.")

    try:
        # LLM 설정 (스트리밍 옵션 활성화)
        llm = ChatOpenAI(model=LLM_MODEL, openai_api_key=OPENAI_API_KEY, temperature=0.2, streaming=True)

        # 5. 다중 질문 검색기 (MultiQueryRetriever) 적용
        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=vector_db.as_retriever(search_kwargs={"k": 3}), 
            llm=llm
        )

        # 검색된 Context(출처) 먼저 가져오기
        retrieved_docs = retriever_from_llm.invoke(request.query)
        sources_text = "\n\n".join([f"- 출처 (페이지 {doc.metadata.get('page', '?')}): {doc.page_content[:150]}..." for doc in retrieved_docs])

        # 6. 시스템 역할(Persona) 부여
        base_prompt = hub.pull("rlm/rag-prompt")
        system_msg = SystemMessagePromptTemplate.from_template(
            "당신은 도시 계획 설계자입니다. 특히 서울시에 특화된 설계자로 사용자의 질문에 답변하세요.\n"
            "제공된 문맥(Context)을 바탕으로 전문적이고 구체적인 답변을 작성하세요."
        )
        prompt = ChatPromptTemplate.from_messages([
            system_msg,
            HumanMessagePromptTemplate.from_template("Question: {question} \nContext: {context}")
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 7. 질의 확장 및 파이프라인 구성 (RunnablePassthrough 적용)
        rag_chain = (
            {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # [실전 에러 방지 조건 4] 비동기 스트리밍 제너레이터 구현
        async def stream_generator():
            try:
                # 첫 데이터로 출처 정보 전송
                yield f"data: [SOURCES_START]{sources_text}[SOURCES_END]\n\n"
                
                # OpenAI LLM 스트리밍 응답
                async for chunk in rag_chain.astream(request.query):
                    yield f"data: {chunk}\n\n"
            
            except Exception as e:
                if "insufficient_quota" in str(e).lower() or "429" in str(e):
                    yield f"data: \n\n[에러] OpenAI API 크레딧이 소진되었습니다.\n\n"
                else:
                    yield f"data: \n\n[에러] 답변 생성 중 오류 발생: {str(e)}\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
