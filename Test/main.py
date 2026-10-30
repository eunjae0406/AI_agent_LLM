import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Langchain 관련 모듈
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# 앱 초기화
app = FastAPI()

# [실전 에러 방지 조건 1] CORS 정책 완벽 해결
# GAS(클라이언트)에서 서버로 요청을 보낼 때 CORS 에러가 나지 않도록 모든 Origin 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 모든 도메인 허용 (실제 프로덕션에선 GAS 도메인만 지정 권장)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수로 Vector DB 저장 (간단한 구현을 위해 메모리 유지)
vector_db = None

# [실전 에러 방지 조건 5] API 사양 명확화 및 키 환경변수 로드
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
# 변경 후 
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-1.5-flash" # 답변 생성 및 다중 질의용 LLM

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"status": "Cloud Run AI Server is Running"}

@app.post("/build_rag")
async def build_rag(file: UploadFile = File(...)):
    """PDF 파일을 받아 텍스트를 추출하고 Chroma DB에 임베딩하여 저장하는 엔드포인트"""
    global vector_db
    try:
        # 1. 업로드된 PDF를 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # 2. PDF 로드 및 텍스트 추출
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # 3. 청크 단위로 쪼개기 (RecursiveCharacterTextSplitter)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # 4. 구글 임베딩 모델 적용하여 Chroma DB 생성
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)
        vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings)

        os.remove(tmp_path) # 임시 파일 삭제
        return {"message": "RAG 벡터 DB 생성 완료", "chunks_count": len(chunks)}
    
    except Exception as e:
        # [실전 에러 방지 조건 3] 429 에러 등 예외 처리 핸들링
        if "429" in str(e) or "quota" in str(e).lower():
            raise HTTPException(status_code=429, detail="API 할당량이 초과되었습니다. (크레딧 부족)")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """사용자의 질의를 받아 RAG 기반 스트리밍 답변을 생성하는 엔드포인트"""
    global vector_db
    if vector_db is None:
        raise HTTPException(status_code=400, detail="먼저 PDF 파일을 업로드하여 RAG를 구성해주세요.")

    try:
        # 기본 LLM 설정
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.2)

        # 5. 다중 질문 검색기 (MultiQueryRetriever) 적용
        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=vector_db.as_retriever(search_kwargs={"k": 3}), 
            llm=llm
        )

        # 검색된 Context(출처) 먼저 가져오기 (클라이언트에 출처 분리 표기를 위함)
        retrieved_docs = retriever_from_llm.invoke(request.query)
        sources_text = "\n\n".join([f"- 출처 (페이지 {doc.metadata.get('page', '?')}): {doc.page_content[:150]}..." for doc in retrieved_docs])

        # 6. 시스템 역할 부여 및 프롬프트 허브 사용
        # LangChain 허브에서 기본 RAG 프롬프트를 가져오되, 시스템 메시지를 덮어씌움
        base_prompt = hub.pull("rlm/rag-prompt")
        
        # [페르소나 적용] 서울시 특화 도시 계획 설계자
        system_msg = SystemMessagePromptTemplate.from_template(
            "당신은 도시 계획 설계자입니다. 특히 서울시에 특화된 설계자로 사용자의 질문에 답변하세요.\n"
            "제공된 문맥(Context)을 바탕으로 전문적이고 구체적인 답변을 작성하세요."
        )
        # 기존 hub 프롬프트의 context 부분과 조합
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
                # 첫 데이터로 출처(Context) 정보 전송 (JSON 형태 회피를 위해 구분자 사용)
                yield f"data: [SOURCES_START]{sources_text}[SOURCES_END]\n\n"
                
                # LLM 스트리밍 응답
                async for chunk in rag_chain.astream(request.query):
                    yield f"data: {chunk}\n\n"
            
            except Exception as e:
                # [실전 에러 방지 조건 3] 스트리밍 도중 발생하는 429 에러 방어
                if "429" in str(e) or "quota" in str(e).lower():
                    yield f"data: \n\n[에러] OpenAI/Google API 크레딧(할당량)이 소진되었습니다. 결제 상태를 확인하세요.\n\n"
                else:
                    yield f"data: \n\n[에러] 답변 생성 중 오류 발생: {str(e)}\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            raise HTTPException(status_code=429, detail="API 할당량이 초과되었습니다.")
        raise HTTPException(status_code=500, detail=str(e))
