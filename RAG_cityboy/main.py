import os
import tempfile
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
# 💡 에러 뿜는 구글 임베딩 대신, 등록해두신 OpenAI 키를 사용하는 구원투수!
from langchain_openai import OpenAIEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI(title="Hyun Jin-geon RAG Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHROMA_PERSIST_DIR = "./chroma_db"

# 💡 404 에러의 주범인 구글 임베딩을 빼버리고 OpenAI로 완벽 우회
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

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
        return {"status": "success", "message": f"'{file.filename}' 벡터 DB 생성 완료! (OpenAI 우회 성공)"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QueryRequest):
    if not os.path.exists(CHROMA_PERSIST_DIR):
        raise HTTPException(status_code=400, detail="벡터 DB가 아직 생성되지 않았습니다. PDF를 먼저 업로드하세요.")
        
    vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    
    # 💡 벡터 검색은 OpenAI가 하지만, 챗봇 대답은 원래대로 현진건(Gemini)이 담당합니다!
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", streaming=True)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

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
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    retrieved_docs = retriever.invoke(request.query)
    source_texts = [doc.page_content for doc in retrieved_docs]

    async def generate_response():
        yield json.dumps({"type": "source", "data": source_texts}) + "\n"
        
        async for chunk in rag_chain.astream(request.query):
            yield json.dumps({"type": "answer", "data": chunk}) + "\n"

    return StreamingResponse(generate_response(), media_type="application/x-ndjson")
