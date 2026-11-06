"""
================================================================================
파일명: main.py
역할: FastAPI 기반 Advanced RAG 백엔드 서버 (Google Cloud Run 배포 및 GAS 연동용)
핵심 파이프라인:
  1. 문서 적재 및 청킹 계층 (PDF Parsing & Chunking Layer)
  2. 하이브리드 인덱싱 계층 (FAISS IndexFlatL2 / IndexHNSWFlat + BM25 Sparse Index)
  3. 다중 쿼리 확장 계층 (Multi-Query Layer: 단일 질의 -> 5가지 관점 파생 쿼리 생성)
  4. 하이브리드 검색 계층 (Hybrid Search Layer: 5개 쿼리 x [FAISS Dense + BM25 Sparse])
  5. 상호 순위 융합 계층 (RAG Fusion & RRF Layer: Reciprocal Rank Fusion 재정렬)
  6. 실시간 스트리밍 핸들러 (StreamHandler: 단계별 진행 상태 및 복잡도 실시간 전송)

알고리즘 복잡도 요약 (Algorithm Complexity):
  - 문서 청킹 및 토크나이징: 시간 O(N), 공간 O(N) (N: 전체 텍스트 길이)
  - FAISS IndexFlatL2 구축: 시간 O(M * D), 검색: 시간 O(Q * M * D) (M: 청크 수, D: 임베딩 차원, Q: 확장 쿼리 수=5)
  - FAISS IndexHNSWFlat 구축: 시간 O(M * log(M) * D), 검색: 시간 O(Q * log(M) * D)
  - BM25(Okapi) 구축: 시간 O(M * L), 검색: 시간 O(Q * |q| * M_posting) (L: 평균 문서 길이, |q|: 쿼리 토큰 수)
  - RRF(Reciprocal Rank Fusion) 융합 및 정렬: 시간 O(Q * K * log(Q * K)), 공간 O(M) (K: 경로별 상위 검색 수)
================================================================================
"""

import base64
import hashlib
import io
import json
import logging
import os
import re
import time
import traceback
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger("advanced_rag_server")

# FastAPI 애플리케이션 초기화
app = FastAPI(
    title="Advanced RAG Fusion Server (FAISS + BM25 + Multi-Query + RRF)",
    description="GAS(Google Apps Script) 연동용 하이브리드 RAG 백엔드 서버",
    version="1.0.0",
)

# CORS 설정 (GAS 및 웹 클라이언트 요청 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# 0. 알고리즘 복잡도 상수 정의 (UI 및 응답 메타데이터에 포함)
# ==============================================================================
ALGORITHM_COMPLEXITY = {
    "document_chunking": {
        "name": "Sliding Window Document Chunking",
        "time_complexity": "O(N)",
        "space_complexity": "O(N)",
        "description": "N바이트 PDF 텍스트를 고정 크기(chunk_size)와 중첩(overlap) 단위로 분할",
    },
    "multi_query_expansion": {
        "name": "LLM Multi-Query Expansion (5 Perspectives)",
        "time_complexity": "O(1) API Call / O(T_gen)",
        "space_complexity": "O(Q) (Q=5)",
        "description": "단일 사용자 쿼리를 5가지 서로 다른 검색 관점(정의/원인/비교/사례/키워드)으로 확장",
    },
    "faiss_flat_l2": {
        "name": "FAISS IndexFlatL2 (Exact Dense Vector Search)",
        "build_time_complexity": "O(M * D)",
        "search_time_complexity": "O(Q * M * D)",
        "space_complexity": "O(M * D)",
        "description": "M개 문서 벡터(차원 D)에 대해 Q(5)개 쿼리와의 유클리드(L2) 거리를 전수 계산",
    },
    "faiss_hnsw_flat": {
        "name": "FAISS IndexHNSWFlat (Approximate Graph Vector Search)",
        "build_time_complexity": "O(M * log(M) * D)",
        "search_time_complexity": "O(Q * log(M) * D)",
        "space_complexity": "O(M * (D + M_links))",
        "description": "계층적 탐색 가능 소세계 그래프(HNSW)를 통해 로그 시간 복잡도로 근사 최근접 이웃 탐색",
    },
    "bm25_sparse": {
        "name": "BM25Okapi Sparse Keyword Search",
        "build_time_complexity": "O(M * L_avg)",
        "search_time_complexity": "O(Q * |q| * M)",
        "space_complexity": "O(M + V)",
        "description": "TF-IDF 및 문서 길이 정규화를 반영한 희소 키워드 매칭 (V: 어휘 사전 크기)",
    },
    "rrf_fusion": {
        "name": "Reciprocal Rank Fusion (RRF Re-ranking)",
        "time_complexity": "O(Q * K * log(Q * K))",
        "space_complexity": "O(U) (U <= 2 * Q * K)",
        "formula": "RRF(d) = Σ_{m ∈ {Dense, Sparse}} Σ_{q=1..5} [ w_m / (k + rank_{m,q}(d)) ], k=60",
        "description": "10개 검색 경로(5개 쿼리 x 2개 검색기)의 순위를 역순위 합산으로 융합 및 재정렬",
    },
}


# ==============================================================================
# 1. 실시간 스트리밍 핸들러 (StreamHandler)
# ==============================================================================
class StreamHandler:
    """
    서버의 파이프라인 진행 상황(청킹 -> 인덱싱 -> 다중 쿼리 생성 -> 하이브리드 검색 -> RRF 융합 -> 답변 생성)을
    NDJSON(Newline Delimited JSON) 스트림 형식으로 클라이언트에 실시간 전달하는 핸들러 클래스.
    """

    @staticmethod
    def format_event(
        stage: str,
        status: str,
        message: str,
        progress: int = 0,
        data: Optional[Dict[str, Any]] = None,
        complexity: Optional[Dict[str, str]] = None,
    ) -> str:
        """단일 스트리밍 이벤트를 JSON 라인 문자열로 직렬화합니다."""
        payload = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stage": stage,           # 현재 실행 계층 식별자
            "status": status,         # processing | completed | error
            "message": message,       # 사용자에게 표시할 상태 메시지
            "progress": progress,     # 진행률 (0 ~ 100)
            "complexity": complexity or {},
            "data": data or {},
        }
        return json.dumps(payload, ensure_ascii=False) + "\n"


# ==============================================================================
# 2. 데이터 모델 정의 (Pydantic Schemas)
# ==============================================================================
class IngestRequest(BaseModel):
    """PDF 파일 또는 텍스트를 받아 벡터 DB 및 BM25 인덱스를 생성하는 요청 스키마"""
    filename: str = Field(..., description="업로드된 PDF 파일명")
    pdf_base64: Optional[str] = Field(None, description="Base64 인코딩된 PDF 바이너리 데이터")
    raw_text: Optional[str] = Field(None, description="직접 입력된 텍스트 데이터(옵션)")
    index_type: str = Field("IndexFlatL2", description="FAISS 인덱스 종류: IndexFlatL2 또는 IndexHNSWFlat")
    chunk_size: int = Field(500, ge=100, le=2000, description="청크당 최대 글자 수")
    chunk_overlap: int = Field(100, ge=0, le=500, description="인접 청크 간 중첩 글자 수")


class QueryRequest(BaseModel):
    """사용자 질의를 받아 Multi-Query -> Hybrid Search -> RRF -> LLM 응답을 수행하는 요청 스키마"""
    query: str = Field(..., min_length=1, description="사용자의 단일 입력 질의")
    top_k: int = Field(5, ge=1, le=20, description="RRF 재정렬 후 최종 반환할 상위 문서 수")
    dense_weight: float = Field(0.5, ge=0.0, le=1.0, description="FAISS Dense 검색 가중치")
    sparse_weight: float = Field(0.5, ge=0.0, le=1.0, description="BM25 Sparse 검색 가중치")
    rrf_k: int = Field(60, ge=1, le=200, description="RRF 공식의 평활화 상수 k (기본값 60)")
    stream: bool = Field(False, description="스트리밍 응답 여부")


# ==============================================================================
# 3. 전역 인메모리 하이브리드 인덱스 저장소 (FAISS + BM25 State Store)
# ==============================================================================
class HybridRAGStore:
    """
    FAISS Dense Index와 BM25 Sparse Index, 그리고 원본 문서 메타데이터를 관리하는 핵심 엔진.
    """

    def __init__(self) -> None:
        self.embedding_dim: int = 384  # 기본 임베딩 벡터 차원
        self.index_type: str = "IndexFlatL2"
        self.faiss_index: Optional[faiss.Index] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.source_filename: Optional[str] = None
        self.last_updated: Optional[str] = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        한국어/영어/숫자 복합 텍스트에서 형태소·단어 단위 토큰 및 바이그램(2-gram)을 추출합니다.
        시간 복잡도: O(L) (L: 텍스트 길이)
        """
        cleaned = re.sub(r"[^\w\s가-힣]", " ", text.lower())
        words = [w for w in cleaned.split() if len(w) >= 1]
        # 한국어 조사 결합 시 매칭률을 높이기 위해 2음절 이상 단어의 서브스트링 토큰 보강
        tokens = list(words)
        for w in words:
            if len(w) >= 3:
                tokens.append(w[:2])
                tokens.append(w[-2:])
        return tokens if tokens else ["empty"]

    def compute_embedding(self, text: str) -> np.ndarray:
        """
        텍스트를 고정 차원(self.embedding_dim) Dense Vector로 임베딩합니다.
        - GEMINI_API_KEY 또는 OPENAI_API_KEY 환경변수가 설정된 경우 상용 임베딩 API를 호출하며,
        - API 키가 없거나 네트워크가 제한된 환경에서는 결정론적 의미 해싱(Feature Hashing + N-gram Projection)을
          사용하여 외부 의존성 없이도 즉각적인 L2 벡터 유사도 계산을 보장합니다.
        """
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                res = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                )
                vec = np.array(res["embedding"], dtype=np.float32)
                self.embedding_dim = len(vec)
                norm = np.linalg.norm(vec)
                return vec / norm if norm > 0 else vec
            except Exception as e:
                logger.warning(f"Gemini 임베딩 호출 실패, 로컬 의미 해싱 벡터로 대체합니다: {e}")

        # 결정론적 N-gram Feature Hashing 임베딩 (O(L) 시간 복잡도)
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        tokens = self.tokenize(text)
        # 단어 및 문자 바이그램/트라이그램 피처 투영
        cleaned = re.sub(r"\s+", " ", text.lower().strip())
        char_ngrams = [cleaned[i : i + 3] for i in range(max(1, len(cleaned) - 2))]

        for feature in tokens * 2 + char_ngrams:
            digest = hashlib.md5(feature.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.embedding_dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vec[idx] += sign

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def build_indices(
        self,
        chunks: List[Dict[str, Any]],
        index_type: str = "IndexFlatL2",
    ) -> Dict[str, Any]:
        """
        문서 청크 리스트로부터 FAISS 인덱스(IndexFlatL2 또는 IndexHNSWFlat)와
        BM25 희소 인덱스를 동시에 구축합니다.
        """
        if not chunks:
            raise ValueError("인덱싱할 문서 청크가 비어 있습니다. PDF 파일에 추출 가능한 텍스트가 있는지 확인하세요.")

        start_time = time.perf_counter()
        self.documents = chunks
        self.index_type = index_type

        # 1. 모든 청크에 대해 임베딩 벡터 계산
        embeddings_list: List[np.ndarray] = []
        for doc in chunks:
            emb = self.compute_embedding(doc["content"])
            embeddings_list.append(emb)

        embeddings_matrix = np.vstack(embeddings_list).astype(np.float32)
        num_docs, dim = embeddings_matrix.shape
        self.embedding_dim = dim

        # 2. FAISS 인덱스 생성 (IndexFlatL2 또는 IndexHNSWFlat 선택)
        if index_type == "IndexHNSWFlat":
            # HNSW 그래프 기반 근사 탐색 인덱스 (M_links=32)
            index = faiss.IndexHNSWFlat(dim, 32)
            index.hnsw.efConstruction = 40
            index.hnsw.efSearch = 32
        else:
            # 정확한 L2 유클리드 거리 계산 인덱스
            self.index_type = "IndexFlatL2"
            index = faiss.IndexFlatL2(dim)

        index.add(embeddings_matrix)
        self.faiss_index = index

        # 3. BM25 키워드 희소 인덱스 생성
        self.tokenized_corpus = [self.tokenize(doc["content"]) for doc in chunks]
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        self.last_updated = time.strftime("%Y-%m-%d %H:%M:%S")

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "total_chunks": num_docs,
            "embedding_dimension": dim,
            "faiss_index_type": self.index_type,
            "bm25_vocab_size": len(self.bm25_index.idf) if self.bm25_index else 0,
            "elapsed_ms": elapsed_ms,
        }


# 싱글톤 RAG 저장소 인스턴스 생성
rag_store = HybridRAGStore()


# ==============================================================================
# 4. 핵심 파이프라인 함수 (PDF 파싱 / Multi-Query 5x / Hybrid Search / RRF / LLM 답변)
# ==============================================================================
def extract_chunks_from_pdf(
    pdf_bytes: bytes,
    filename: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    PDF 바이너리에서 페이지별 텍스트를 추출하고 슬라이딩 윈도우 방식으로 청킹합니다.
    시간 복잡도: O(N) (N: 전체 텍스트 길이)
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    chunks: List[Dict[str, Any]] = []
    chunk_id = 0

    for page_idx, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if not page_text:
            continue

        # 연속 공백 정리
        cleaned_text = re.sub(r"\s+", " ", page_text)
        step = max(1, chunk_size - chunk_overlap)

        for start in range(0, len(cleaned_text), step):
            snippet = cleaned_text[start : start + chunk_size].strip()
            if len(snippet) < 20 and len(chunks) > 0:
                continue
            chunks.append(
                {
                    "doc_id": f"chunk_{chunk_id}",
                    "chunk_index": chunk_id,
                    "source": filename,
                    "page": page_idx,
                    "char_Count": len(snippet),
                    "content": snippet,
                }
            )
            chunk_id += 1

    return chunks


def extract_chunks_from_text(
    raw_text: str,
    filename: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """일반 텍스트 입력을 슬라이딩 윈도우 방식으로 청킹합니다."""
    cleaned_text = re.sub(r"\s+", " ", raw_text.strip())
    if not cleaned_text:
        return []
    step = max(1, chunk_size - chunk_overlap)
    chunks: List[Dict[str, Any]] = []
    for idx, start in enumerate(range(0, len(cleaned_text), step)):
        snippet = cleaned_text[start : start + chunk_size].strip()
        if snippet:
            chunks.append(
                {
                    "doc_id": f"chunk_{idx}",
                    "chunk_index": idx,
                    "source": filename,
                    "page": (idx // 3) + 1,
                    "char_Count": len(snippet),
                    "content": snippet,
                }
            )
    return chunks


def generate_multi_queries(original_query: str) -> List[Dict[str, str]]:
    """
    [다중 쿼리 확장 계층 (Multi-Query Layer)]
    사용자의 단일 입력 쿼리를 LLM(또는 다각도 규칙 엔진)을 통해 서로 다른 5가지 관점의 파생 쿼리로 확장합니다.
    단일 쿼리 검색의 어휘 불일치(Lexical Gap)와 문맥 누락을 방지합니다.
    반환: 5개의 확장 쿼리 리스트 [{id, perspective, query}, ...]
    """
    q_clean = original_query.strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                "당신은 RAG 검색 최적화 전문가입니다. 사용자의 질문을 바탕으로 문서 검색 재현율(Recall)을 극대화하기 위해 "
                "서로 다른 5가지 관점(1.개념/정의, 2.핵심원리/원인, 3.구체적 수치/조건, 4.적용사례/절차, 5.핵심 키워드 조합)의 "
                "파생 검색 쿼리 5개를 JSON 배열 형식으로만 출력하세요.\n"
                f"사용자 질문: {q_clean}\n"
                '출력 형식 예시: [{"perspective": "개념 및 정의", "query": "..."}, ...]'
            )
            resp = model.generate_content(prompt)
            text = resp.text.strip()
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list) and len(parsed) >= 5:
                    return [
                        {
                            "id": f"MQ-{i+1}",
                            "perspective": item.get("perspective", f"관점 {i+1}"),
                            "query": item.get("query", q_clean),
                        }
                        for i, item in enumerate(parsed[:5])
                    ]
        except Exception as e:
            logger.warning(f"LLM Multi-Query 생성 중 예외 발생, 다중 관점 템플릿 확장으로 전환합니다: {e}")

    # 핵심 키워드 추출
    keywords = " ".join(HybridRAGStore.tokenize(q_clean)[:6])

    # 5가지 서로 다른 검색 관점으로 쿼리 확장
    expanded_queries = [
        {
            "id": "MQ-1",
            "perspective": "1. 핵심 개념 및 정의 관점",
            "query": f"{q_clean} 핵심 개념 정의 및 기본 의미",
        },
        {
            "id": "MQ-2",
            "perspective": "2. 배경 원인 및 목적 관점",
            "query": f"{q_clean} 주요 배경 원인 및 목적 이유",
        },
        {
            "id": "MQ-3",
            "perspective": "3. 세부 절차 및 작동 구조 관점",
            "query": f"{q_clean} 구성 요소 작동 방식 및 세부 절차 단계",
        },
        {
            "id": "MQ-4",
            "perspective": "4. 핵심 수치·특징 및 비교 관점",
            "query": f"{q_clean} 주요 특징 장단점 및 관련 기준 수치",
        },
        {
            "id": "MQ-5",
            "perspective": "5. 핵심 키워드 밀집 매칭 관점",
            "query": f"{q_clean} {keywords}",
        },
    ]
    return expanded_queries


def perform_hybrid_search_and_rrf(
    expanded_queries: List[Dict[str, str]],
    top_k: int = 5,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    rrf_k: int = 60,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    [하이브리드 검색 계층 (Hybrid Search) + 상호 순위 융합 계층 (RAG Fusion & RRF)]
    1. 확장된 5개 쿼리 각각에 대해:
       - FAISS Dense Vector 검색 (L2 거리 기반 의미적 유사도 순위 산출)
       - BM25 Sparse Keyword 검색 (단어 빈도/역문서 빈도 기반 정확한 키워드 매칭 순위 산출)
    2. 10개 검색 경로(5개 쿼리 x 2개 검색 엔진)에서 수집된 문서 순위를 RRF 공식으로 통합:
       RRF_Score(d) = Σ_{q ∈ Queries} [ w_dense / (k + rank_dense(q, d)) + w_sparse / (k + rank_sparse(q, d)) ]
    3. 최종 RRF 점수가 높은 상위 top_k 문서를 재정렬(Re-ranking)하여 반환합니다.
    """
    if rag_store.faiss_index is None or rag_store.bm25_index is None or not rag_store.documents:
        raise RuntimeError("벡터 DB가 아직 생성되지 않았습니다. 먼저 PDF 문서를 업로드하여 벡터 DB를 생성해 주세요.")

    num_docs = len(rag_store.documents)
    candidate_k = min(num_docs, max(top_k * 3, 10))

    # 문서별 누적 RRF 점수 및 상세 순위 추적 딕셔너리
    rrf_scores: Dict[str, float] = {}
    doc_trace: Dict[str, Dict[str, Any]] = {}
    per_query_logs: List[Dict[str, Any]] = []

    for mq in expanded_queries:
        q_id = mq["id"]
        q_text = mq["query"]

        # ------------------------------------------------------------------
        # (A) FAISS Dense Vector Search (의미 기반 검색)
        # 시간 복잡도: IndexFlatL2 기준 O(M * D), IndexHNSWFlat 기준 O(log(M) * D)
        # ------------------------------------------------------------------
        q_vec = rag_store.compute_embedding(q_text).reshape(1, -1).astype(np.float32)
        distances, indices = rag_store.faiss_index.search(q_vec, candidate_k)

        dense_hits: List[str] = []
        for rank_idx, (dist, doc_idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if doc_idx < 0 or doc_idx >= num_docs:
                continue
            doc = rag_store.documents[int(doc_idx)]
            doc_id = doc["doc_id"]
            dense_hits.append(doc_id)

            # RRF Dense 기여도 합산: w_dense / (k + rank_dense)
            dense_rrf = dense_weight * (1.0 / (rrf_k + rank_idx))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + dense_rrf

            if doc_id not in doc_trace:
                doc_trace[doc_id] = {
                    "doc": doc,
                    "best_dense_rank": rank_idx,
                    "best_sparse_rank": None,
                    "min_l2_distance": float(dist),
                    "max_bm25_score": 0.0,
                    "dense_rrf_sum": 0.0,
                    "sparse_rrf_sum": 0.0,
                    "matched_queries": set(),
                }
            else:
                if doc_trace[doc_id]["best_dense_rank"] is None or rank_idx < doc_trace[doc_id]["best_dense_rank"]:
                    doc_trace[doc_id]["best_dense_rank"] = rank_idx
                if float(dist) < doc_trace[doc_id]["min_l2_distance"]:
                    doc_trace[doc_id]["min_l2_distance"] = float(dist)

            doc_trace[doc_id]["dense_rrf_sum"] += dense_rrf
            doc_trace[doc_id]["matched_queries"].add(q_id)

        # ------------------------------------------------------------------
        # (B) BM25 Sparse Keyword Search (키워드 기반 검색)
        # 시간 복잡도: O(|q| * M)
        # ------------------------------------------------------------------
        q_tokens = HybridRAGStore.tokenize(q_text)
        bm25_scores = rag_store.bm25_index.get_scores(q_tokens)
        sorted_bm25_indices = np.argsort(bm25_scores)[::-1][:candidate_k]

        sparse_hits: List[str] = []
        for rank_idx, doc_idx in enumerate(sorted_bm25_indices, start=1):
            score_val = float(bm25_scores[int(doc_idx)])
            doc = rag_store.documents[int(doc_idx)]
            doc_id = doc["doc_id"]
            sparse_hits.append(doc_id)

            # RRF Sparse 기여도 합산: w_sparse / (k + rank_sparse)
            sparse_rrf = sparse_weight * (1.0 / (rrf_k + rank_idx))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + sparse_rrf

            if doc_id not in doc_trace:
                doc_trace[doc_id] = {
                    "doc": doc,
                    "best_dense_rank": None,
                    "best_sparse_rank": rank_idx,
                    "min_l2_distance": None,
                    "max_bm25_score": score_val,
                    "dense_rrf_sum": 0.0,
                    "sparse_rrf_sum": 0.0,
                    "matched_queries": set(),
                }
            else:
                if doc_trace[doc_id]["best_sparse_rank"] is None or rank_idx < doc_trace[doc_id]["best_sparse_rank"]:
                    doc_trace[doc_id]["best_sparse_rank"] = rank_idx
                if score_val > doc_trace[doc_id]["max_bm25_score"]:
                    doc_trace[doc_id]["max_bm25_score"] = score_val

            doc_trace[doc_id]["sparse_rrf_sum"] += sparse_rrf
            doc_trace[doc_id]["matched_queries"].add(q_id)

        per_query_logs.append(
            {
                "query_id": q_id,
                "perspective": mq["perspective"],
                "query": q_text,
                "top_dense_docs": dense_hits[:3],
                "top_sparse_docs": sparse_hits[:3],
            }
        )

    # ------------------------------------------------------------------
    # (C) RRF 점수 기준 내림차순 재정렬 (Re-ranking)
    # 시간 복잡도: O(U * log(U)) (U: 후보 문서 수)
    # ------------------------------------------------------------------
    ranked_doc_ids = sorted(rrf_scores.keys(), key=lambda d_id: rrf_scores[d_id], reverse=True)[:top_k]

    reranked_results: List[Dict[str, Any]] = []
    for final_rank, doc_id in enumerate(ranked_doc_ids, start=1):
        trace = doc_trace[doc_id]
        doc_info = trace["doc"]
        reranked_results.append(
            {
                "final_rank": final_rank,
                "doc_id": doc_id,
                "source": doc_info["source"],
                "page": doc_info["page"],
                "content": doc_info["content"],
                "rrf_score": round(rrf_scores[doc_id], 6),
                "dense_rrf_contribution": round(trace["dense_rrf_sum"], 6),
                "sparse_rrf_contribution": round(trace["sparse_rrf_sum"], 6),
                "best_dense_rank": trace["best_dense_rank"],
                "best_sparse_rank": trace["best_sparse_rank"],
                "min_l2_distance": round(trace["min_l2_distance"], 4)
                if trace["min_l2_distance"] is not None
                else None,
                "max_bm25_score": round(trace["max_bm25_score"], 4),
                "matched_queries": sorted(list(trace["matched_queries"])),
            }
        )

    fusion_diagnostics = {
        "total_indexed_chunks": num_docs,
        "expanded_query_count": len(expanded_queries),
        "total_retrieval_paths": len(expanded_queries) * 2,
        "unique_candidates_evaluated": len(rrf_scores),
        "rrf_constant_k": rrf_k,
        "weights": {"dense": dense_weight, "sparse": sparse_weight},
        "per_query_retrieval_summary": per_query_logs,
    }

    return reranked_results, fusion_diagnostics


def synthesize_rag_answer(original_query: str, reranked_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    RRF로 선별된 상위 문서 스니펫만을 근거로 LLM 최종 답변과 명확한 출처(파일명, 페이지, 청크 ID)를 생성합니다.
    """
    if not reranked_docs:
        return {
            "answer": "검색된 관련 문서가 없습니다. 먼저 관련 PDF 문서를 업로드하여 벡터 DB를 생성해 주세요.",
            "citations": [],
            "generation_mode": "empty",
        }

    citations = [
        {
            "ref_id": f"[{doc['final_rank']}]",
            "doc_id": doc["doc_id"],
            "source": doc["source"],
            "page": doc["page"],
            "rrf_score": doc["rrf_score"],
        }
        for doc in reranked_docs
    ]

    context_blocks = "\n\n".join(
        [
            f"[{doc['final_rank']}] (출처: {doc['source']}, {doc['page']}페이지, RRF점수: {doc['rrf_score']})\n{doc['content']}"
            for doc in reranked_docs
        ]
    )

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                "당신은 신뢰할 수 있는 AI 어시스턴트입니다. 아래 제공된 [RAG 검색 문서]만을 근거로 사용자의 질문에 답변하세요.\n"
                "답변 작성 시 반드시 인용한 문장 끝에 [1], [2]와 같이 출처 번호를 표기하고, 마지막에 출처 요약을 덧붙이세요.\n\n"
                f"[사용자 질문]\n{original_query}\n\n"
                f"[RAG 검색 문서 (RRF 상위 문서)]\n{context_blocks}"
            )
            resp = model.generate_content(prompt)
            return {
                "answer": resp.text.strip(),
                "citations": citations,
                "generation_mode": "gemini-1.5-flash",
            }
        except Exception as e:
            logger.warning(f"LLM 답변 생성 호출 실패, RRF 근거 기반 구조화 요약 답변으로 대체합니다: {e}")

    # API 키 미설정 시 RRF 상위 문맥을 기반으로 출처가 명시된 구조화 답변 생성
    summary_lines = [
        f"질문하신 **\"{original_query}\"**에 대해 Multi-Query(5가지 관점) 및 Hybrid Search(FAISS + BM25) 후 "
        f"**RRF(상호 순위 융합)**로 재정렬된 상위 {len(reranked_docs)}개 문서를 분석한 답변입니다.\n"
    ]
    for doc in reranked_docs[:3]:
        rank = doc["final_rank"]
        snippet = doc["content"][:220] + ("..." if len(doc["content"]) > 220 else "")
        summary_lines.append(
            f"- **핵심 근거 [{rank}]** *(RRF: {doc['rrf_score']}, Dense 순위: {doc['best_dense_rank']}, BM25 순위: {doc['best_sparse_rank']})*: "
            f"{snippet} **[{rank}: {doc['source']} p.{doc['page']}]**"
        )

    summary_lines.append("\n---\n**참고 출처 (Citations):**")
    for doc in reranked_docs:
        summary_lines.append(
            f"- **[{doc['final_rank']}]** `{doc['source']}` (페이지: {doc['page']}p, 청크: `{doc['doc_id']}`, RRF 점수: `{doc['rrf_score']}`)"
        )

    return {
        "answer": "\n".join(summary_lines),
        "citations": citations,
        "generation_mode": "extractive-rag-fusion",
    }


# ==============================================================================
# 5. API 엔드포인트 정의 (/health, /ingest, /query, /query/stream)
# ==============================================================================
@app.get("/health")
async def health_check() -> JSONResponse:
    """
    서버 동작 상태, 현재 적재된 FAISS/BM25 인덱스 현황, 그리고 알고리즘 복잡도 정보를 반환합니다.
    """
    return JSONResponse(
        content={
            "status": "healthy",
            "server_message": "FastAPI Advanced RAG 서버가 정상 작동 중입니다.",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "index_status": {
                "is_ready": rag_store.faiss_index is not None and len(rag_store.documents) > 0,
                "source_filename": rag_store.source_filename,
                "total_chunks": len(rag_store.documents),
                "faiss_index_type": rag_store.index_type,
                "embedding_dim": rag_store.embedding_dim,
                "last_updated": rag_store.last_updated,
            },
            "algorithm_complexity": ALGORITHM_COMPLEXITY,
        }
    )


@app.post("/ingest")
async def ingest_document(req: IngestRequest) -> JSONResponse:
    """
    PDF 파일(Base64) 또는 원문 텍스트를 받아 청킹 후 FAISS Dense Index 및 BM25 Sparse Index를 구축합니다.
    단계별 진행 로그와 알고리즘 복잡도를 함께 반환합니다.
    """
    progress_logs: List[Dict[str, Any]] = []
    try:
        # 1단계: 입력 데이터 검증 및 텍스트 추출
        progress_logs.append(
            {
                "step": 1,
                "stage": "PDF Parsing & Chunking",
                "status": "processing",
                "message": f"'{req.filename}' 파일에서 텍스트 추출 및 청킹(size={req.chunk_size}, overlap={req.chunk_overlap}) 시작",
                "complexity": ALGORITHM_COMPLEXITY["document_chunking"],
            }
        )

        if req.pdf_base64:
            try:
                # Data URL prefix (data:application/pdf;base64,...) 제거 처리
                b64_clean = req.pdf_base64.split(",")[-1].strip()
                pdf_bytes = base64.b64decode(b64_clean)
            except Exception as decode_err:
                raise HTTPException(
                    status_code=400,
                    detail=f"[Base64 디코딩 오류] 유효하지 않은 PDF Base64 데이터입니다: {str(decode_err)}",
                )
            chunks = extract_chunks_from_pdf(
                pdf_bytes=pdf_bytes,
                filename=req.filename,
                chunk_size=req.chunk_size,
                chunk_overlap=req.chunk_overlap,
            )
        elif req.raw_text:
            chunks = extract_chunks_from_text(
                raw_text=req.raw_text,
                filename=req.filename,
                chunk_size=req.chunk_size,
                chunk_overlap=req.chunk_overlap,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="[입력 데이터 누락] pdf_base64 또는 raw_text 중 하나는 반드시 포함되어야 합니다.",
            )

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail=f"[텍스트 추출 실패] '{req.filename}'에서 텍스트를 추출할 수 없습니다. 스캔된 이미지 PDF인지 확인해 주세요.",
            )

        progress_logs.append(
            {
                "step": 2,
                "stage": "Chunking Completed",
                "status": "completed",
                "message": f"총 {len(chunks)}개의 문서 청크 생성 완료",
                "progress": 45,
            }
        )

        # 2단계: FAISS + BM25 하이브리드 인덱스 구축
        index_stats = rag_store.build_indices(chunks=chunks, index_type=req.index_type)
        rag_store.source_filename = req.filename

        faiss_comp_key = "faiss_hnsw_flat" if rag_store.index_type == "IndexHNSWFlat" else "faiss_flat_l2"
        progress_logs.append(
            {
                "step": 3,
                "stage": "FAISS & BM25 Indexing Completed",
                "status": "completed",
                "message": (
                    f"FAISS({rag_store.index_type}) 및 BM25 인덱스 생성 완료 "
                    f"(청크 수: {index_stats['total_chunks']}, 차원: {index_stats['embedding_dimension']}, "
                    f"소요 시간: {index_stats['elapsed_ms']}ms)"
                ),
                "progress": 100,
                "complexity": {
                    "faiss": ALGORITHM_COMPLEXITY[faiss_comp_key],
                    "bm25": ALGORITHM_COMPLEXITY["bm25_sparse"],
                },
            }
        )

        return JSONResponse(
            content={
                "status": "success",
                "message": f"'{req.filename}' 기반 하이브리드 벡터 DB(FAISS + BM25) 구축이 완료되었습니다.",
                "filename": req.filename,
                "index_stats": index_stats,
                "progress_logs": progress_logs,
                "algorithm_complexity": {
                    "chunking": ALGORITHM_COMPLEXITY["document_chunking"],
                    "faiss": ALGORITHM_COMPLEXITY[faiss_comp_key],
                    "bm25": ALGORITHM_COMPLEXITY["bm25_sparse"],
                },
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"인덱스 구축 중 서버 오류: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"[벡터 DB 생성 오류] {type(e).__name__}: {str(e)}",
        )


@app.post("/query")
async def query_rag_pipeline(req: QueryRequest) -> JSONResponse:
    """
    [단일 JSON 및 스트리밍 호환 엔드포인트]
    1. Multi-Query 확장 (5가지 관점)
    2. Hybrid Search (FAISS Dense + BM25 Sparse)
    3. RAG Fusion (RRF 재정렬)
    4. LLM 답변 생성 및 출처 표기
    """
    start_time = time.perf_counter()
    stream_events: List[Dict[str, Any]] = []

    try:
        if rag_store.faiss_index is None or not rag_store.documents:
            raise HTTPException(
                status_code=400,
                detail="[벡터 DB 미생성] 현재 적재된 문서 인덱스가 없습니다. 먼저 PDF 파일을 선택하고 '벡터 DB 생성' 버튼을 눌러주세요.",
            )

        # Step 1: Multi-Query Layer (5개 관점 쿼리 확장)
        expanded_queries = generate_multi_queries(req.query)
        stream_events.append(
            json.loads(
                StreamHandler.format_event(
                    stage="Multi-Query Layer",
                    status="completed",
                    message=f"단일 질의를 5가지 관점의 파생 쿼리로 확장 완료",
                    progress=30,
                    data={"expanded_queries": expanded_queries},
                    complexity=ALGORITHM_COMPLEXITY["multi_query_expansion"],
                )
            )
        )

        # Step 2 & 3: Hybrid Search (FAISS + BM25) & RRF Fusion Layer
        reranked_docs, diagnostics = perform_hybrid_search_and_rrf(
            expanded_queries=expanded_queries,
            top_k=req.top_k,
            dense_weight=req.dense_weight,
            sparse_weight=req.sparse_weight,
            rrf_k=req.rrf_k,
        )
        faiss_comp_key = "faiss_hnsw_flat" if rag_store.index_type == "IndexHNSWFlat" else "faiss_flat_l2"
        stream_events.append(
            json.loads(
                StreamHandler.format_event(
                    stage="Hybrid Search & RRF Fusion Layer",
                    status="completed",
                    message=(
                        f"10개 검색 경로(5개 쿼리 x [FAISS + BM25])에서 후보 문서를 수집하고 "
                        f"RRF(k={req.rrf_k})로 상위 {len(reranked_docs)}개 문서 재정렬 완료"
                    ),
                    progress=70,
                    data={"diagnostics": diagnostics},
                    complexity=ALGORITHM_COMPLEXITY["rrf_fusion"],
                )
            )
        )

        # Step 4: LLM Answer Generation with Citations
        answer_bundle = synthesize_rag_answer(req.query, reranked_docs)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        stream_events.append(
            json.loads(
                StreamHandler.format_event(
                    stage="Answer Generation Layer",
                    status="completed",
                    message=f"RRF 상위 문서 기반 최종 답변 및 출처 생성 완료 ({elapsed_ms}ms)",
                    progress=100,
                )
            )
        )

        return JSONResponse(
            content={
                "status": "success",
                "original_query": req.query,
                "expanded_queries": expanded_queries,
                "reranked_documents": reranked_docs,
                "llm_answer": answer_bundle["answer"],
                "citations": answer_bundle["citations"],
                "generation_mode": answer_bundle["generation_mode"],
                "diagnostics": diagnostics,
                "elapsed_ms": elapsed_ms,
                "stream_events": stream_events,
                "algorithm_complexity": {
                    "multi_query": ALGORITHM_COMPLEXITY["multi_query_expansion"],
                    "faiss_dense": ALGORITHM_COMPLEXITY[faiss_comp_key],
                    "bm25_sparse": ALGORITHM_COMPLEXITY["bm25_sparse"],
                    "rrf_fusion": ALGORITHM_COMPLEXITY["rrf_fusion"],
                },
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"질의 처리 중 서버 오류: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"[질의 파이프라인 오류] {type(e).__name__}: {str(e)}",
        )


@app.post("/query/stream")
async def query_rag_stream(req: QueryRequest) -> StreamingResponse:
    """
    [StreamHandler 전용 실시간 NDJSON 스트리밍 엔드포인트]
    각 파이프라인 계층이 완료될 때마다 즉시 스트림 청크를 클라이언트로 푸시합니다.
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        start_time = time.perf_counter()
        try:
            if rag_store.faiss_index is None or not rag_store.documents:
                yield StreamHandler.format_event(
                    stage="Validation",
                    status="error",
                    message="[벡터 DB 미생성] 먼저 PDF 파일을 업로드하여 벡터 DB를 생성해 주세요.",
                    progress=0,
                )
                return

            yield StreamHandler.format_event(
                stage="Multi-Query Layer",
                status="processing",
                message="사용자 질의를 5가지 관점의 파생 쿼리로 확장 중...",
                progress=15,
                complexity=ALGORITHM_COMPLEXITY["multi_query_expansion"],
            )

            expanded_queries = generate_multi_queries(req.query)
            yield StreamHandler.format_event(
                stage="Multi-Query Layer",
                status="completed",
                message="5가지 관점의 파생 쿼리 생성 완료",
                progress=35,
                data={"expanded_queries": expanded_queries},
                complexity=ALGORITHM_COMPLEXITY["multi_query_expansion"],
            )

            reranked_docs, diagnostics = perform_hybrid_search_and_rrf(
                expanded_queries=expanded_queries,
                top_k=req.top_k,
                dense_weight=req.dense_weight,
                sparse_weight=req.sparse_weight,
                rrf_k=req.rrf_k,
            )
            yield StreamHandler.format_event(
                stage="Hybrid Search & RRF Layer",
                status="completed",
                message=f"FAISS + BM25 하이브리드 검색 및 RRF 상위 {len(reranked_docs)}건 재정렬 완료",
                progress=75,
                data={"reranked_documents": reranked_docs, "diagnostics": diagnostics},
                complexity=ALGORITHM_COMPLEXITY["rrf_fusion"],
            )

            answer_bundle = synthesize_rag_answer(req.query, reranked_docs)
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

            faiss_comp_key = "faiss_hnsw_flat" if rag_store.index_type == "IndexHNSWFlat" else "faiss_flat_l2"
            yield StreamHandler.format_event(
                stage="Final Response",
                status="completed",
                message=f"답변 생성 완료 ({elapsed_ms}ms)",
                progress=100,
                data={
                    "status": "success",
                    "original_query": req.query,
                    "expanded_queries": expanded_queries,
                    "reranked_documents": reranked_docs,
                    "llm_answer": answer_bundle["answer"],
                    "citations": answer_bundle["citations"],
                    "generation_mode": answer_bundle["generation_mode"],
                    "diagnostics": diagnostics,
                    "elapsed_ms": elapsed_ms,
                    "algorithm_complexity": {
                        "multi_query": ALGORITHM_COMPLEXITY["multi_query_expansion"],
                        "faiss_dense": ALGORITHM_COMPLEXITY[faiss_comp_key],
                        "bm25_sparse": ALGORITHM_COMPLEXITY["bm25_sparse"],
                        "rrf_fusion": ALGORITHM_COMPLEXITY["rrf_fusion"],
                    },
                },
            )
        except Exception as e:
            yield StreamHandler.format_event(
                stage="Error",
                status="error",
                message=f"[스트리밍 파이프라인 오류] {type(e).__name__}: {str(e)}",
                progress=0,
            )

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

