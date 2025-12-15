from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Tuple, Dict, List, Annotated, Union, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import pandas as pd


def merge_list(left: Optional[List[Any]], right: Optional[List[Any]]) -> List[Any]:
    if left == right:
        return left or []
    return (left or []) + (right or [])


class MappedKeyword(BaseModel):
    raw_keyword: str
    mapped_keyword: str
    confidence: float


class UnmappedKeyword(BaseModel):
    raw_keyword: str
    candidate_entities: List[str]
    confidences: List[float]


class CandidateChunk(BaseModel):
    element_id: str
    chunk_type: str  # "keyword", "fact", "section"
    collection_name: str
    content: str
    confidence: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MergedCandidate(BaseModel):
    """Vector/Graph 병합 결과의 표준 스키마"""

    id: str
    labels: List[str] = Field(default_factory=list)
    type: Optional[str] = None
    content: str = ""
    weight: float = 0.0
    confidence: float = 0.0
    connections: int = 0
    sources: List[str] = Field(default_factory=list)
    section_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # 관계는 현재 dict 기반으로 유지 (기존 유틸과의 호환성)
    relation: List[Dict[str, Any]] = Field(default_factory=list)
    # 이미지 관련(옵션)
    image_base64: Optional[str] = None
    images: Optional[Dict[str, Any]] = None

class Candidate(BaseModel):
    "graph 결과"
    # dict 형태의 결과
    result : Optional[Dict[str, Any]] # Neo4j 쿼리 return 값
    # result를 파싱해서 content에 저장 (이미지 데이터 제외)
    content  : str
    source : List[Dict[str, Any]] = Field(default_factory=list) # result 정보 중 _id로 나타내어지는 source 정보 모음
    # 이미지 처리
    image_base64 : List[str] = Field(default_factory=list) # 추출된 이미지 base64 리스트
    is_image : bool = Field(default=False) # 이미지 포함 여부
    # 이 결과를 얻는데 사용한 Cypher 쿼리
    cypher_query : Optional[str] = None
    
class CypherQuery(BaseModel):
    cypher_query: str
    schema_info: str
    success: Optional[bool] = None
    error: Optional[str] = None
    reuse: Optional[bool] = None
    params: Optional[Dict[str, Any]] = None  # Cypher 쿼리 파라미터
class CollectionWeights(BaseModel):
    keyword_weight: float = 1.0
    fact_weight: float = 0.8
    section_weight: float = 0.6

    def normalize_weights(self) -> "CollectionWeights":
        """가중치 정규화"""
        total = self.keyword_weight + self.fact_weight + self.section_weight
        if total == 0:
            return CollectionWeights()
        return CollectionWeights(
            keyword_weight=self.keyword_weight / total,
            fact_weight=self.fact_weight / total,
            section_weight=self.section_weight / total,
        )


class UnifiedRAGState(BaseModel):
    # === 입력 데이터 ===
    model_config = ConfigDict(arbitrary_types_allowed=True)

    original_query: str
    revised_query: Optional[str] = None
    retrieved_contexts: Optional[List[str]] = None
    raw_keywords: Optional[List[str]] = None
    selected_document_ids: Optional[List[str]] = None
    selected_storage_paths: Optional[List[str]] = None
    auto_selected_folders: Optional[List[str]] = None
    select_all_requested: bool = False
    auto_select_requested: bool = False
    auto_select_threshold: int = 100
    selection_reason: Optional[str] = None
    hidden_prompt_ids: Optional[List[str]] = None
    hidden_prompts: Optional[List[str]] = None
    combined_hidden_prompt: Optional[str] = None

    conversation_classification: Optional[Dict[str, Any]] = None

    # Intent 라우팅 관련 필드
    intent_query_type: Optional[str] = None  # "DOCUMENT_OVERVIEW", "COMPARISON", "DETAIL_SEARCH"
    needs_clarification: bool = Field(default=False)  # clarification 응답 필요 여부
    clarification_message: Optional[str] = None  # clarification 메시지
    suggested_document: Optional[Dict[str, Any]] = None  # 제안된 문서 정보
    conversation_skipped: bool = Field(default=False)  # 대화 스킵 여부 (CASUAL 또는 clarification)

    keyword_mapping_results: Optional[Dict[str, List[str]]] = None
    canonical_mapping_results: Optional[Dict[str, List[str]]] = None

    # === Vector Search 결과 (컬렉션별) ===
    keyword_candidates: List[CandidateChunk] = Field(default_factory=list)
    fact_candidates: List[CandidateChunk] = Field(default_factory=list)
    section_candidates: List[CandidateChunk] = Field(default_factory=list)
    candidate_chunks: List[CandidateChunk] = Field(default_factory=list)

    # === Graph Search 결과 ===
    cypher_group: Optional[List[str]] = None
    cypher_fewshot: Optional[List[str]] = None
    cypher_query: Annotated[
        Optional[List[CypherQuery]], lambda x, y: (x or []) + (y or [])
    ] = None
    records: Annotated[
        Optional[List[Dict[str, Any]]], lambda x, y: (x or []) + (y or [])
    ] = None
    expanded_candidates: List[CandidateChunk] = Field(default_factory=list)

    # === Merge Search 결과 ===
    merged_candidates: List[MergedCandidate] = Field(default_factory=list)

    # === 최종 처리 결과 ===
    ranked_candidates: List[CandidateChunk] = Field(default_factory=list)
    formatted_context: Optional[str] = None
    
    candidate_group : List[Candidate] = Field(default_factory=list) # 검색된 정보
    # 저장용 컨텍스트(이미지 data URI를 [Image]로 치환한 마크다운)
    save_context: Optional[str] = None
    final_answer: Optional[str] = None
    # synthesizer outputs for traceability
    used_evidence_ids: List[str] = Field(default_factory=list)
    used_candidate_ids: List[str] = Field(default_factory=list)  # unused
    element_to_ref_map: Optional[Dict[str, Any]] = None # ref:1에 출처가 여러개인 경우에는 list형태로 저장됨, list[dict[str, str]]
    ref_to_element_map: Optional[Dict[str, str]] = None
    # mapping for batch evidences to child candidate IDs
    batch_children_map: Dict[str, List[str]] = Field(default_factory=dict)  # unused

    # === 메타데이터 ===
    search_metadata: Dict[str, Any] = Field(default_factory=dict)
    tool_logs: List[str] = Field(default_factory=list)

    # === 데이터 관련 ===
    data: Optional[Dict[str, Union[pd.DataFrame, Dict[str, Any]]]] = Field(default=None)
    
    # === 시각화 결과 (GeneralAnswerState와 호환) ===
    visualization_results: Annotated[
        List["VisualizationResult"], lambda x, y: (x or []) + (y or [])
    ] = Field(default_factory=list)

    # === 임시 키값 ===
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)

    def get_all_vector_candidates(self) -> List[CandidateChunk]:
        """모든 벡터 검색 결과 통합"""
        return self.keyword_candidates + self.fact_candidates + self.section_candidates

    def get_weighted_candidates(
        self, weights: CollectionWeights
    ) -> List[CandidateChunk]:
        """가중치를 적용한 후보 통합"""
        weighted_candidates = []

        # 각 collection별로 가중치 적용
        for candidate in self.keyword_candidates:
            weighted_candidate = candidate.copy()
            weighted_candidate.confidence *= weights.keyword_weight
            weighted_candidate.metadata["original_confidence"] = candidate.confidence
            weighted_candidate.metadata["collection_weight"] = weights.keyword_weight
            weighted_candidates.append(weighted_candidate)

        for candidate in self.fact_candidates:
            weighted_candidate = candidate.copy()
            weighted_candidate.confidence *= weights.fact_weight
            weighted_candidate.metadata["original_confidence"] = candidate.confidence
            weighted_candidate.metadata["collection_weight"] = weights.fact_weight
            weighted_candidates.append(weighted_candidate)

        for candidate in self.section_candidates:
            weighted_candidate = candidate.copy()
            weighted_candidate.confidence *= weights.section_weight
            weighted_candidate.metadata["original_confidence"] = candidate.confidence
            weighted_candidate.metadata["collection_weight"] = weights.section_weight
            weighted_candidates.append(weighted_candidate)

        # confidence 기준으로 정렬
        return sorted(weighted_candidates, key=lambda x: x.confidence, reverse=True)

    def get_candidates_by_type(self, chunk_type: str) -> List[CandidateChunk]:
        """타입별 후보 추출"""
        return [
            c for c in self.get_all_vector_candidates() if c.chunk_type == chunk_type
        ]

    # === 재생성 관련 필드 ===
    cypher_regeneration_count: int = Field(default=0, description="Cypher 쿼리 재생성 횟수")
    max_regeneration_attempts: int = Field(default=1, description="최대 재생성 시도 횟수")

    def get_element_ids(self) -> List[str]:
        """모든 후보의 element_id 리스트 반환"""
        all_candidates = (
            self.get_all_vector_candidates()
            + self.expanded_candidates
            + self.ranked_candidates
        )
        return list(set([c.element_id for c in all_candidates]))


class Evidence(BaseModel):
    """분석을 통해 추출한 핵심 증거 정보"""

    source: str = Field(description="증거 출처 (문서명, 청크ID 등)")
    content: str = Field(description="핵심 증거 내용 (요약 또는 핵심 포인트)")
    relevance_score: float = Field(description="질문과의 관련성 점수 (0.0-1.0)")
    source_id: List[Dict[str, str]] = Field(default_factory=list, description="원본 출처 ID 목록 (청크/섹션/기타)")
    evidence_type: str = Field(
        default="rag", description="증거 타입 (rag, web, analysis 등)"
    )
    toc_title: Optional[str] = Field(
        default=None, description="(옵션) 연결된 TOC/섹션 제목"
    )
    toc_summary: Optional[str] = Field(
        default=None, description="(옵션) 연결된 TOC 요약/설명"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="원본 후보 메타데이터 (traceability)"
    )


class VisualizationResult(BaseModel):
    """시각화 결과"""

    image_path: str = Field(description="이미지 경로")
    instruction: str = Field(description="사용자 요청")
    chart_created: bool = Field(description="차트 생성 여부")
    base64_image: str = Field(description="이미지 Base64 인코딩")
    timestamp: str = Field(description="생성 시간")
    chart_type: str = Field(description="차트 타입")


class GeneralAnswerState(BaseModel):
    """범용 답변 생성을 위한 확장된 State"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 사용자 질문 저장
    # RAG State와 공유되는 필드
    original_query: str
    revised_query: Optional[str] = None
    formatted_context: Optional[str] = None
    # 저장용 컨텍스트(이미지 data URI를 [Image]로 치환한 마크다운)
    save_context: Optional[str] = None

    # 기본 메시지 및 워크플로우 관리
    messages: Annotated[List[BaseMessage], add_messages]
    next_node: Optional[str] = None
    current_instruction: Optional[str] = None

    # 스텝 제한 (무한 반복 방지)
    current_step: int = Field(default=0)
    max_steps: int = Field(default=20)

    # 정보 수집 시도 관리
    rag_analysis_attempts: int = Field(default=0)
    max_search_attempts: int = Field(default=2, description="각 타입별 최대 시도 횟수")

    # 데이터 관련 (통계 분석용)
    data: Optional[Dict[str, Union[pd.DataFrame, Dict[str, Any]]]] = None
    images: Annotated[List[str], merge_list] = Field(default_factory=list)
    analysis_results: Annotated[List[Dict[str, Any]], merge_list] = Field(
        default_factory=list
    )
    visualization_results: Annotated[List[VisualizationResult], merge_list] = Field(
        default_factory=list
    )

    # RAG 관련 (UnifiedRAGState와 호환)
    ranked_candidates: Annotated[List[CandidateChunk], merge_list] = Field(
        default_factory=list, description="최종 처리된 RAG 후보들 (UnifiedRAGState와 공유)"
    )
    # 새로운 검색 결과 구조
    candidate_group: List[Candidate] = Field(default_factory=list, description="검색된 정보 (새로운 구조)")
    rag_evidences: Annotated[List[Evidence], merge_list] = Field(
        default_factory=list, description="RAG에서 추출한 핵심 증거들"
    )
    rag_analysis_done: bool = Field(default=False)

    forced_synthesis: bool = Field(default=False, description="정보 부족하지만 강제 답변 모드")

    # 기타 검색 결과
    search_results: Annotated[List[Dict[str, Any]], merge_list] = Field(
        default_factory=list, description="모든 검색 결과들"
    )

    # 작업 히스토리 및 로그
    operation_history: Annotated[List[str], merge_list] = Field(default_factory=list)
    tool_logs: Annotated[List[str], merge_list] = Field(default_factory=list)

    # 최종 답변 관련
    final_answer: Optional[str] = None
    # synthesizer outputs for traceability
    used_evidence_ids: Annotated[List[str], merge_list] = Field(default_factory=list)
    element_to_ref_map: Optional[Dict[str, Any]] = None # ref:1에 출처가 여러개인 경우에는 list형태로 저장됨, list[dict[str, str]]
    ref_to_element_map: Optional[Dict[str, str]] = None


class DataAnalysisReActState(BaseModel):
    """ReAct 데이터 분석 전용 상태 - GeneralAnswerState와 독립"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 🔗 호환성 키들 (기존 시스템과 연동용)
    messages: Annotated[List[BaseMessage], add_messages] = Field(
        default_factory=list
    )  # 슈퍼바이저와 공유되는 메세지 (final_answer만 저장)

    data: Optional[Dict[str, pd.DataFrame]] = Field(default=None)
    analysis_results: Annotated[
        List[Dict[str, Any]], lambda x, y: (x or []) + (y or [])
    ] = Field(default_factory=list)
    # 사용자 질문 저장
    revised_query: Optional[str] = None

    # ----------------------------------------------------------------

    # 🤖 ReAct 내부 메시지 관리 (슈퍼바이저와 분리)
    react_messages: Annotated[List[BaseMessage], add_messages] = Field(
        default_factory=list, description="ReAct 과정의 내부 메시지들"
    )

    # 🎯 ReAct 전용 필드들
    current_instruction: str = Field(description="현재 수행 중인 작업 지시")

    # ReAct 사이클 관리
    current_iteration: int = Field(default=0)
    max_iterations: int = Field(default=20)

    # 현재 진행 상태
    current_thought: Optional[str] = None
    current_action: Optional[str] = None
    current_action_input: Optional[Any] = None
    current_observation: Optional[str] = None

    # 도구 관련
    tool_execution_history: Annotated[
        List[Dict[str, Any]], lambda x, y: (x or []) + (y or [])
    ] = Field(default_factory=list)

    # 완료 상태
    task_completed: bool = Field(default=False)
    final_answer: Optional[str] = None
    completion_reason: Optional[str] = None  # "success", "max_iterations", "error"

    # 오류 처리
    last_error: Optional[str] = None
    error_count: int = Field(default=0)
    max_errors: int = Field(default=10)

    # 🆕 텍스트 분석 관련 필드들
    text_candidate_chunks: Annotated[
        List[CandidateChunk], lambda x, y: (x or []) + (y or [])
    ] = Field(default_factory=list, description="데이터프레임에서 추출한 텍스트 CandidateChunk들")
    rag_evidences: Annotated[
        List[Evidence], lambda x, y: (x or []) + (y or [])
    ] = Field(
        default_factory=list,
        description="RAG analyzer로부터 받은 텍스트 분석 결과 (GeneralAnswerState와 키 공유)",
    )
    requires_text_analysis: bool = Field(default=False, description="텍스트 분석이 필요한지 여부")
    text_analysis_completed: bool = Field(default=False, description="텍스트 분석 완료 여부")


class VisualizationReActState(BaseModel):
    """시각화 ReAct 전용 상태 - GeneralAnswerState와 독립적인 내부 상태 관리"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 🔗 Supervisor 공유 영역 (GeneralAnswerState와 호환)
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    data: Optional[Dict[str, pd.DataFrame]] = Field(default=None)
    visualization_results: Annotated[
        List[VisualizationResult], lambda x, y: (x or []) + (y or [])
    ] = Field(default_factory=list)
    images: Annotated[List[str], lambda x, y: (x or []) + (y or [])] = Field(
        default_factory=list
    )
    # 🤖 ReAct 내부 메시지 관리 (슈퍼바이저와 분리)
    react_messages: Annotated[List[BaseMessage], add_messages] = Field(
        default_factory=list, description="ReAct 과정의 내부 메시지들"
    )

    # 🎯 ReAct 전용 필드들
    current_instruction: str = Field(description="현재 수행 중인 시각화 지시")
    revised_query: Optional[str] = Field(default=None, description="수정된 쿼리")

    # ReAct 사이클 관리
    current_iteration: int = Field(default=0)
    max_iterations: int = Field(default=20)

    # 현재 진행 상태
    current_thought: Optional[str] = None
    current_action: Optional[str] = None
    current_action_input: Optional[Any] = None
    current_observation: Optional[str] = None

    # 시각화 전용 필드들
    generated_image_paths: List[str] = Field(default_factory=list)
    successful_charts: int = Field(default=0)
    failed_charts: int = Field(default=0)

    # 도구 관련
    tool_execution_history: Annotated[
        List[Dict[str, Any]], lambda x, y: (x or []) + (y or [])
    ] = Field(default_factory=list)

    # 완료 상태
    task_completed: bool = Field(default=False)
    final_answer: Optional[str] = None
    completion_reason: Optional[str] = None  # "success", "max_iterations", "error"

    # 오류 처리
    last_error: Optional[str] = None
    error_count: int = Field(default=0)
    max_errors: int = Field(default=10)
