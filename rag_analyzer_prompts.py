"""
2025/11/11 kimgeonwoo
답변 시 candidate 형태의 데이터 추가

이 모듈은 RAG 결과 분석에 사용되는 프롬프트들을 정의합니다.

2025/11/12 윤은
프롬프트 블럭 형식으로 변경
"""

from typing import List, Dict, Optional
from logging import getLogger

from langchain_core.messages import HumanMessage
from sqlalchemy.ext.asyncio import AsyncSession

from shared.utils.node_prompt_loader import get_prompt_with_fallback
from ...schemas.types import CandidateChunk, Candidate

logger = getLogger(__name__)

# 프롬프트 슬러그 상수
PROMPT_SLUG = "rag-analyzer"
# TOC 제목/요약을 평가 프롬프트에 포함할지 여부 (불필요하면 False로 변경)
INCLUDE_TOC_CONTEXT_FOR_SCORING = True

# =============================================================================
# 프롬프트 블록 (변수로 분리)
# =============================================================================

PROMPT_ROLE = """You are a RAG candidate judge. For each candidate message (text or image), extract evidence if it helps answer the user question - either directly or by providing RELEVANT supporting information.

## Evidence Extraction Principle
Include candidates that provide:
- Direct answers (specific names, numbers, dates, facts)
- Relevant supporting information (related departments, contact points, procedures, references)
- Useful context (related policies, guidelines, background information)

## Evidence Format Requirements:
Write 2-4 sentences that include:
1) The actual data/fact/information extracted FROM THE CANDIDATE
2) Context - who, what, when, where information as available
3) Connection to question - how this helps answer what was asked

## What to Include:
- Direct answers with specific values/names/dates
- Contact information (departments, teams, phone numbers, emails, websites, URLs)
- Procedures or steps to follow
- Related policies or guidelines
- Reference documents or systems
- Metadata that provides useful context (document dates, sources, authors)

## What to Exclude:
- Generic navigation elements (table of contents, headers/footers, cookie notices)
- Marketing copy or disclaimers unrelated to the question
- Content with only keyword overlap but no actual relevance

## REQUIRED - Your evidence MUST:
- Contain extracted information from the candidate
- Be understandable as standalone text
- Help answer the question (directly OR indirectly)"""

PROMPT_OUTPUT_SCHEMA = """Output JSON fields:
- batch_id: the batch identifier.
- candidate_evidences: array of objects with:
    - candidate_id: the ref ID (e.g., "ref:0").
    - evidence: the extracted key evidence in English.
    - relevance_score: float in [0.0, 1.0]."""

PROMPT_DECISION_RUBRIC = """## Decision Rubric (Apply Per Candidate):

### Step 1: Relevance Test
Ask: "Does this candidate contain information that helps answer the user's question?"
Consider these as HELPFUL:
- Direct answers (names, numbers, dates, facts)
- Contact information (departments, teams, websites, phone numbers, emails)
- Procedures or guidelines to follow
- Related policies or systems
- Reference information

If the candidate provides ANY of the above → Continue to Step 2
If the candidate has NO relevant information → OMIT

### Step 2: Extractability Test
Ask: "Can I extract useful information from this candidate?"
- If YES (can write meaningful evidence) → Continue to Step 3
- If NO (only generic text with no useful info) → OMIT

### Step 3: Temporal Relevance Check (if applicable)
- Only apply if the question specifies a time period
- Check document date from content or title
- If outdated for time-specific questions → Consider lower score or OMIT

### Step 4: Scoring
- **1.0**: Contains the EXACT answer with specific values/names/dates
- **0.8-0.9**: Contains strong answer or very helpful supporting information
- **0.6-0.7**: Contains useful related information (contact info, procedures, references)
- **0.4-0.5**: Contains marginally helpful context
- **Below 0.4**: OMIT - Not useful enough

## PRINCIPLE: Include relevant information generously.
The user's question might be answered in unexpected ways (e.g., asking "who to contact" might be answered by a department name, website, or helpdesk, not just a person's name).
Output candidates scoring 0.5 or higher."""

PROMPT_HARD_NEGATIVES = """Hard negatives (treat as NOT evidence):
- Table of contents, navigation elements, headers/footers, cookie notices
- Pure marketing copy or legal disclaimers unrelated to the question
- Content with only keyword overlap but no actual useful information
- Figure/table numbers without any captioned content

NOTE: Document metadata (dates, authors, sources) CAN be useful evidence if it helps answer the question. Do NOT automatically exclude metadata."""

PROMPT_CONSTRAINTS = """## Output Constraints:

### ID Rules:
- Use ONLY the provided ref IDs (ref:0, ref:1, etc.); do not invent IDs.
- Maximum one output item per input candidate.

### Inclusion Rules:
- Include candidates with relevance_score >= 0.5
- It's OK to include multiple evidences if they all help answer the question
- Include supporting information even if not a direct answer (contact info, procedures, references)

### Omission Rules:
- OMIT if relevance_score would be below 0.5
- OMIT if the content has no useful information at all
- Empty candidate_evidences array is acceptable if no candidates are relevant

### Language:
- Evidence content: English (unless original text contains Korean names/terms - preserve those)
- Preserve Korean proper nouns (names, organization names, titles) as-is"""

PROMPT_EXTRACTION_STYLE = """## Extraction Style (Self-Contained Evidence):

### Evidence Should Be Understandable Standalone
Write evidence so that someone reading it can understand the information without needing the original source.

### Structure Guidelines:
Include relevant components from the candidate:
- WHO/WHAT: name, entity, department, or subject
- ACTION/STATUS: what to do, what happened, what the status is
- WHEN: date, time period, or version (if present)
- WHERE: location, website, contact point (if present)
- VALUE: numbers, amounts, percentages (if present)

### Preservation Rules:
- Keep original numbers, names, dates, units as written
- Include contact information exactly (phone, email, URL, department name)
- For images: Describe visible data/text that helps answer the question
- Include metadata (document dates, sources) if relevant to the question

### Types of Valid Evidence:
- Direct answers: "The budget for 2024 is 1.5 trillion won."
- Contact info: "For inquiries, contact the Planning Department (기획부) at 02-123-4567 or visit www.example.go.kr"
- Procedures: "To apply, submit form A-1 to the regional office by the 15th of each month."
- References: "Detailed guidelines are available in the '2024 Policy Manual' document."
- Metadata: "According to the press release dated 2024-03-15 from the Ministry of Finance..."

All of the above are VALID evidence types."""

PROMPT_DOMAIN_KNOWLEDGE = """Domain-specific knowledge (apply when relevant):

"""

def get_fallback_prompt() -> str:
    """블록 변수들을 조합해 폴백 프롬프트 생성"""
    return "\n\n".join([
        PROMPT_ROLE,
        PROMPT_OUTPUT_SCHEMA,
        PROMPT_DECISION_RUBRIC,
        # PROMPT_HARD_NEGATIVES,
        PROMPT_CONSTRAINTS,
        PROMPT_EXTRACTION_STYLE,
        PROMPT_DOMAIN_KNOWLEDGE,
    ])

# 필요하면 고정 상수로도 노출 가능 (선호에 따라 사용)
PROMPT_FALLBACK = get_fallback_prompt()

# =============================================================================
# 시스템 프롬프트 로딩
# =============================================================================

async def get_system_prompt(db: Optional[AsyncSession] = None) -> str:
    """
    RAG 분석을 위한 시스템 프롬프트를 반환합니다.
    DB에서 프롬프트를 로드하고, 없으면 블록 변수 기반 폴백을 사용합니다.
    """
    return await get_prompt_with_fallback(
        slug=PROMPT_SLUG,
        fallback_prompt=PROMPT_FALLBACK,
        db=db,
    )

# =============================================================================
# 메시지 빌더
# =============================================================================

def _format_metadata_string(metadata: Dict) -> str:
    """메타데이터를 헤더에 표시할 문자열로 변환한다."""
    if not metadata:
        return ""
    items = []
    for k, v in metadata.items():
        # base64 코드와 이미지 패스는 제외
        if k in ("image_base64", "image_path"):
            continue
        elif k == "properties" and isinstance(v, dict):
            filtered_props = {
                prop_k: prop_v
                for prop_k, prop_v in v.items()
                if prop_k not in ["image_base64", "image_path"]
            }
            if filtered_props:
                items.append(f"  - {k if k is not None else 'None'}: {filtered_props}")
        else:
            items.append(f"  - {k if k is not None else 'None'}: {v if v is not None else 'None'}")
    return f" <메타데이터>{chr(10).join(items)}</메타데이터>" if items else ""

def _is_image_candidate(candidate: CandidateChunk) -> bool:
    """후보가 이미지 타입인지 판별한다."""
    try:
        is_image = str(getattr(candidate, "chunk_type", "")).lower() == "image"
        if not is_image and candidate.metadata:
            label = str(candidate.metadata.get("label", "")).lower()
            is_image = label == "image"
        return is_image
    except Exception:
        return False

def _extract_toc_context(metadata: Dict) -> str:
    """toc_title/toc_summary를 평가용 헤더에 추가하기 위한 헬퍼"""
    try:
        payload = metadata.get("payload", {}) if isinstance(metadata.get("payload"), dict) else {}
        toc_title = (
            payload.get("toc_title")
            or payload.get("section_title")
            or metadata.get("toc_title")
            or metadata.get("section_title")
        )
        toc_summary = (
            payload.get("toc_summary")
            or payload.get("section_summary")
            or metadata.get("toc_summary")
            or metadata.get("section_summary")
        )
        logger.debug(
            "[TOC Context] toc_title=%s | toc_summary_preview=%s",
            toc_title,
            (toc_summary or "")[:80],
        )
        toc_lines = []
        if toc_title:
            toc_lines.append(f"TOC 제목: {toc_title}")
        if toc_summary:
            toc_lines.append(f"TOC 요약: {toc_summary}")
        return "\n".join(toc_lines)
    except Exception:
        return ""

def _get_image_base64(candidate: CandidateChunk) -> Optional[str]:
    """후보에서 이미지 base64 문자열을 추출한다."""
    if not candidate.metadata:
        return None
    return candidate.metadata.get(
        "image_base64",
        candidate.metadata.get("properties", {}).get("image_base64", None),
    )

def _build_text_message_from_candidate(header_text: str, candidate: CandidateChunk) -> HumanMessage:
    """텍스트 후보 메시지 생성 (strip/빈문자열 보호 포함)"""
    if candidate.content is None:
        logger.warning(f"Candidate {candidate.element_id} has no content")
        text_content = ""
    else:
        stripped = str(candidate.content).strip()
        if not stripped:
            logger.warning(f"Candidate {candidate.element_id} content is empty after strip")
            text_content = ""
        else:
            text_content = stripped

    return HumanMessage(content=[
        {"type": "text", "text": header_text},
        {"type": "text", "text": text_content},
    ])

def get_batch_prompt(
    batch: List[CandidateChunk],
    query: str,
    id_to_ref: Dict[str, str]
) -> List[HumanMessage]:
    """
    CandidateChunk 리스트를 멀티모달 HumanMessage 목록으로 변환.
    - 텍스트/이미지 후보 각각에 맞게 content 구성
    - 헤더에는 사용자 질문, ref ID, 컬렉션/타입, 메타데이터 포함
    """
    per_candidate_messages: List[HumanMessage] = []
    for candidate in batch:
        metadata_str = _format_metadata_string(candidate.metadata)
        ref = id_to_ref.get(candidate.element_id, f"ref:unknown:{candidate.element_id}")
        toc_context = ""
        if INCLUDE_TOC_CONTEXT_FOR_SCORING:
            toc_context = _extract_toc_context(candidate.metadata or {})
            toc_context = f"\n{toc_context}" if toc_context else ""
        header_text = (
            f"User Question: {query} | "
            f"Candidate ID:{ref} | {candidate.collection_name}:{candidate.chunk_type} "
            f"{metadata_str}{toc_context}"
        )

        if _is_image_candidate(candidate):
            base64_str = _get_image_base64(candidate)
            if isinstance(base64_str, str) and base64_str.strip():
                data_url = f"data:image/png;base64,{base64_str}"
                per_candidate_messages.append(HumanMessage(content=[
                    {"type": "text", "text": header_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]))
            else:
                logger.warning(f"Candidate {candidate.element_id} is image candidate but has no image base64")
                per_candidate_messages.append(_build_text_message_from_candidate(header_text, candidate))
        else:
            per_candidate_messages.append(_build_text_message_from_candidate(header_text, candidate))

    return per_candidate_messages

def get_batch_prompt_for_candidates(
    candidates: List[Candidate],
    query: str,
    revised_query: Optional[str] = None,
    id_to_ref: Optional[Dict[str, str]] = None,
) -> HumanMessage:
    """
    Candidate 객체들을 위한 멀티모달 HumanMessage 배치 프롬프트 생성.
    - 이미지가 있는 경우 멀티모달 메시지로 구성
    - id_to_ref가 있으면 ref 매핑, 없으면 ref:{idx} 사용
    """
    content_parts = []
    
    # 명확한 구분선으로 시스템 프롬프트와 후보 데이터 분리
    separator = """
# ============================================================
# CANDIDATES TO EVALUATE (Below is the actual data to judge)
# ============================================================
# IMPORTANT: The text below is RAW DATA from the search results.
# Extract evidence ONLY from this data. Do NOT use any information
# from the system prompt examples or instructions above.
# ============================================================
"""
    content_parts.append({"type": "text", "text": separator})
    content_parts.append({"type": "text", "text": f"사용자 질문: {query}\n수정된 질문: {revised_query}\n"})
    content_parts.append({"type": "text", "text": "# ============================================================\n"})
    
    for idx, candidate in enumerate(candidates):
        candidate_id = getattr(candidate, 'id', f"candidate_{idx}")
        ref_id = id_to_ref.get(candidate_id, f"ref:{idx}") if id_to_ref else f"ref:{idx}"
        toc_context = ""
        if INCLUDE_TOC_CONTEXT_FOR_SCORING:
            toc_context = _extract_toc_context(getattr(candidate, "metadata", {}) or {})
            toc_context = f"\n{toc_context}" if toc_context else ""
        
        # # 출처 정보
        # source_info = ""
        # if getattr(candidate, "source", None):
        #     try:
        #         source_info = f"\n출처: {', '.join([str(s) for s in candidate.source])}"
        #     except Exception:
        #         source_info = f"\n출처: {str(candidate.source)}"

        # Cypher 쿼리 정보
        cypher_info = ""
        if getattr(candidate, "cypher_query", None):
            cypher_info = f"\n사용된 Cypher 쿼리: {candidate.cypher_query}"

        # 텍스트 내용 추가 (각 후보마다 구분)
        text_content = f"""
# -------------------------- [{ref_id}] --------------------------
{cypher_info}
{toc_context}
내용: {candidate.content}
# ----------------------------------------------------------------
"""
        content_parts.append({"type": "text", "text": text_content})
        
        # 이미지가 있으면 멀티모달로 추가
        if candidate.is_image:
            for img_base64 in candidate.image_base64:
                if img_base64 and len(img_base64) > 100:
                    data_url = f"data:image/png;base64,{img_base64.strip()}"
                    content_parts.append({"type": "image_url", "image_url": {"url": data_url}})
    
    # 후보 데이터 끝 표시
    content_parts.append({"type": "text", "text": "\n# ============================================================\n# END OF CANDIDATES - Now evaluate and extract evidence\n# ============================================================"})
    
    return HumanMessage(content=content_parts)
