"""
최초 작성자: 이진혁
최초 작성일: 2025-09-12
최초 작성 목적: 프롬프트 생성 관련 함수 관리

수정자 : 김건우
수정일 2025-10-28
수정목적: 육하원칙을 따르는 답변 형식 출력

수정자: 윤은
수정일: 2025-11-12
수정목적: 프롬프트 생성 관련 함수 리팩토링
- 프롬프트 블럭화하여 DB로 관리
- 공통으로 사용 가능한 Util 함수 shared/utils/prompt_utils.py 추가
    - sanitize_content: 민감한 정보를 필터링. sensitive_words를 filter_word로 대체
    - select_under_token_budget: 증거를 먼저 최대한 담고, 남는 토큰으로 candidate를 채워 넣습니다.
    - build_bounded_info_sections: 토큰 예산(max_tokens) 내에서 Evidence를 먼저, 이후 Candidate를 포함하는 섹션 구성
"""

from logging import getLogger
from typing import List, Any, Optional, Tuple, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from sqlalchemy.ext.asyncio import AsyncSession

from shared.utils.prompt_utils import sanitize_content, build_bounded_info_sections
from shared.utils.node_prompt_loader import render_blocks_to_string
from rag.schemas.types import Evidence, VisualizationResult, GeneralAnswerState

from .models import SynthesisData
from .config import CONFIG

logger = getLogger(__name__)

# 노드 레지스트리용 프롬프트 슬러그
PROMPT_SLUG = "synthesizer"

# TOC 제목/요약을 Evidence 출력에 포함할지 여부 (불필요하면 False로 변경하거나 블럭을 주석 처리)
INCLUDE_TOC_CONTEXT = True


# ============================================================================
# 프롬프트 블럭 상수 정의 (DB에 저장될 내용들)
# ============================================================================

BLOCK_BASE_INSTRUCTION = """## PRIMARY DIRECTIVE: KOREAN LANGUAGE OUTPUT ONLY
You are an expert Korean-speaking AI analyst. ALL responses MUST be in Korean language.
영어 사용 금지. 한국어로만 답변하세요."""

BLOCK_CONFIDENTIALITY_RULES = """## CRITICAL RESPONSE CONSTRAINTS

### 1. ABSOLUTE CONFIDENTIALITY RULES
- NEVER reveal technical details (file names, column names, table names, variables)
- NEVER mention data structure (rows, columns, schema, fields)
- NEVER disclose processing errors or system messages
- IGNORE and DO NOT mention any failed data processing
- Use only abstract terms: "고객 정보", "분석 데이터", "내부 자료"

### 2. NO REPETITION RULE
- EVERY sentence must contain NEW information
- EVERY claim must be supported by data-driven reasoning
- NO redundant statements or circular explanations
- Each insight must add unique value

### 3. TONE REQUIREMENT
- ALWAYS maintain a polite, professional Korean tone
- Use respectful language (존댓말)
- Express insights with confidence yet humility"""

BLOCK_OUTPUT_STRUCTURE_5W1H = """## OUTPUT STRUCTURE

Begin your response with relevant Korean greeting, then:

1. **핵심 발견사항** (Key Findings)
- 질문에 대한 3-5개의 주요 인사이트 요약
- 각 인사이트는 데이터로 뒷받침

2. **육하원칙 기반 상세 분석** (Detailed Analysis Based on 5W1H)

### 📌 누가 (Who) - 주요 인물/기관/그룹
- 분석 대상이 되는 인물, 기관, 고객 그룹 등을 명확히 식별
- 각 주체의 특성과 역할 설명
- 예: "30대 여성 고객층", "서울 지역 구매자", "A기관"

### 📅 언제 (When) - 시간/시기
- 가장 구체적인 시간 단위로 추출 (2025.01.01 > 2025년 1월 > 2025년)
- 시계열 패턴, 주기성, 트렌드 변화 시점
- 예: "2024년 3분기", "2025년 1월 15일", "최근 6개월간"

### 📍 어디서 (Where) - 장소/지역
- 지리적 위치, 지역별 특성
- 공간적 분포와 패턴
- 예: "서울 강남구", "온라인 채널", "A지점"

### 🎯 무엇을 (What) - 사건/행적/현상
- 주요 사건, 구매 행동, 발생한 현상
- 문서, 문단, 표 단위로 추출된 핵심 사실
- 구체적인 제품명, 서비스명, 활동 내역
- 예: "프리미엄 제품 구매 급증", "고객 이탈 현상", "신규 서비스 도입"

### ⚙️ 어떻게 (How) - 방법/평가/수준
- 실행 방법, 프로세스, 메커니즘
- 정량적 평가: 수치, 비율, 증감률
- 정성적 평가: 긍정/부정, 상/중/하, 우수/보통/미흡
- 예: "전년 대비 25% 증가", "만족도 '상' 수준", "온라인 결제 방식으로"

### 💡 왜 (Why) - 원인/목적/배경
- 행동의 동기와 심리적 요인
- 현상의 근본 원인과 배경
- 사회문화적 맥락과 영향 요인
- 전략적 목적과 의도

3. **전략적 시사점** (Strategic Implications)
- 실행 가능한 권장사항
- 구체적인 실행 방법과 목적 명시

4. **결론** (Conclusion)
- 핵심 메시지 재정리
- 미래 지향적 관점

5. **요약** (Executive Summary)
- 3-5문장 이내 압축"""

BLOCK_CITATION_WITH_REF = """### CITATION REQUIREMENTS
- 각 문장 또는 문단 끝에 해당 근거의 참조를 [ref:k] 형태로 표시하세요.
- 위의 AVAILABLE INFORMATION 섹션에 부여된 ref 라벨을 그대로 사용하세요.
- 새로운 ref를 만들지 마세요.
- 여러 ref를 동시에 인용할 때는 각 ref를 별도의 대괄호로 각각 표시하세요. 예: [ref:2] [ref:5]
- 절대 한 쌍의 대괄호 안에 여러 ref를 넣지 마세요. 잘못된 예: [ref:2, ref:5]
- 답변의 근거를 무조건 표시하세요. 문장단위로 표시하세요.
- ref가 없는 경우 근거를 표시하지 마세요."""

BLOCK_FINAL_CHECKLIST = """## FINAL CHECKLIST
✓ 모든 답변이 한국어로 작성되었는가?
✓ 반복되는 내용 없이 새로운 정보만 포함했는가?
✓ 데이터 구조나 기술적 세부사항을 노출하지 않았는가?
✓ 정중하고 전문적인 어조를 유지했는가?
✓ 모든 그룹 조합을 빠짐없이 분석했는가?
✓ 각 주장이 데이터 분석에 근거하는가?
✓ 답변 속, 말의 모순이나 아이러니는 없는가?"""

BLOCK_OUTPUT_STRUCTURE_USER_QUERY_FORMAT = """## OUTPUT STRUCTURE

Begin your response with relevant Korean greeting, then:

1. **핵심 발견사항** (Key Findings)
- 질문에 대한 3-5개의 주요 인사이트 요약
- 각 인사이트는 데이터로 뒷받침

2. **답변내용** (User Query Format)
- 유저의 질문에 대한 직접적인 답변을 제공
- 단답형 질문이라도 충분한 맥락과 근거를 함께 제시
- **시간에 따라 다른 정보가 있는 경우**: 각 시점별로 명확히 구분하여 설명 (예: "2024년 1월~6월: A, 2024년 7월~현재: B")
- **여러 증거가 있는 경우**: 각 증거의 시점/출처를 명시하고 종합하여 설명
- 단순히 이름, 숫자만 나열하지 말고 "언제", "왜", "어떤 맥락에서" 등을 함께 설명

3. **요약** (Executive Summary)
- 답변 내용에 대한 간단한 요약
- 구체적인 숫자, 날짜, 장소, 인물 등 키워드 포함
- 3-5문장 이내로 압축"""

BLOCK_RESPONSE_FRAMEWORK = """## RESPONSE FRAMEWORK

**IMPORTANT ADDITION:** Summarize the answer in a short executive summary after giving the full structured response.

### Question Type Classification
Analyze the question and follow the appropriate path:

1. **Group-Based Analysis** (e.g., "지역별 및 연령별 분석")
- Generate blocks for EVERY group combination
- Maintain identical structure for each group
- Include ALL combinations without exception

2. **Ranking/Top-N Questions** (e.g., "상위 10개 제품")
- Return EXACTLY N distinct items
- Provide unique insights for each item
- No duplicates unless explicitly different

3. **Summary/Trend Questions**
- Synthesize overall patterns
- Highlight key findings and anomalies

### Content Requirements

1. **Depth of Analysis**
- Psychological motivations behind behaviors
- Socio-cultural context and influences
- Real-world applications and scenarios
- Future implications and strategies

2. **Specificity Standards**
- Include concrete examples from data
- Provide actionable recommendations
- Connect insights to business impact

3. **Completeness Check**
- Address ALL parts of multi-part questions
- Include ALL requested group combinations
- Cover ALL data categories mentioned

4. **Avoid Contradictions/Inconsistencies**
- Ensure all statements maintain strict logical consistency throughout the response.
- Avoid self-contradictory or ironic phrasing. All parts of a statement, and statements within a paragraph or section, must align logically.
- **Example of what to avoid:** "Purchases of office supplies remain high, but interest in technology products is relatively low. They are likely to share information about simple technology usage related to daily life, health, or leisure activities." (This is contradictory: low interest in tech vs. high likelihood of sharing tech info.)
- If a nuance exists (e.g., low interest in *complex* tech vs. high interest in *simple, daily-life* tech), clarify it explicitly rather than stating a general contradiction.
"""


def _collect_image_uris_from_evidences(evidences: List[Evidence]) -> List[str]:
    """rag_evidences의 metadata에서 image_base64를 추출하여 data URI 목록 생성"""
    image_uris: List[str] = []
    for ev in evidences or []:
        try:
            md = ev.metadata or {}
            b64 = md.get("image_base64")
            if isinstance(b64, str) and b64.strip():
                image_uris.append(f"data:image/png;base64,{b64}")
        except Exception:
            continue
    return image_uris


def _extract_conversation_history(state: GeneralAnswerState) -> str:
    """대화 히스토리를 추출하고 내부 정보를 필터링"""
    recent_messages = []

    # 최근 8개 메시지만 (너무 길어지지 않도록)
    for msg in state.messages[-8:]:
        if hasattr(msg, "content") and msg.content:
            msg_type = msg.__class__.__name__
            # 내용을 필터링 (content가 list일 수 있어 문자열화)
            sensitive_words = [r"superstore", r"\.csv\s*파일", r"\.xlsx?\s*파일"]
            sanitized_content = sanitize_content(
                content=str(msg.content),
                sensitive_words=sensitive_words,
                filter_word="data"
            )

            if msg_type == "HumanMessage":
                recent_messages.append(f"👤 User: {sanitized_content}")
            elif msg_type == "AIMessage":
                recent_messages.append(f"🤖 AI: {sanitized_content}")

    return "\n".join(recent_messages) if recent_messages else "Conversation start"


def extract_synthesis_data(state: GeneralAnswerState) -> SynthesisData:
    """State에서 합성에 필요한 데이터를 추출"""
    # 최근 사용자 메시지 가져오기
    # TODO: 쿼리를 뭘 사용할지 결정 필요
    original_query = state.original_query or state.revised_query or ""
    current_instruction = state.current_instruction or ""

    # 대화 히스토리 추출
    conversation_history = _extract_conversation_history(state)

    # evidence에서 이미지(data URI) 수집
    rag_evidences: List[Evidence] = state.rag_evidences
    evidence_image_uris = _collect_image_uris_from_evidences(rag_evidences)

    return SynthesisData(
        original_query=original_query,
        revised_query=state.revised_query,
        current_instruction=current_instruction,
        rag_evidences=rag_evidences,
        analysis_results=state.analysis_results,
        visualization_results=state.visualization_results,
        ranked_candidates=state.ranked_candidates,
        forced_synthesis=state.forced_synthesis,
        conversation_history=conversation_history,
        formatted_text=state.formatted_context,
        image_base64_list=evidence_image_uris,
    )


def _format_evidence_content(
    evidences: List[Evidence], element_to_ref_map: Dict[str, str] | None = None
) -> str:
    """증거 리스트를 포맷팅된 문자열로 변환.

    - element_to_ref_map이 주어지면 각 항목에 [ref:*] 라벨을 부착한다.
    - ref 태그가 있는 Evidence만 포함하고, ref 번호를 Evidence 번호로 사용한다.
    """
    lines: List[str] = []
    evidence_counter = 1  # 실제 표시되는 Evidence 번호
    
    for i, ev in enumerate(evidences):
        # source_id가 리스트이므로 첫 번째 항목을 사용 (임시 호환성)
        source_id = ev.source_id[0]['value'] if ev.source_id and len(ev.source_id) > 0 and 'value' in ev.source_id[0] else f"evidence_{i}"
        
        # ref 태그가 있는 Evidence만 포함
        if element_to_ref_map and source_id in element_to_ref_map:
            ref_label = element_to_ref_map[source_id]  # 예: "ref:1"
            ref_number = ref_label.split(':')[1] if ':' in ref_label else str(evidence_counter)  # "1" 추출
            source_info = f"**출처:** {ev.source}\n" if ev.source else ""
            # 필요시 주석 처리 쉽게: toc_title/toc_summary를 Evidence 표시문에 포함
            toc_info = ""
            if INCLUDE_TOC_CONTEXT:
                metadata = ev.metadata or {}
                payload = metadata.get("payload", {}) if isinstance(metadata.get("payload"), dict) else {}
                toc_title = ev.toc_title or payload.get("toc_title") or payload.get("section_title")
                toc_summary = ev.toc_summary or payload.get("toc_summary") or payload.get("section_summary")
                if toc_title or toc_summary:
                    toc_lines: List[str] = []
                    if toc_title:
                        toc_lines.append(f"TOC 제목: {toc_title}")
                    if toc_summary:
                        toc_lines.append(f"TOC 요약: {toc_summary}")
                    toc_info = "\n".join(toc_lines) + "\n"
            
            # 로깅: Evidence 출처 정보 확인
            logger.info(f"[Evidence {ref_number}] source='{ev.source}' | relevance={ev.relevance_score:.2f} | content_preview={ev.content[:100]}...")
            
            lines.append(
                f"**{ref_number}. Evidence [{ref_label}]** (Relevance: {ev.relevance_score:.2f})\n{source_info}{toc_info}{ev.content}"
            )
            evidence_counter += 1
        else:
            # ref 태그가 없는 Evidence는 제외 (로그만 남김)
            logger.debug(f"[Evidence Skipped {i+1}] No ref mapping | source='{ev.source}' | relevance={ev.relevance_score:.2f}")
    
    formatted_result = "\n".join(lines)
    logger.debug(f"[Formatted Evidence] Total {len(lines)} items with refs:\n{formatted_result[:500]}...")
    return formatted_result

def _format_visualization_content(
    visualization_results: List[VisualizationResult],
) -> str:
    """시각화 결과 리스트를 포맷팅된 문자열로 변환"""
    return "\n".join(
        [
            f"• {result.instruction} ({result.chart_type})"
            for result in visualization_results
        ]
    )


def _format_results_content(results: List[Any]) -> str:
    """결과 리스트를 포맷팅된 문자열로 변환"""
    return "\n".join([f"• {result}" for result in results])


def _build_bounded_info_sections(
    evidences: List[Evidence], 
    max_tokens: int = 30000
) -> Tuple[List[str], List[str], Dict[str, str], Dict[str, Any]]:
    """
    Evidence 전용 토큰 예산 관리 함수 (범용 함수 래퍼)
    
    Args:
        evidences: Evidence 리스트
        max_tokens: 최대 토큰 수
    
    Returns:
        (섹션 리스트, 선택된 ID 리스트, element_to_ref 매핑, ref_to_element 매핑)
    """
    # relevance_score 기준 내림차순 정렬 (높은 relevance 우선 선택)
    sorted_evidences = sorted(
        evidences,
        key=lambda ev: ev.relevance_score,
        reverse=True
    )
    logger.info(f"[Evidence Sorting] Sorted {len(sorted_evidences)} evidences by relevance (desc)")
    if sorted_evidences:
        logger.info(f"  Top relevance: {sorted_evidences[0].relevance_score:.2f}, Bottom: {sorted_evidences[-1].relevance_score:.2f}")
    
    # 범용 함수 호출을 위한 헬퍼 함수들
    def content_getter(ev: Evidence) -> str:
        return ev.content or ""
    
    def id_getter(ev: Evidence) -> str:
        # source_id가 리스트이므로 첫 번째 항목의 value를 반환 (ref 매핑용)
        if ev.source_id and len(ev.source_id) > 0 and 'value' in ev.source_id[0]:
            return ev.source_id[0]['value']
        return ""
    
    def filter_func(ev: Evidence) -> bool:
        return ev.relevance_score >= CONFIG.RELEVANCE_SCORE_THRESHOLD
    
    def formatter(items: List[Evidence], ref_map: Dict[str, str]) -> str:
        return _format_evidence_content(items, ref_map)
    
    def full_id_getter(ev: Evidence) -> List[Dict[str, str]]:
        # 전체 source_id 리스트를 반환
        return ev.source_id if ev.source_id else []
    
    # 범용 함수 호출 (정렬된 evidences 사용)
    return build_bounded_info_sections(
        items=sorted_evidences,
        max_tokens=max_tokens,
        content_getter=content_getter,
        id_getter=id_getter,
        formatter=formatter,
        filter_func=filter_func,
        section_title="Bounded Analyzed Evidence",
        ref_prefix="ref",
        full_id_getter=full_id_getter
    )


def build_multimodal_evidence_images_message(
    evidences: List[Evidence], element_to_ref_map: Dict[str, str]
) -> Optional[HumanMessage]:
    """Evidence 메타데이터의 이미지를 멀티모달 메시지로 구성하며 [ref:*] 태그를 함께 포함한다."""
    content_blocks: List[dict] = []
    for ev in evidences or []:
        try:
            md = ev.metadata or {}
            b64 = md.get("image_base64", md.get("properties", {}).get("image_base64", None))
            if not (isinstance(b64, str) and b64.strip()):
                continue
            source_id = str(ev.source_id or "")
            ref = element_to_ref_map.get(source_id)
            if ref:
                content_blocks.append({"type": "text", "text": f"Image {ref}"})
            url = f"data:image/png;base64,{b64}"
            content_blocks.append({"type": "image_url", "image_url": {"url": url}})
        except Exception:
            continue

    if not content_blocks:
        return None
    logger.debug("Multimodal evidence images: %d", len(content_blocks))
    return HumanMessage(content=content_blocks)  # type: ignore[arg-type]



async def build_prompt_from_blocks(
    data: SynthesisData,
    db: Optional[AsyncSession] = None,
    use_user_query_format: bool = False,
    info_sections: Optional[List[str]] = None
) -> str:
    """
    블럭을 조합하여 프롬프트 생성

    Args:
        data: 합성 데이터
        db: DB 세션 (선택적)
        use_user_query_format: User Query Format 사용 여부 (기본값: False, 5W1H 사용)
        info_sections: 미리 계산된 정보 섹션 (선택적). None이면 내부에서 계산

    Returns:
        완성된 프롬프트 문자열
    """
    # 정보 섹션 구성 (전달받지 않은 경우에만 계산)
    if info_sections is None:
        info_sections, _, _, _ = _build_bounded_info_sections(data.rag_evidences, max_tokens=8192)
    
    if data.analysis_results:
        analysis_content = _format_results_content(data.analysis_results)
        info_sections.append(f"**📊 Data Analysis Results ({len(data.analysis_results)} items):**\n{analysis_content}")
    
    if data.visualization_results:
        visualization_content = _format_visualization_content(data.visualization_results)
        info_sections.append(f"**📈 Visualization Results ({len(data.visualization_results)} items):**\n{visualization_content}")
    
    # 블럭 구성
    blocks = [
        {"type": "text", "content": BLOCK_BASE_INSTRUCTION},
        {
            "type": "text", 
            "content": f"""## TASK CONTEXT
**User Question (Original):** {data.original_query}
**User Question (Refined):** {data.revised_query}
**Supervisor Instruction:** {data.current_instruction}

## CONVERSATION HISTORY
{data.conversation_history}

## AVAILABLE INFORMATION
{chr(10).join(info_sections) if info_sections else 'No information collected.'}"""
        },
        {"type": "text", "content": BLOCK_CONFIDENTIALITY_RULES},
        {"type": "text", "content": BLOCK_RESPONSE_FRAMEWORK},  
    ]
    
    # 출력 구조 선택
    if use_user_query_format:
        blocks.append({"type": "text", "content": BLOCK_OUTPUT_STRUCTURE_USER_QUERY_FORMAT})
    else:
        blocks.append({"type": "text", "content": BLOCK_OUTPUT_STRUCTURE_5W1H})
    
    # 모드별 지침
    mode_text = "**⚠️ Limited Information Mode**: Work only with available verified facts. State limitations clearly." if data.forced_synthesis else "**✅ Sufficient Information Mode**: Utilize all collected information comprehensively."
    blocks.append({"type": "text", "content": f"## MODE-SPECIFIC GUIDANCE\n{mode_text}"})
    
    blocks.append({"type": "text", "content": BLOCK_FINAL_CHECKLIST})
    
    # 증거가 있으면 인용 지침 추가
    if data.rag_evidences:
        blocks.append({"type": "text", "content": BLOCK_CITATION_WITH_REF})
    
    blocks.append({"type": "text", "content": "\n<|assistant|>\n분석 결과를 말씀드리겠습니다."})
    
    # 렌더링
    return await render_blocks_to_string(blocks, {}, db)


async def get_system_prompt_and_collect_ids(data: SynthesisData) -> Tuple[SystemMessage, List[str], Dict[str, str], Dict[str, Any]]:
    """
    프롬프트 생성 (레거시 호환)

    블럭 기반 시스템을 사용합니다.
    """
    # 토큰 예산 내 정보 섹션 구성 (Evidence) - IDs 수집용 (한 번만 호출)
    info_sections, used_evidence_ids, element_to_ref_map, ref_to_element_map = _build_bounded_info_sections(data.rag_evidences, max_tokens=8192)

    # 블럭 기반 프롬프트 생성 (이미 계산된 info_sections 전달)
    prompt_content = await build_prompt_from_blocks(data, db=None, info_sections=info_sections)
    system_prompt = SystemMessage(content=f"<|system|>\n{prompt_content}")

    return system_prompt, used_evidence_ids, element_to_ref_map, ref_to_element_map

async def get_system_prompt_user_query_format_and_collect_ids(data: SynthesisData) -> Tuple[SystemMessage, List[str], Dict[str, str], Dict[str, Any]]:
    """
    프롬프트 생성 (User Query Format) - 블럭 기반

    유저의 질문 형식에 맞춰 답변하는 프롬프트를 생성합니다.
    """
    # 토큰 예산 내 정보 섹션 구성 (Evidence) - IDs 수집용 (한 번만 호출)
    info_sections, used_evidence_ids, element_to_ref_map, ref_to_element_map = _build_bounded_info_sections(data.rag_evidences, max_tokens=8192)

    # 블럭 기반 프롬프트 생성 (이미 계산된 info_sections 전달)
    prompt_content = await build_prompt_from_blocks(data, db=None, use_user_query_format=True, info_sections=info_sections)
    system_prompt = SystemMessage(content=f"<|system|>\n{prompt_content}")

    return system_prompt, used_evidence_ids, element_to_ref_map, ref_to_element_map
