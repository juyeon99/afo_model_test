"""
2025/11/11 kimgeonwoo
답변 시 candidate 형태의 데이터 추가
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional, cast, Callable

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig


from shared.config.config import llm_config
from shared.factory.llm_factory import get_llm
from rag.graph.progress import send_progress
from shared.utils.logging import (
    wrap_long_text, 
    log_section, 
    log_step, 
    log_data
)

from .prompts import get_system_prompt, get_batch_prompt, get_batch_prompt_for_candidates
from .validation import validate_input_data
from .response_formats import (
    create_no_candidates_response, 
    create_success_response, 
    create_error_response, 
    synthesize_analysis_from_evidences
)
from .models import BatchEvidenceLLMResult
from .utils import group_candidates_into_batches, map_ref_id_to_element_id
from rag.schemas.types import (
    GeneralAnswerState,
    Evidence,
    CandidateChunk,
    Candidate,
)

logger = logging.getLogger(__name__)


def _find_toc_title_from_relations(relations: Any) -> Optional[str]:
    """TOC-Section 관계 정보에서 toc title을 추출하기 위한 헬퍼"""
    try:
        if not isinstance(relations, list):
            return None
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            # 명시적 필드 우선
            toc_title = rel.get("toc_title") or rel.get("tocTitle")
            if toc_title:
                return toc_title
            # 노드 정보에 TOC 라벨이 있는 경우 title/name 사용
            for node_key in ("start_node", "end_node", "start", "end"):
                node = rel.get(node_key)
                if not isinstance(node, dict):
                    continue
                labels = []
                if isinstance(node.get("labels"), list):
                    labels = [str(lbl).lower() for lbl in node.get("labels")]
                node_type = str(node.get("type", "")).lower()
                is_toc = "toc" in labels or node_type == "toc"
                if is_toc:
                    title = node.get("title") or node.get("name")
                    if title:
                        return title
    except Exception:
        return None
    return None


def _extract_toc_title(metadata: Dict[str, Any], candidate_result: Optional[Dict[str, Any]] = None, fallback_section_title: Optional[str] = None) -> Optional[str]:
    """payload나 관계 정보에서 toc_title을 최대한 추출 (없으면 섹션 제목으로 폴백)"""
    try:
        # 1) payload 기준 직접 필드
        payload = metadata.get("payload") if isinstance(metadata.get("payload"), dict) else {}
        toc_title = None
        if isinstance(payload, dict):
            toc_title = (
                payload.get("toc_title")
                or payload.get("tocTitle")
                or (isinstance(payload.get("toc"), dict) and payload.get("toc", {}).get("title"))
            )
            if not toc_title:
                toc_title = _find_toc_title_from_relations(
                    payload.get("relations") or payload.get("relationships")
                )
        # 2) 메타데이터 루트에서도 탐색
        if not toc_title and isinstance(metadata, dict):
            toc_title = (
                metadata.get("toc_title")
                or metadata.get("tocTitle")
                or (isinstance(metadata.get("toc"), dict) and metadata.get("toc", {}).get("title"))
            )
            if not toc_title:
                toc_title = _find_toc_title_from_relations(
                    metadata.get("relations") or metadata.get("relationships")
                )
        # 3) candidate.result 내부 context에서 탐색 (그래프 결과용)
        if not toc_title and isinstance(candidate_result, dict):
            toc_title = candidate_result.get("toc_title")
            if not toc_title:
                toc_title = _find_toc_title_from_relations(
                    candidate_result.get("relations") or candidate_result.get("relationships")
                )
            if not toc_title:
                contexts = candidate_result.get("context_sections")
                if isinstance(contexts, list):
                    for ctx in contexts:
                        if isinstance(ctx, dict) and ctx.get("toc_title"):
                            toc_title = ctx.get("toc_title")
                            break
        return toc_title or fallback_section_title
    except Exception:
        return fallback_section_title
async def generate_evidences_for_batch(
    batch: List[CandidateChunk], batch_id: str, query: str
) -> List[Evidence]:
    """단일 LLM 호출로 배치 내 각 후보에 대한 Evidence를 직접 생성한다."""
    try:
        llm = get_llm(llm_config)
        structured_llm = llm.with_structured_output(BatchEvidenceLLMResult)
        # ref 매핑
        ref_to_id = {f"ref:{idx}": candidate.element_id for idx, candidate in enumerate(batch)}
        id_to_ref = {v: k for k, v in ref_to_id.items()}

        batch_messages = get_batch_prompt(batch, query, id_to_ref)
        system_text = await get_system_prompt()

        messages: List[Any] = [
            SystemMessage(content=system_text),
            *batch_messages,
        ]

        try:
            result: BatchEvidenceLLMResult = cast(BatchEvidenceLLMResult, await structured_llm.ainvoke(messages))
            if len(result.candidate_evidences) > len(batch):
                logger.warning(
                    f"{batch_id} [EVID] LLM returned more items than inputs: outputs={len(result.candidate_evidences)}, inputs={len(batch)}"
                )
        except Exception as llm_error:
            logger.error(f"{batch_id} [EVID] LLM API call failed: {llm_error}")
            return []

        # 결과를 Evidence로 변환 (ref -> element_id)
        valid_element_ids = {str(c.element_id) for c in batch}
        candidate_map = {c.element_id: c for c in batch}
        evidences: List[Evidence] = []

        for item in result.candidate_evidences:
            element_id = map_ref_id_to_element_id(item.candidate_id, ref_to_id, valid_element_ids)
            if not element_id:
                continue
            candidate = candidate_map.get(element_id)
            if not candidate:
                continue
            
            # 메타데이터에서 문서 제목 및 섹션 제목 추출
            metadata = candidate.metadata or {}
            doc_title = None
            section_title = None
            toc_summary = None
            
            # 1) Qdrant payload에서 정보 추출 (벡터 검색 결과)
            if isinstance(metadata.get("payload"), dict):
                payload = metadata["payload"]
                doc_title = payload.get("document_title") or payload.get("doc_title")
                section_title = _extract_toc_title(metadata, fallback_section_title=payload.get("section_title"))
                toc_summary = payload.get("toc_summary") or payload.get("section_summary")
                logger.debug(
                    "[Evidence Payload] element_id=%s | toc_title=%s | toc_summary=%s",
                    element_id,
                    section_title,
                    toc_summary,
                )
            
            # 2) element_id에서 문서명 추출 (그래프 검색 결과)
            # Neo4j Section/Fact/Table 노드에는 document_title이 없으므로 element_id에서 파싱
            if not doc_title:
                try:
                    # Section/Fact: "문서명_toc_XXX_sec_YYY" 형식
                    if "_toc_" in element_id:
                        doc_title = element_id.split("_toc_")[0]
                    # Table/Image: "문서명_page_N_table_M" 형식
                    elif "_page_" in element_id:
                        doc_title = element_id.split("_page_")[0]
                except Exception:
                    pass
            
            # source 구성: 문서명과 섹션명을 함께 표시
            if doc_title and section_title:
                source = f"{doc_title} > {section_title} ({candidate.chunk_type})"
            elif doc_title:
                source = f"{doc_title} ({candidate.chunk_type})"
            else:
                source = f"{candidate.collection_name} - {candidate.chunk_type}"
            
            # 로깅: Evidence source 생성 정보
            logger.info(
                "[Evidence Created] element_id=%s | doc_title='%s' | section_title='%s' | toc_summary_preview='%s' | final_source='%s'",
                element_id,
                doc_title,
                section_title,
                (toc_summary or "")[:80],
                source,
            )
            
            evidences.append(
                Evidence(
                    source=source,
                    content=item.evidence,
                    relevance_score=item.relevance_score,
                    source_id=[{"key": "element_id", "value": element_id}],
                    evidence_type="rag_extracted",
                    toc_title=section_title,
                    toc_summary=toc_summary,
                    metadata=metadata,
                )
            )

        return evidences

    except Exception as e:
        logger.error(f"{batch_id} [EVID] unexpected error: {e}", exc_info=True)
        return []


async def generate_evidences_for_candidate_batch(
    batch: List[Candidate], batch_id: str, query: str, revised_query: Optional[str] = None  # CandidateChunk → Candidate로 변경
) -> List[Evidence]:
    """단일 LLM 호출로 배치 내 각 후보에 대한 Evidence를 직접 생성한다."""
    try:
        llm = get_llm(llm_config)
        structured_llm = llm.with_structured_output(BatchEvidenceLLMResult)
        
        # ref 매핑 (Candidate 객체용)
        ref_to_id = {f"ref:{idx}": f"candidate_{idx}" for idx, candidate in enumerate(batch)}
        id_to_ref = {v: k for k, v in ref_to_id.items()}

        batch_messages = get_batch_prompt_for_candidates(batch, query, revised_query, id_to_ref)
        system_text = await get_system_prompt()

        messages = [
            SystemMessage(content=system_text),
            batch_messages,
        ]

        # LLM 입력 로그 (이미지 데이터 제외하고 텍스트만 출력)
        logger.info(f"[LLM Input] batch={batch_id}, candidates={len(batch)}, query={query}")
        if isinstance(batch_messages.content, list):
            text_parts = [p.get("text", "") for p in batch_messages.content if p.get("type") == "text"]
            image_count = sum(1 for p in batch_messages.content if p.get("type") == "image_url")
            logger.info(f"[LLM Input] batch={batch_id}, text={''.join(text_parts)}, images={image_count}")
        else:
            logger.info(f"[LLM Input] batch={batch_id}, messages={batch_messages.content}")

        # LLM 호출
        result: BatchEvidenceLLMResult = cast(BatchEvidenceLLMResult, await structured_llm.ainvoke(messages))
        
        # LLM 출력 로그
        logger.info(f"[LLM Output] batch={batch_id}, evidences_count={len(result.candidate_evidences)}")
        for idx, ev in enumerate(result.candidate_evidences):
            logger.info(f"[LLM Output] batch={batch_id}, idx={idx}, candidate_id={ev.candidate_id}, relevance={ev.relevance_score}, evidence={ev.evidence}")
        
        # Evidence 객체 생성 (relevance threshold 미만은 필터링)
        evidences = []
        RELEVANCE_THRESHOLD = 0.3  # 최소 relevance score
        
        for evidence_item in result.candidate_evidences:  # 올바른 필드명 사용
            # relevance_score가 threshold 미만이면 스킵
            if evidence_item.relevance_score < RELEVANCE_THRESHOLD:
                logger.info(f"[Evidence Filtered] batch={batch_id}, candidate_id={evidence_item.candidate_id}, relevance={evidence_item.relevance_score} < {RELEVANCE_THRESHOLD}, skipping")
                continue
            
            # ref_id를 실제 candidate 정보로 매핑
            candidate_idx = int(evidence_item.candidate_id.split(":")[1])
            candidate = batch[candidate_idx]
            logger.debug("candidate", candidate)
            candidate_result = getattr(candidate, "result", None)
            # source 리스트를 문자열로 변환 (사용자 표시용)
            source_str = "Unknown"
            if candidate.source:
                logger.debug(f"[Evidence ID Extraction] candidate_idx={candidate_idx}, source={candidate.source}")
                source_parts = []
                for src in candidate.source:
                    if isinstance(src, dict):
                        # dict에서 의미있는 정보 추출 (사용자 표시용)
                        if 'document_name' in src:
                            source_parts.append(src['document_name'])
                        elif 'file_name' in src:
                            source_parts.append(src['file_name'])
                        elif 'title' in src:
                            source_parts.append(src['title'])
                        else:
                            # ID 정보는 사용자 표시용 source에서 제외
                            continue
                    else:
                        source_parts.append(str(src))
                source_str = ", ".join(source_parts) if source_parts else "Unknown"
            
            # candidate.source 전체를 source_id로 사용 (드롭다운용)
            source_id_list = candidate.source if candidate.source else []

            evidence = Evidence(
                source=source_str,  # 문자열로 변환된 소스
                content=evidence_item.evidence,  # 올바른 필드명 사용
                relevance_score=evidence_item.relevance_score,
                toc_title=_extract_toc_title(candidate.metadata or {}, candidate_result=candidate_result),
                toc_summary=getattr(candidate, "metadata", {}).get("toc_summary")
                if isinstance(getattr(candidate, "metadata", {}), dict)
                else None,
                source_id=source_id_list,  # candidate.source 전체 사용
                evidence_type="rag",
                metadata={
                    "original_result": candidate.result,
                    "original_source": candidate.source,  # 원본 소스 정보 보존
                    "ref_id": evidence_item.candidate_id,
                    "batch_id": batch_id
                }
            )
            logger.info(f"[Evidence Created] batch={batch_id}, candidate_idx={candidate_idx}, source_ids_count={len(source_id_list)}, relevance={evidence_item.relevance_score}")
            evidences.append(evidence)
        
        return evidences
        
    except Exception as e:
        logger.error(f"배치 {batch_id} Evidence 생성 실패: {e}")
        return []


async def gather_direct_evidences(
    batches: List[List[CandidateChunk]],
    user_query: str,
    callback: Optional[Callable] = None,
) -> List[Evidence]:
    """
    모든 배치에서 직접 Evidence를 수집 (스트리밍).
    """

    # 작업 생성 및 매핑
    created_tasks: List[asyncio.Task] = []
    for i, batch in enumerate(batches):
        t = asyncio.create_task(generate_evidences_for_batch(batch, f"batch_{i+1}", user_query))
        created_tasks.append(t)

    evidences: List[Evidence] = []
    total_batches = len(batches)
    success = 0

    # 완료되는 순서대로 처리
    for fut in asyncio.as_completed(created_tasks):
        try:
            result = await fut
            items = cast(List[Evidence], result)
            evidences.extend(items)
            success += 1
            await send_progress(
                callback,
                "RAG Candidate Analyzer",
                f"Evidences {success}/{total_batches} generated",
                int(20 + (success / total_batches) * 70),
            )
        except Exception as err:
            logger.error(f"[EVID] generation failed: {err}", exc_info=True)

    return evidences


async def gather_direct_evidences_for_candidates(
    batches: List[List[Candidate]],
    user_query: str,
    revised_query: Optional[str] = None,
    callback: Optional[Callable] = None,
) -> List[Evidence]:
    """
    모든 Candidate 배치에서 직접 Evidence를 수집 (스트리밍).
    """

    # 작업 생성 및 매핑
    created_tasks: List[asyncio.Task] = []
    for i, batch in enumerate(batches):
        t = asyncio.create_task(generate_evidences_for_candidate_batch(batch, f"batch_{i+1}", user_query, revised_query))
        created_tasks.append(t)

    evidences: List[Evidence] = []
    total_batches = len(batches)
    success = 0

    # 완료되는 순서대로 처리
    for fut in asyncio.as_completed(created_tasks):
        try:
            result = await fut
            items = cast(List[Evidence], result)
            evidences.extend(items)
            success += 1
            await send_progress(
                callback,
                "RAG Candidate Analyzer",
                f"Evidences {success}/{total_batches} generated",
                int(20 + (success / total_batches) * 70),
            )
        except Exception as err:
            logger.error(f"[EVID] generation failed: {err}", exc_info=True)

    return evidences


async def rag_analyzer_node(
    state: GeneralAnswerState, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """RAG 검색 후보들을 배치 단위로 분석하고 정보 적합성을 평가하는 노드 (Candidate 직접 처리)"""
    
    log_section(logger, "RAG CANDIDATE ANALYZER (ASYNC)")
    configurable = config.get("configurable", {}) if config else {}
    callback = configurable.get("progress_callback")

    # 초기 진행 상황 메시지
    await send_progress(
        callback,
        "RAG Candidate Analyzer",
        "Starting analysis of RAG retrieved candidates...",
        0,
    )

    try:
        # 입력 데이터 검증 (이제 List[Candidate] 반환)
        instruction, user_query, revised_query, candidates = validate_input_data(state)

        if not candidates:
            await send_progress(
                callback,
                "RAG Candidate Analyzer",
                "No candidates available for analysis.",
                100,
            )
            return create_no_candidates_response(state)

        # 1단계: 후보들을 배치로 그룹핑 (Candidate 객체용)
        batches = group_candidates_into_batches(candidates, max_batch_size=3)
        
        await send_progress(
            callback,
            "RAG Candidate Analyzer",
            f"Grouped {len(candidates)} candidates into {len(batches)} batches.",
            20,
        )

        # 2단계: 배치별로 직접 Evidence 생성 (Candidate용 스트리밍)
        evidences = await gather_direct_evidences_for_candidates(batches, user_query, revised_query, callback)

        # 3단계: 생성된 Evidence로 간단 분석 산출
        analysis_result = synthesize_analysis_from_evidences(
            evidences, candidates
        )

        await send_progress(callback, "RAG Candidate Analyzer", "Finished analysis of RAG retrieved candidates.", 100)
        
        # 기본 응답 생성
        response = create_success_response(
            state, evidences, analysis_result, user_query, candidates, batches
        )
        
        # 4단계: Evidence가 생성되었으면 자동으로 data_generator 실행
        if evidences:
            logger.info("🔄 RAG Analyzer: Evidence 생성 완료, data_generator 자동 실행 시작")
            try:
                # data_generator_node를 임포트하고 실행
                from rag.nodes.data_generator.generate_data import data_generator_node
                
                # 현재 state에 response를 병합하여 새로운 state 생성
                updated_state = GeneralAnswerState(**{**state.__dict__, **response})
                
                # data_generator 실행
                data_gen_result = await data_generator_node(updated_state, config)
                
                # data_generator 결과를 response에 병합
                if data_gen_result and isinstance(data_gen_result, dict):
                    logger.info(f"🔄 RAG Analyzer: data_generator 실행 완료, 결과 병합")
                    response.update(data_gen_result)
                else:
                    logger.warning("🔄 RAG Analyzer: data_generator가 빈 결과를 반환함")
                    
            except Exception as data_gen_error:
                logger.error(f"🔄 RAG Analyzer: data_generator 실행 중 오류 발생: {data_gen_error}", exc_info=True)
                # data_generator 실패는 치명적이지 않으므로 계속 진행
        else:
            logger.info("🔄 RAG Analyzer: Evidence가 없어 data_generator 스킵")
        
        return response

    except Exception as e:
        await send_progress(callback, "RAG Candidate Analyzer", "Finished analysis of RAG retrieved candidates.", 100)
        logger.error(f"Error in RAG candidate analysis: {e}", exc_info=True)
        return create_error_response(state, e)
