"""
생성된 Cypher 쿼리를 Neo4j 그래프 DB에서 실행하는 노드
실행 시 검색된 결과를 image_search.py와 동일한 형태로 반환
실패 시 에러 정보 포함하여 반환
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import re
from langchain_core.runnables import RunnableConfig

from rag.schemas.types import UnifiedRAGState, Candidate, CypherQuery
from shared.db import async_neo4j_db

logger = logging.getLogger(__name__)


async def cypher_query_execute_node(
    state: UnifiedRAGState, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    success=None인 Cypher 쿼리들을 순차적으로 실행하고 결과를 통합하여 반환하는 노드
    
    Args:
        state: UnifiedRAGState - cypher_query 필드에 CypherQuery 객체 리스트 포함
        config: RunnableConfig - 설정 정보
        
    Returns:
        Dict containing:
        - cypher_query: 업데이트된 CypherQuery 객체 리스트 (success/error 필드 업데이트됨)
        - candidate_group: 모든 쿼리의 통합 검색 결과 (Candidate 리스트)
    """
    try:
        logger.info("Cypher 쿼리 실행 시작")
        
        # state에서 CypherQuery 객체 리스트 가져오기
        cypher_query_objects = getattr(state, 'cypher_query', [])
        
        if not cypher_query_objects:
            logger.warning("cypher_query 리스트가 비어있습니다")
            # 기존 candidate_group 유지
            existing_candidates = getattr(state, 'candidate_group', []) or []
            return {
                "candidate_group": existing_candidates,
            }
        
        # CypherQuery 객체가 아닌 경우 처리 (하위 호환성)
        if cypher_query_objects and isinstance(cypher_query_objects[0], str):
            logger.warning("문자열 형태의 쿼리가 전달되었습니다. CypherQuery 객체로 변환합니다.")
            cypher_query_objects = [
                CypherQuery(cypher_query=q, schema_info="", success=None, error=None, reuse=None)
                for q in cypher_query_objects if q and q.strip()
            ]
        
        # success=None인 쿼리만 필터링 (아직 실행되지 않은 쿼리)
        pending_queries = [q for q in cypher_query_objects if q.success is None and q.cypher_query.strip()]
        
        if not pending_queries:
            logger.info("실행할 대기 중인 Cypher 쿼리가 없습니다 (모든 쿼리가 이미 실행됨)")
            # 기존 candidate_group 유지
            existing_candidates = getattr(state, 'candidate_group', []) or []
            logger.info(f"대기 쿼리 없음: 기존 candidate {len(existing_candidates)}개 유지")
            return {
                "candidate_group": existing_candidates,
            }
        
        logger.info(f"총 {len(pending_queries)}개의 대기 중인 쿼리를 병렬로 실행합니다")
        
        # 공통 쿼리 파라미터 준비
        query_params = {}
        if state.keyword_mapping_results:
            keyword_mapping_list = [item for v in state.keyword_mapping_results.values() for item in v]
            pattern = re.compile(r'value\s*:\s*"([^"]*)"')
            keyword_values = [
                m.group(1) for s in keyword_mapping_list if (m := pattern.search(s))
            ]
            query_params["keyword_values"] = keyword_values
        else:
            query_params["keyword_values"] = state.raw_keywords
        if state.selected_storage_paths:
            query_params["selected_storage_paths"] = state.selected_storage_paths
        
        async def execute_single_query(
            query_obj: CypherQuery, 
            index: int, 
            total: int
        ) -> Tuple[int, bool, List[Candidate], Dict[str, Any]]:
            """단일 Cypher 쿼리를 실행하고 결과 반환"""
            cypher_query = query_obj.cypher_query
            try:
                # 쿼리에 필터 섹션 추가
                if state.selected_storage_paths:
                    if query_obj.schema_info != "ontology":
                        full_query = _get_document_filter_section() + "\n" + cypher_query
                    else:
                        full_query = _get_canonical_filter_section() + "\n" + cypher_query
                else:
                    full_query = cypher_query
                
                # LIMIT 절 처리: 없으면 LIMIT 100 추가, 있으면 +10 증가
                full_query = full_query.rstrip().rstrip(';')
                limit_match = re.search(r'\bLIMIT\s+(\d+)\s*$', full_query, re.IGNORECASE)
                if limit_match:
                    current_limit = int(limit_match.group(1))
                    new_limit = current_limit + 10
                    full_query = re.sub(r'\bLIMIT\s+\d+\s*$', f'LIMIT {new_limit}', full_query, flags=re.IGNORECASE)
                else:
                    full_query = full_query + " LIMIT 100"
                
                logger.info(f"쿼리 {index+1}/{total} 실행 시작: {full_query[:100]}...")
                
                # Neo4j에서 쿼리 실행 (20초 타임아웃)
                result = await asyncio.wait_for(
                    async_neo4j_db.execute_generate_query(full_query, **query_params),
                    timeout=20.0
                )

                # Section 결과에 상위 TOC 요약/텍스트를 주입하여 Evidence 스코어링 시 함께 참조
                result = await _attach_toc_summaries(result)
                
                if not result:
                    logger.info(f"쿼리 {index+1} 실행 결과가 비어있습니다")
                    query_obj.success = False
                    query_obj.error = "There are no search result"
                    return (index, False, [], {
                        "query_index": index+1,
                        "success": False,
                        "result_count": 0,
                        "error": "There are no search result"
                    })
                
                # schema_info가 "auto"가 아닌 경우(LLM 생성 쿼리)만 cypher_query 정보 포함
                query_for_candidate = cypher_query if query_obj.schema_info != "auto" else None
                candidate_group = _convert_to_records(result, cypher_query=query_for_candidate)
                result_count = len(candidate_group)
                
                logger.info(f"쿼리 {index+1} 실행 완료: {result_count}개 결과")
                query_obj.success = True
                query_obj.error = None
                
                return (index, True, candidate_group, {
                    "query_index": index+1,
                    "success": True,
                    "result_count": result_count,
                    "error": None
                })
                
            except asyncio.TimeoutError:
                logger.error(f"쿼리 {index+1} 실행 타임아웃 (20초)")
                query_obj.success = False
                query_obj.error = "쿼리 실행 타임아웃 (20초 초과)"
                return (index, False, [], {
                    "query_index": index+1,
                    "success": False,
                    "result_count": 0,
                    "error": "쿼리 실행 타임아웃 (20초 초과)"
                })
                
            except Exception as db_error:
                logger.error(f"쿼리 {index+1} 실행 중 오류: {db_error}")
                error_message = str(db_error)
                if "SyntaxError" in error_message or "Invalid" in error_message:
                    error_message = f"Cypher 쿼리 구문 오류: {error_message}"
                query_obj.success = False
                query_obj.error = error_message
                return (index, False, [], {
                    "query_index": index+1,
                    "success": False,
                    "result_count": 0,
                    "error": error_message
                })
        
        # 모든 쿼리를 병렬로 실행
        total_queries = len(pending_queries)
        tasks = [
            execute_single_query(query_obj, i, total_queries) 
            for i, query_obj in enumerate(pending_queries)
        ]
        results = await asyncio.gather(*tasks)
        
        # 결과 취합
        all_results = []
        execution_details = []
        total_result_count = 0
        
        for index, success, candidates, detail in results:
            all_results.extend(candidates)
            execution_details.append(detail)
            total_result_count += len(candidates)
        
        # 실행 결과 요약
        successful_queries = sum(1 for detail in execution_details if detail["success"])
        failed_queries = len(execution_details) - successful_queries
        pending_queries = sum(1 for q in cypher_query_objects if q.success is None)
        
        logger.info(f"쿼리 실행 결과: 성공 {successful_queries}, 실패 {failed_queries}, 대기 {pending_queries}")
        logger.info(f"전체 쿼리 실행 완료: {successful_queries}개 성공, {failed_queries}개 실패, 총 {total_result_count}개 결과")
        
        # 재생성 가능한 실패 쿼리 확인
        regeneratable_failures = [q for q in cypher_query_objects if q.success is False and q.reuse is not True]
        logger.info(f"재생성 가능한 실패 쿼리: {len(regeneratable_failures)}개")
        
        # 재생성 조건 상세 로깅
        if regeneratable_failures:
            logger.info("실패한 쿼리 상세:")
            for i, failed_query in enumerate(regeneratable_failures, 1):
                error_msg = failed_query.error or "Unknown error"
                logger.info(f"  {i}. 에러: {error_msg}")
                logger.info(f"     쿼리: {failed_query.cypher_query[:100]}...")
        else:
            logger.info("재생성이 필요한 실패 쿼리가 없습니다")
        
        # 기존 candidate_group과 병합 (재생성 시 이전 결과 유지)
        existing_candidates = getattr(state, 'candidate_group', []) or []
        merged_candidates = existing_candidates + all_results
        
        logger.info(f"Candidate 병합: 기존 {len(existing_candidates)}개 + 신규 {len(all_results)}개 = 총 {len(merged_candidates)}개")
        
        # cypher_query는 반환하지 않음 - 기존 상태의 객체들이 in-place로 업데이트됨
        return {
            "candidate_group": merged_candidates,  # 기존 결과 + 새 결과
        }
            
    except Exception as e:
        logger.error(f"Cypher 쿼리 실행 노드에서 오류 발생: {e}")
        # 에러 발생 시에도 기존 candidate_group 유지
        existing_candidates = getattr(state, 'candidate_group', []) or []
        logger.info(f"에러 발생으로 기존 candidate {len(existing_candidates)}개만 유지합니다")
        return {
            "candidate_group": existing_candidates,
        }


async def _attach_toc_summaries(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Section 결과에 연결된 TOC의 요약/본문/제목을 주입한다.
    - section_id가 결과 어디에 있든 재귀적으로 수집
    - TOC 정보를 조회해 동일한 dict 레벨에 toc_* 필드를 채워 넣음
    """
    section_ids: set[str] = set()

    def collect_section_ids(obj: Any):
        if isinstance(obj, dict):
            if "section_id" in obj and isinstance(obj["section_id"], str):
                section_ids.add(obj["section_id"])
            for v in obj.values():
                collect_section_ids(v)
        elif isinstance(obj, list):
            for v in obj:
                collect_section_ids(v)

    collect_section_ids(records)

    if not section_ids:
        return records

    query = """
    MATCH (toc:TOC)-[:HAS_SECTION]->(s:Section)
    WHERE s.section_id IN $section_ids
    RETURN s.section_id AS section_id,
           toc.toc_summary AS toc_summary,
           toc.core_text AS toc_core_text
    """

    try:
        toc_rows = await async_neo4j_db.execute_query_with_retry(
            query,
            {"section_ids": list(section_ids)},
            result_type="list",
        )
    except Exception as e:
        logger.warning(f"TOC 요약 조회 실패 - 섹션 {len(section_ids)}개: {e}")
        return records

    toc_map = {
        row.get("section_id"): {
            "toc_summary": row.get("toc_summary"),
            "toc_core_text": row.get("toc_core_text"),
        }
        for row in toc_rows or []
        if row.get("section_id")
    }

    def inject(obj: Any):
        if isinstance(obj, dict):
            sec_id = obj.get("section_id")
            if isinstance(sec_id, str) and sec_id in toc_map:
                toc_info = toc_map[sec_id]
                for k, v in toc_info.items():
                    if v and k not in obj:
                        obj[k] = v
            for v in obj.values():
                inject(v)
        elif isinstance(obj, list):
            for v in obj:
                inject(v)

    inject(records)
    return records


def _convert_to_records(result: List[Dict[str, Any]], cypher_query: Optional[str] = None) -> List[Candidate]:
    """
    Neo4j 쿼리 결과를 Candidate 형태로 변환
    이미지 데이터는 content에서 제외하고 image_base64 필드에 별도 저장
    핵심 정보만 추출하여 content 구성

    Args:
        result: Neo4j 쿼리 결과
        cypher_query: 이 결과를 얻는데 사용한 Cypher 쿼리

    Returns:
        List of Candidate
    """
    candidate_group = []
    for item in result:
        source = []
        image_list = []
        
        # source_id와 이미지 추출 (기존 방식 유지)
        _parse_dict(item, source_ids=source, image_list=image_list)
        
        # TOC 필드를 분리 추출 (content에는 포함하지 않음)
        toc_info = _extract_toc_fields(item)

        # 핵심 정보만 추출하여 content 구성 (TOC는 별도 필드로)
        content = _extract_essential_content(item, include_toc=False)
        
        # content가 비어있으면 기존 방식으로 폴백
        if not content.strip():
            content = _parse_dict(item, source_ids=[], image_list=[])

        candidate = Candidate(
            result=item,
            content=content,
            source=source,
            image_base64=image_list,
            is_image=len(image_list) > 0,
            cypher_query=cypher_query,
            toc_title=toc_info.get("toc_title"),
            toc_summary=toc_info.get("toc_summary"),
            toc_core_text=toc_info.get("toc_core_text"),
        )
        candidate_group.append(candidate)
    return candidate_group



def _get_document_filter_section() -> str:
    """선택된 문서에 대한 필터 쿼리 생성"""
    filter_query = """
MATCH (d:Document)
WHERE d.document_minio_url IN $selected_storage_paths 
WITH DISTINCT d
"""
    return filter_query


def _get_canonical_filter_section() -> str:
    filter_query = """
MATCH (c:Canonical)
WHERE ANY(url in c.document_minio_urls WHERE url IN $selected_storage_paths)
WITH DISTINCT c
"""
    return filter_query


def _cypher_query_example() -> str:
    query = f"""
MATCH (toc:TOC)-[:HAS_SECTION]->(s:Section)-[:HAS_FACT]->(f:Fact)-[:HAS_KEYWORD]->(k:Keyword)
WHERE k.value contains "한국"
RETURN f.fact
"""
    return query



# 이미지 관련 필드명 패턴
IMAGE_FIELD_PATTERNS = {'image_base64', 'base64', 'imagebase64', 'img_base64'}

# 제외할 필드명 패턴 (LLM에게 불필요한 시스템 메타데이터)
EXCLUDED_FIELD_PATTERNS = {
    # 타임스탬프 관련
    'canonical_extracted_at', 'extracted_at', 'created_at', 'updated_at', 'indexed_at',
    # 위치/인덱스 메타데이터
    'position_start_index', 'position_end_index', 
    'section_start_page', 'section_end_page',
    'first_ref', 'last_ref',
    # 기술적 메타데이터
    'bbox', 'loader', 'neo4jimportid', 'word_count',
    'toc_item_id', 'canonical_extracted',
    # 내부 참조
    'table_ids',
}

def _extract_toc_fields(data: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """TOC 관련 필드만 추출하여 별도 반환"""
    target_keys = {'toc_summary', 'toc_core_text'}
    collected: Dict[str, List[str]] = {}

    def find(obj: Any):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.lower() in target_keys and isinstance(v, str) and v.strip():
                    collected.setdefault(k.lower(), []).append(v.strip())
                if isinstance(v, (dict, list)):
                    find(v)
        elif isinstance(obj, list):
            for it in obj:
                find(it)

    find(data)

    def pick_first(key: str, limit: Optional[int] = None) -> Optional[str]:
        vals = collected.get(key)
        if not vals:
            return None
        val = vals[0]
        if limit and isinstance(val, str) and len(val) > limit:
            return val[:limit] + "..."
        return val

    return {
        "toc_summary": pick_first("toc_summary"),
        "toc_core_text": pick_first("toc_core_text", limit=600),
    }


def _extract_essential_content(data: Dict[str, Any], include_toc: bool = True) -> str:
    """
    Neo4j 결과에서 핵심 정보만 추출하여 깔끔한 텍스트로 변환
    
    핵심 정보: text, fact, document_id, section_summary
    불필요한 메타데이터(bbox, loader, neo4jImportId 등)는 제외
    """
    parts = []
    seen_texts = set()  # 중복 방지
    seen_facts = set()
    document_id = None
    
    def find_all_values(obj: Any, target_keys: set, results: dict) -> None:
        """재귀적으로 특정 키의 값들을 찾아서 수집"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_lower = key.lower()
                if key_lower in target_keys:
                    if key_lower not in results:
                        results[key_lower] = []
                    results[key_lower].append(value)
                # 중첩된 구조 탐색
                if isinstance(value, (dict, list)):
                    find_all_values(value, target_keys, results)
        elif isinstance(obj, list):
            for item in obj:
                find_all_values(item, target_keys, results)
    
    # 핵심 필드들 수집
    target_keys = {
        'text',
        'fact',
        'document_id',
        'section_summary',
        'subject',
        'predicate',
        'object',
    }
    if include_toc:
        target_keys = target_keys | {'toc_summary', 'toc_core_text'}
    collected = {}
    find_all_values(data, target_keys, collected)
    
    # document_id 추출 (첫 번째 것만)
    if 'document_id' in collected and collected['document_id']:
        doc_id = collected['document_id'][0]
        if doc_id and isinstance(doc_id, str):
            document_id = doc_id
            parts.append(f"[출처] {document_id}")
    
    # section_summary 추출 (비어있지 않은 것만)
    if 'section_summary' in collected:
        for summary in collected['section_summary']:
            if summary and isinstance(summary, str):
                summary = summary.strip()
                if summary and summary not in seen_texts and not summary.startswith('Prelude'):
                    parts.append(f"[요약] {summary}")
                    seen_texts.add(summary)
                    break  # 첫 번째 유효한 요약만

    # toc_summary / toc_core_text도 추가 (가능하면 요약이 우선)
    # include_toc=True일 때만 TOC 정보까지 content에 포함
    if include_toc:
        if 'toc_summary' in collected:
            for summary in collected['toc_summary']:
                if summary and isinstance(summary, str):
                    summary = summary.strip()
                    if summary and summary not in seen_texts:
                        parts.append(f"[TOC 요약] {summary}")
                        seen_texts.add(summary)
                        break
        if 'toc_core_text' in collected:
            for core_text in collected['toc_core_text']:
                if core_text and isinstance(core_text, str):
                    core_text = core_text.strip()
                    if core_text and core_text not in seen_texts:
                        # 길면 잘라서 제공
                        if len(core_text) > 600:
                            core_text = core_text[:600] + "..."
                        parts.append(f"[TOC 본문] {core_text}")
                        seen_texts.add(core_text)
                        break

    # text 필드 추출 (이미지 참조 제외)
    if 'text' in collected:
        for text_val in collected['text']:
            if text_val and isinstance(text_val, str):
                text_val = text_val.strip()
                # 이미지 참조만 있는 텍스트는 제외
                if text_val and not text_val.startswith('![') and text_val not in seen_texts:
                    # 너무 긴 텍스트는 앞부분만
                    if len(text_val) > 1000:
                        text_val = text_val[:1000] + "..."
                    parts.append(f"[텍스트] {text_val}")
                    seen_texts.add(text_val)
    
    # fact 필드 추출
    if 'fact' in collected:
        for fact_val in collected['fact']:
            if fact_val and isinstance(fact_val, str):
                fact_val = fact_val.strip()
                if fact_val and fact_val not in seen_facts:
                    parts.append(f"[사실] {fact_val}")
                    seen_facts.add(fact_val)
    
    return "\n".join(parts) if parts else ""


def _is_base64_image(value: Any) -> bool:
    """값이 base64 이미지 데이터인지 판별"""
    if not isinstance(value, str):
        return False
    # 최소 100자 이상이고 base64 패턴인지 확인
    if len(value) < 100:
        return False
    # base64 문자만 포함하는지 확인 (공백 제거 후)
    cleaned = value.strip().replace('\n', '').replace(' ', '')
    import re
    return bool(re.match(r'^[A-Za-z0-9+/=]+$', cleaned[:200]))


def _strip_minio_path_prefix(value: str) -> str:
    """minio: 경로에서 앞의 4개 세그먼트(minio:bucket/yyyy/mm/dd)를 제거하고 파일명만 반환"""
    if not value.startswith("minio:"):
        return value
    # minio:interview/2025/10/30/파일명.hwp -> 파일명.hwp
    parts = value.split("/")
    if len(parts) > 4:
        return "/".join(parts[4:])
    return value


def _parse_dict(
    dicts: Dict[str, Any], 
    indent_level: int = 0, 
    source_ids: Optional[List[Dict[str, str]]] = None,
    image_list: Optional[List[str]] = None
) -> str:
    """계층적으로 딕셔너리를 파싱하여 문자열로 변환하고 _id로 끝나는 값들을 수집, 이미지 데이터는 별도 추출"""
    if source_ids is None:
        source_ids = []
    if image_list is None:
        image_list = []

    indent = "  " * indent_level
    parts = []

    for key, value in dicts.items():
        # _id로 끝나는 키의 값을 source_ids에 딕셔너리 형태로 추가
        if key.endswith('_id') and isinstance(value, str):
            source_ids.append({"key": key, "value": value})

        key_lower = key.lower()

        # 제외할 필드 건너뛰기 (추출 시간 등 혼란을 줄 수 있는 메타데이터)
        if key_lower in EXCLUDED_FIELD_PATTERNS:
            continue

        # 이미지 필드 확인 및 추출 (content에서 제외)
        if key_lower in IMAGE_FIELD_PATTERNS or _is_base64_image(value):
            if isinstance(value, str) and len(value) > 100:
                image_list.append(value.strip())
                parts.append(f"{indent}{key}: [IMAGE_DATA]\n")
                continue

        if isinstance(value, dict):
            parts.append(f"{indent}{key}:\n")
            parts.append(_parse_dict(value, indent_level + 1, source_ids, image_list))
        elif isinstance(value, list):
            # 리스트가 단일 원소이고 간단한 값이면 한 줄로 출력
            if len(value) == 1 and not isinstance(value[0], (dict, list)):
                single_val = value[0]
                # minio: 경로에서 앞의 날짜 경로 제거
                if isinstance(single_val, str) and single_val.startswith("minio:"):
                    single_val = _strip_minio_path_prefix(single_val)
                parts.append(f"{indent}{key}: {single_val}\n")
            else:
                parts.append(f"{indent}{key}:\n")
                parts.append(_parse_list(value, indent_level + 1, source_ids, image_list))
        else:
            # minio: 경로에서 앞의 날짜 경로 제거
            if isinstance(value, str) and value.startswith("minio:"):
                value = _strip_minio_path_prefix(value)
            # 긴 텍스트는 줄바꿈 처리
            if isinstance(value, str) and len(value) > 80:
                parts.append(f"{indent}{key}:\n{indent}  {value}\n")
            else:
                parts.append(f"{indent}{key}: {value}\n")

    return "".join(parts)


def _parse_list(
    lists: List[Any], 
    indent_level: int = 0, 
    source_ids: Optional[List[Dict[str, str]]] = None,
    image_list: Optional[List[str]] = None
) -> str:
    """계층적으로 리스트를 파싱하여 문자열로 변환하고 _id로 끝나는 값들을 수집, 이미지 데이터는 별도 추출"""
    if source_ids is None:
        source_ids = []
    if image_list is None:
        image_list = []

    indent = "  " * indent_level
    parts = []

    for i, item in enumerate(lists):
        if isinstance(item, dict):
            parts.append(f"{indent}[{i}]:\n")
            parts.append(_parse_dict(item, indent_level + 1, source_ids, image_list))
        elif isinstance(item, list):
            parts.append(f"{indent}[{i}]:\n")
            parts.append(_parse_list(item, indent_level + 1, source_ids, image_list))
        else:
            # 리스트 내 문자열도 이미지인지 확인
            if _is_base64_image(item):
                image_list.append(item.strip())
                parts.append(f"{indent}- [IMAGE_DATA]\n")
            else:
                # minio: 경로에서 앞의 날짜 경로 제거
                if isinstance(item, str) and item.startswith("minio:"):
                    item = _strip_minio_path_prefix(item)
                parts.append(f"{indent}- {item}\n")

    return "".join(parts)
