"""
Unified Upload Utilities for Document Entities

Centralized upload logic for document ingestion pipeline:
- Documents, TOC, Sections, Facts
- Keywords (canonical-derived) and Attributes (table-derived)
- Images and Tables

Core features:
- Neo4j graph database updates
- Qdrant vector database updates
- Batch processing optimization
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data.schemas.types import CanonicalEntity, Table
from data.nodes.canonical_maker.utils import sanitize_property_key
from shared.db.neo4j.async_db import AsyncNeo4jDB
from shared.db.qdrant.async_db import QdrantAsyncDB
from shared.encoders.dense import OpenAIEncoder
from shared.utils.decorators import retry_async_on_rate_limit
from shared.config.config import embedding_config
logger = logging.getLogger(__name__)


# 라벨별 기본키 정의(도메인 키로 통일)
NODE_PK_MAP = {
    "Document": "document_id",
    "TOC": "toc_id",
    "Section": "section_id",
    "Fact": "fact_id",
    "Keyword": "keyword_id",
    "Table": "table_id",
    "TableColumn": "column_id",
    "TableRow": "row_id",
    "Measure": "measure_id",
}


# ================================================================
# Helper Functions
# ================================================================

def remove_empty_lists(d: Dict[str, Any]) -> Dict[str, Any]:
    """None과 빈 리스트 제거"""
    return {k: v for k, v in d.items() if v is not None and v != []}


def _sanitize_neo4j_value(value: Any) -> Any:
    """Neo4j 노드/관계 속성에 허용되는 타입으로 변환.
    허용: str, int, float, bool, list(위 원시 타입), 빈값(None)
    나머지(dict, 객체, list 내 비원시)는 JSON 문자열로 변환. 빈 dict(Map{})은 None으로 제거.
    """
    try:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            sanitized_list = []
            for elem in value:
                if isinstance(elem, (str, int, float, bool)):
                    sanitized_list.append(elem)
                else:
                    sanitized_list.append(json.dumps(elem, ensure_ascii=False))
            return sanitized_list
        if isinstance(value, dict):
            if not value:
                return None  # 빈 Map{}은 제거
            return json.dumps(value, ensure_ascii=False)
        # 기타 타입은 문자열화
        return str(value)
    except Exception:
        return None


def _sanitize_items_for_upload(label: str, items: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
    """노드 업로드 직전 필터링/정규화 로직을 공통화"""
    id_key = NODE_PK_MAP.get(label, "id")
    processed_items: List[Dict[str, Any]] = []
    
    logger.info(f"_sanitize_items_for_upload - items: {items}")

    for item in items:
        if id_key not in item:
            logger.error(f"❌ {label} 노드에 {id_key} 키 없음: {item}")
            continue
        clean_item = remove_empty_lists(item)
        safe_item = {}
        for k, v in clean_item.items():
            sanitized = _sanitize_neo4j_value(v)
            if sanitized is not None:
                safe_item[k] = sanitized
        if safe_item:
            processed_items.append(safe_item)

    return id_key, processed_items


def _save_toc_snapshot(doc_id: str, toc_items: List[Dict[str, Any]], stage: str = "before_neo4j") -> None:
    """TOC 업로드 전후 상태를 JSON으로 기록"""
    try:
        logger.info(f"_save_toc_snapshot 들어옴.")
        
        base_dir = Path(__file__).resolve().parents[4] / "logs" / "toc_snapshots"
        base_dir.mkdir(parents=True, exist_ok=True)

        safe_doc_id = str(doc_id or "unknown_doc").replace("/", "_")
        timestamp = datetime.utcnow().isoformat() + "Z"
        _, processed_items = _sanitize_items_for_upload("TOC", toc_items)

        snapshot = {
            "timestamp_utc": timestamp,
            "doc_id": doc_id,
            "stage": stage,
            "toc_count": len(toc_items),
            "processed_toc_count": len(processed_items),
            "toc_ids": [t.get("toc_id") for t in toc_items if isinstance(t, dict)],
            "processed_toc_ids": [t.get("toc_id") for t in processed_items if isinstance(t, dict)],
            "sample_titles": [t.get("title") for t in toc_items[:10] if isinstance(t, dict)],
            "toc_items": toc_items,
            "processed_items": processed_items,
        }

        snapshot_path = base_dir / f"{safe_doc_id}_{stage}_{int(time.time())}.json"
        snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
        logger.info(f"[TOC Debug] 스냅샷 저장: {snapshot_path}")
    except Exception as e:
        logger.warning(f"[TOC Debug] 스냅샷 저장 실패(stage={stage}, doc_id={doc_id}): {e}")


def _dump_toc_items(doc_id: str, toc_items: List[Dict[str, Any]], stage: str = "before_neo4j") -> None:
    """요청 시점 toc_items를 그대로 JSON으로 저장"""
    try:
        base_dir = Path(__file__).resolve().parents[4] / "logs" / "toc_snapshots"
        base_dir.mkdir(parents=True, exist_ok=True)

        safe_doc_id = str(doc_id or "unknown_doc").replace("/", "_")
        file_path = base_dir / f"{safe_doc_id}_{stage}_toc_items_{int(time.time())}.json"

        payload = {
            "doc_id": doc_id,
            "stage": stage,
            "count": len(toc_items),
            "toc_items": toc_items,
        }
        file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        # 로그로도 핵심 정보 출력 (개수, 첫 5개 제목/ID)
        sample = [
            {
                "toc_id": t.get("toc_id"),
                "title": t.get("title"),
                "level": t.get("level"),
            }
            for t in toc_items[:5]
            if isinstance(t, dict)
        ]
        logger.info(f"[TOC Debug] toc_items 덤프 저장: {file_path} (count={len(toc_items)}) sample={sample}")
    except Exception as e:
        logger.warning(f"[TOC Debug] toc_items 덤프 실패(stage={stage}, doc_id={doc_id}): {e}")


async def ensure_qdrant_collection(
    qdrant_db,
    collection_name: str,
    vector_size: Optional[int] = None,
    distance = None,
    force_recreate: bool = False,
):
    """Ensure Qdrant collection exists with correct vector dimension.
    
    Args:
        qdrant_db: Qdrant database connection
        collection_name: Name of collection to create
        vector_size: Explicit vector dimension (e.g., 1536 for canonical)
                     If None, uses config.vector_size (e.g., 4096 for documents)
        distance: Distance metric (optional, defaults to COSINE)
    """
    try:
        from qdrant_client.models import Distance, VectorParams, SparseVectorParams
        
        # Use explicit dimension if provided, otherwise use config
        dim = vector_size if vector_size is not None else qdrant_db.config.vector_size
        
        collections = await qdrant_db.client.get_collections()
        existing_names = [c.name for c in collections.collections]
        
        recreate_needed = force_recreate or (collection_name not in existing_names)

        if not recreate_needed and collection_name in existing_names:
            try:
                info = await qdrant_db.client.get_collection(collection_name)
                vectors_config = getattr(info, "vectors", None) or getattr(info, "vectors_config", None)
                existing_dim = None
                if isinstance(vectors_config, dict):
                    # multi-vector config: pick first dense vector size
                    for v in vectors_config.values():
                        if hasattr(v, "size"):
                            existing_dim = v.size
                            break
                elif hasattr(vectors_config, "size"):
                    existing_dim = vectors_config.size

                if existing_dim is not None and existing_dim != dim:
                    logger.warning(
                        f"[Qdrant] Dimension mismatch for '{collection_name}': existing={existing_dim}, desired={dim} → recreating"
                    )
                    recreate_needed = True
            except Exception as e:
                logger.warning(f"[Qdrant] Failed to inspect collection '{collection_name}', recreating: {e}")
                recreate_needed = True

        if recreate_needed:
            if distance is None:
                distance = Distance.COSINE
                
            dense_params = VectorParams(size=dim, distance=distance)
            sparse_params = {"sparse": SparseVectorParams()}
            
            await qdrant_db.client.recreate_collection(
                collection_name=collection_name,
                vectors_config={"dense": dense_params},
                sparse_vectors_config=sparse_params,
            )
            logger.info(f"[Qdrant] Created collection '{collection_name}' with dimension {dim}")
        else:
            logger.debug(f"[Qdrant] Collection '{collection_name}' already exists with matching dimension")
    except Exception as e:
        logger.error(f"[Qdrant] Collection setup failed: {e}")
        raise


# ================================================================
# Document Operations (Neo4j)
# ================================================================

async def upload_document_nodes(documents: List[Dict[str, Any]], db, batch_size: int = 500):
    """Upload Document nodes to Neo4j.
    
    Args:
        documents: List of document dictionaries
        db: Neo4j database connection
        batch_size: Batch size for upload
    """
    await upload_nodes("Document", documents, db, batch_size)


# ================================================================
# TOC Operations (Neo4j)
# ================================================================

async def upload_toc_nodes(toc_items: List[Dict[str, Any]], db, batch_size: int = 500):
    """Upload TOC nodes to Neo4j.
    
    Args:
        toc_items: List of TOC item dictionaries
        db: Neo4j database connection
        batch_size: Batch size for upload
    """
    await upload_nodes("TOC", toc_items, db, batch_size)


# ================================================================
# Section Operations (Neo4j + Qdrant)
# ================================================================

async def upload_section_nodes(sections: List[Dict[str, Any]], db, batch_size: int = 500):
    """Upload Section nodes to Neo4j.
    
    Args:
        sections: List of section dictionaries
        db: Neo4j database connection
        batch_size: Batch size for upload
    """
    await upload_nodes("Section", sections, db, batch_size)


@retry_async_on_rate_limit(max_retries=3, backoff=2.0)
async def upload_sections_to_qdrant(sections, doc_title, collection_name, encoder, db):
    """Upload sections to Qdrant with embeddings.
    
    Args:
        sections: List of section dictionaries
        doc_title: Document title
        collection_name: Qdrant collection name
        encoder: Text encoder for embeddings
        db: Qdrant database connection
        
    Returns:
        Number of sections uploaded
    """
    logger.info(f"[Qdrant] Section 임베딩 시작 ({len(sections)}개)")
    await ensure_qdrant_collection(
        db,
        collection_name,
        vector_size=embedding_config.embedding_dimensions,
        force_recreate=True,
    )
    
    # 텍스트가 있는 섹션만 필터링 (최적화)
    valid_sections = [s for s in sections if s.get("text")]
    if not valid_sections:
        logger.info("[Qdrant] Section 텍스트 없음, 스킵")
        return 0
    
    section_texts = [s["text"] for s in valid_sections]
    embeddings = await encoder.aembed_documents(section_texts)
    
    # 포인트 생성 (최적화: 한 번의 루프로)
    points = [
        {
            "id": str(uuid.uuid4()),
            "vector": {"dense": embedding},
            "payload": {
                "document_title": doc_title,
                "section_id": section["section_id"],
                "text": section["text"],
                "summary": section.get("summary", ""),
                "word_count": section.get("word_count", 0),
                "page": section.get("page", 0),
                "topic": section.get("topic", ""),
                "document_id": section.get("document_id", "")
            }
        }
        for section, embedding in zip(valid_sections, embeddings)
    ]
    
    if points:
        logger.info(f"[Qdrant] Section {len(points)}개 업로드 시도")
        await db.upload_points(collection_name=collection_name, points=points)
        logger.info(f"[Qdrant] Section {len(points)}개 업로드 완료")
    return len(points)


# ================================================================
# Fact Operations (Neo4j + Qdrant)
# ================================================================

async def upload_fact_nodes(facts: List[Dict[str, Any]], db, batch_size: int = 500):
    """Upload Fact nodes to Neo4j.
    
    Args:
        facts: List of fact dictionaries
        db: Neo4j database connection
        batch_size: Batch size for upload
    """
    await upload_nodes("Fact", facts, db, batch_size)


@retry_async_on_rate_limit(max_retries=3, backoff=2.0)
async def upload_facts_to_qdrant(facts, doc_title, collection_name, encoder, db):
    """Upload facts to Qdrant with embeddings.
    
    Args:
        facts: List of fact dictionaries
        doc_title: Document title
        collection_name: Qdrant collection name
        encoder: Text encoder for embeddings
        db: Qdrant database connection
        
    Returns:
        Number of facts uploaded
    """
    logger.info(f"[Qdrant] Fact 임베딩 시작 ({len(facts)}개)")
    await ensure_qdrant_collection(
        db,
        collection_name,
        vector_size=embedding_config.embedding_dimensions,
        force_recreate=True,
    )

    # 텍스트가 있는 팩트만 필터링 (최적화)
    valid_facts = [f for f in facts if f.get("fact")]
    if not valid_facts:
        logger.info("[Qdrant] Fact 텍스트 없음, 스킵")
        return 0
    
    fact_texts = [f["fact"] for f in valid_facts]
    embeddings = await encoder.aembed_documents(fact_texts)
    
    # 포인트 생성 (최적화: 한 번의 루프로)
    points = [
        {
            "id": str(uuid.uuid4()),
            "vector": {"dense": embedding},
            "payload": {
                "document_title": doc_title,
                "fact_id": fact["fact_id"],
                "fact": fact["fact"],
                "confidence": fact.get("confidence", 1.0),
                "semantic_chunk_id": fact.get("semantic_chunk_id", ""),
                "section_id": fact.get("section_id", ""),
                "document_id": fact.get("document_id", "")
            }
        }
        for fact, embedding in zip(valid_facts, embeddings)
    ]
    
    if points:
        logger.info(f"[Qdrant] Fact {len(points)}개 업로드 시도")
        await db.upload_points(collection_name=collection_name, points=points)
        logger.info(f"[Qdrant] Fact {len(points)}개 업로드 완료")
    return len(points)


# ================================================================
# Keyword Operations (Neo4j + Qdrant) - Canonical-derived
# ================================================================

def extract_keywords_from_entities(
    entities: List[CanonicalEntity],
    document_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Extract keywords from entity properties.
    Each keyword gets a unique keyword_id: {section_id}_kw_{index}

    CRITICAL: After merge_duplicate_canonicals, entities may have multiple section_ids.
    We must create keywords for ALL section_ids to maintain consistency.

    Args:
        entities: List of canonical entities
        document_id: Document ID for deletion filtering (optional)

    Returns:
        List of keyword dictionaries with keyword_id
    """
    keywords: List[Dict[str, Any]] = []
    keyword_counter: Dict[str, int] = {}  # Track keyword count per section

    for entity in entities:
        if not entity.section_ids:
            logger.warning(f"Entity {entity.canonical_id} has no section_ids, skipping keywords")
            continue

        # CRITICAL: Create keywords for ALL section_ids (entity may appear in multiple sections)
        for section_id in entity.section_ids:
            # Initialize counter for this section if needed
            if section_id not in keyword_counter:
                keyword_counter[section_id] = 0

            # Create keywords for each property in this section
            for prop in entity.properties:
                keyword_counter[section_id] += 1
                keyword_id = f"{section_id}_kw_{keyword_counter[section_id]}"

                keyword_data = {
                    "keyword_id": keyword_id,
                    "key": sanitize_property_key(prop.key),
                    "value": prop.value,
                    "section_id": section_id,  # Store as string for Keyword node
                    "canonical_id": entity.canonical_id,
                }
                # Add document_id if provided
                if document_id:
                    keyword_data["document_id"] = document_id
                keywords.append(keyword_data)

    logger.info(f"Extracted {len(keywords)} keywords from {len(entities)} entities")
    return keywords


async def upload_keywords_to_neo4j(
    neo4j_db: AsyncNeo4jDB,
    keywords: List[Dict[str, Any]]
) -> Tuple[int, int]:
    """
    Upload keywords to Neo4j and link to sections and canonicals.
    
    Creates:
    - Keyword nodes
    - Section -[:HAS_KEYWORD]-> Keyword relationships
    - Keyword -[:REPRESENTED_BY]-> Canonical relationships
    
    Args:
        neo4j_db: Neo4j database connection
        keywords: List of keyword dictionaries with keyword_id
        
    Returns:
        Tuple of (keywords_created, links_created)
    """
    if not keywords:
        logger.info("[Keywords] No keywords to upload")
        return 0, 0
    
    logger.info(f"[Keywords] Uploading {len(keywords)} keywords with links")
    
    # Step 1: Create Keyword nodes and link to Sections or Tables
    query = """
    UNWIND $keywords AS kw
    MERGE (k:Keyword {keyword_id: kw.keyword_id})
    ON CREATE SET
        k.key = kw.key,
        k.value = kw.value,
        k.version = kw.version,
        k.section_id = kw.section_id,
        k.document_id = kw.document_id,
        k.created_at = datetime()
    ON MATCH SET
        k.updated_at = datetime()
    WITH k, kw
    OPTIONAL MATCH (s:Section {section_id: kw.section_id})
    OPTIONAL MATCH (t:Table {id: kw.section_id})
    WITH COALESCE(s, t) AS source_node, k, kw
    WHERE source_node IS NOT NULL
    MERGE (source_node)-[:HAS_KEYWORD]->(k)
    RETURN count(DISTINCT k) AS keyword_count
    """
    
    result = await neo4j_db.execute_query(
        query,
        keywords=keywords,
        result_type="list"
    )
    
    keyword_count = result[0]["keyword_count"] if result else 0
    logger.info(f"[Keywords] Created/updated {keyword_count} keywords")
    
    # Step 2: Link Keywords to Canonicals
    link_query = """
    UNWIND $keywords AS kw
    MATCH (k:Keyword {keyword_id: kw.keyword_id})
    MATCH (c:Canonical {canonical_id: kw.canonical_id})
    MERGE (k)-[:REPRESENTED_BY]->(c)
    RETURN count(*) AS link_count
    """
    
    link_result = await neo4j_db.execute_query(
        link_query,
        keywords=keywords,
        result_type="list"
    )
    
    link_count = link_result[0]["link_count"] if link_result else 0
    logger.info(f"[Keywords] Created {link_count} REPRESENTED_BY relationships")
    
    return keyword_count, link_count


async def upload_keywords_to_qdrant(
    qdrant_db,
    encoder,
    keywords: List[Dict[str, Any]],
    collection_name: str,
    embed_batch_size: int = 1000,
    document_id: Optional[str] = None
) -> int:
    """
    Upload keywords to Qdrant with embeddings.

    Each keyword is embedded based on its key-value pair.
    Point ID is derived from keyword_id for deterministic updates.

    Args:
        qdrant_db: Qdrant database connection
        encoder: Text encoder
        keywords: List of keyword dictionaries with keyword_id, key, value, etc.
        collection_name: Qdrant collection name (e.g., "neo4j_keywords")
        embed_batch_size: Number of keywords to embed per batch (default: 1000)
        document_id: Document ID for deletion filtering (optional)

    Returns:
        Number of keywords uploaded
    """
    if not keywords:
        logger.info("[Keywords] No keywords to upload to Qdrant")
        return 0

    logger.info(f"[Keywords] Uploading {len(keywords)} keywords to Qdrant collection: {collection_name}, document_id={document_id}")
    
    # Build embedding texts (key: value format)
    keyword_texts = []
    for kw in keywords:
        text = f"{kw['key']}: {kw['value']}"
        keyword_texts.append(text)
    
    # Process embeddings in batches
    points = []
    total_batches = (len(keyword_texts) + embed_batch_size - 1) // embed_batch_size
    
    for batch_idx in range(0, len(keyword_texts), embed_batch_size):
        chunk_texts = keyword_texts[batch_idx:batch_idx + embed_batch_size]
        chunk_keywords = keywords[batch_idx:batch_idx + embed_batch_size]
        batch_num = (batch_idx // embed_batch_size) + 1
        
        logger.info(
            f"[Keywords] Processing embedding batch {batch_num}/{total_batches} "
            f"({len(chunk_texts)} keywords)"
        )
        
        # Batch embedding API call
        embeddings = await encoder.aembed_documents(chunk_texts)
        
        # Build points for this batch
        for kw, embedding in zip(chunk_keywords, embeddings):
            # Generate deterministic point ID from keyword_id
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, kw['keyword_id']))
            
            # Build payload
            payload = {
                "keyword_id": kw['keyword_id'],
                "key": kw['key'],
                "value": kw['value'],
                "section_id": kw['section_id'],
                "canonical_id": kw['canonical_id'],
                "source_type": "canonical"  # Distinguish from table attributes
            }
            # Add document_id for deletion filtering (if provided)
            if document_id:
                payload["document_id"] = document_id
            
            # Build point
            points.append({
                "id": point_id,
                "vector": {"dense": embedding},  # Named vector format
                "payload": payload
            })
    
    # Upload all points to Qdrant
    logger.info(f"[Keywords] Uploading {len(points)} keyword points to Qdrant")
    await qdrant_db.upload_points(
        collection_name=collection_name,
        points=points
    )
    
    logger.info(f"[Keywords] Successfully uploaded {len(points)} keywords to Qdrant")
    return len(points)


# ================================================================
# Attribute Operations (Neo4j + Qdrant) - Table-derived
# ================================================================

def extract_attributes_from_tables(tables: List[Table]) -> List[Dict[str, Any]]:
    """Extract attributes from table rows for keyword indexing.
    
    Similar to extract_keywords_from_entities but for table attributes.
    Each attribute gets: attribute_id, column_name, value, table_id, row_id.
    
    Args:
        tables: List of Table objects
        
    Returns:
        List of attribute dictionaries
    """
    attributes: List[Dict[str, Any]] = []
    attribute_counter: Dict[str, int] = {}  # Track attribute count per table
    
    for table in tables:
        table_id = table.id
        if not table_id:
            logger.warning(f"Table has no id, skipping attributes")
            continue
        
        # Initialize counter for this table if needed
        if table_id not in attribute_counter:
            attribute_counter[table_id] = 0
        
        # Extract attributes from each row
        for row in table.rows:
            row_id = row.id
            if not row_id:
                continue
            
            for attr in row.attributes:
                attribute_counter[table_id] += 1
                attribute_id = f"{table_id}_attr_{attribute_counter[table_id]}"
                
                attributes.append({
                    "attribute_id": attribute_id,
                    "column_name": attr.column_name,
                    "value": attr.value,
                    "table_id": table_id,
                    "row_id": row_id,
                    "row_index": row.row_index,
                })
    
    logger.info(f"Extracted {len(attributes)} attributes from {len(tables)} tables")
    return attributes


async def upload_attributes_to_neo4j(
    neo4j_db: AsyncNeo4jDB,
    attributes: List[Dict[str, Any]]
) -> Tuple[int, int]:
    """Upload table attributes as Keyword nodes with HAS_KEYWORD from Row.
    
    Different from canonical keywords:
    - Links from Row → Keyword (not Section → Keyword)
    - No REPRESENTED_BY to Canonical
    - Includes row_id, table_id in payload
    
    Args:
        neo4j_db: Neo4j database connection
        attributes: List of attribute dictionaries
        
    Returns:
        Tuple of (attributes_created, links_created)
    """
    if not attributes:
        logger.info("[Attributes] No attributes to upload")
        return 0, 0
    
    logger.info(f"[Attributes] Uploading {len(attributes)} table attributes as keywords")
    
    # Step 1: Create Keyword nodes for attributes
    query = """
    UNWIND $attributes AS attr
    MERGE (k:Keyword {keyword_id: attr.attribute_id})
    ON CREATE SET 
        k.key = attr.column_name,
        k.value = attr.value,
        k.version = attr.version,
        k.table_id = attr.table_id,
        k.row_id = attr.row_id,
        k.row_index = attr.row_index,
        k.created_at = datetime()
    ON MATCH SET 
        k.updated_at = datetime()
    RETURN count(DISTINCT k) AS attribute_count
    """
    
    result = await neo4j_db.execute_query(
        query,
        attributes=attributes,
        result_type="list"
    )
    
    attribute_count = result[0]["attribute_count"] if result else 0
    logger.info(f"[Attributes] Created/updated {attribute_count} attribute keywords")
    
    # Step 2: Link Attributes to Rows
    link_query = """
    UNWIND $attributes AS attr
    MATCH (k:Keyword {keyword_id: attr.attribute_id})
    MATCH (r:Row {id: attr.row_id})
    MERGE (r)-[:HAS_KEYWORD]->(k)
    RETURN count(*) AS link_count
    """
    
    link_result = await neo4j_db.execute_query(
        link_query,
        attributes=attributes,
        result_type="list"
    )
    
    link_count = link_result[0]["link_count"] if link_result else 0
    logger.info(f"[Attributes] Created {link_count} Row-Keyword relationships")
    
    return attribute_count, link_count


async def upload_attributes_to_qdrant(
    qdrant_db,
    encoder,
    attributes: List[Dict[str, Any]],
    collection_name: str,
    embed_batch_size: int = 1000
) -> int:
    """Upload table attributes to shared keywords collection with embeddings.
    
    Both table attributes and canonical keywords use:
    - Same collection: {database}-keywords
    - Same encoder: encoder
    - Different source_type in payload for filtering ("table" vs "canonical")
    
    Payload includes:
    - table_id and row_id for table context
    - column_name as key
    - source_type: "table" for filtering
    
    Args:
        qdrant_db: Qdrant database connection
        encoder: Text encoder
        attributes: List of attribute dictionaries
        collection_name: Qdrant collection name (same as keywords)
        embed_batch_size: Number of attributes to embed per batch
        
    Returns:
        Number of attributes uploaded
    """
    if not attributes:
        logger.info("[Attributes] No attributes to upload to Qdrant")
        return 0

    logger.info(f"[Attributes] Uploading {len(attributes)} table attributes to Qdrant collection: {collection_name} (using {encoder.model_name})")
    
    # Build embedding texts (column: value format)
    attribute_texts = []
    for attr in attributes:
        text = f"{attr['column_name']}: {attr['value']}"
        attribute_texts.append(text)
    
    # Process embeddings in batches
    points = []
    total_batches = (len(attribute_texts) + embed_batch_size - 1) // embed_batch_size
    
    for batch_idx in range(0, len(attribute_texts), embed_batch_size):
        chunk_texts = attribute_texts[batch_idx:batch_idx + embed_batch_size]
        chunk_attributes = attributes[batch_idx:batch_idx + embed_batch_size]
        batch_num = (batch_idx // embed_batch_size) + 1
        
        logger.info(
            f"[Attributes] Processing embedding batch {batch_num}/{total_batches} "
            f"({len(chunk_texts)} attributes)"
        )
        
        # Batch embedding API call
        embeddings = await encoder.aembed_documents(chunk_texts)
        
        # Build points for this batch
        for attr, embedding in zip(chunk_attributes, embeddings):
            # Generate deterministic point ID from attribute_id
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, attr['attribute_id']))
            
            # Build payload
            payload = {
                "keyword_id": attr['attribute_id'],
                "key": attr['column_name'],
                "value": attr['value'],
                "table_id": attr['table_id'],
                "row_id": attr['row_id'],
                "row_index": attr.get('row_index', 0),
                "version": attr.get('version', 3),
                "source_type": "table"  # Distinguish from canonical keywords
            }
            
            # Build point
            points.append({
                "id": point_id,
                "vector": {"dense": embedding},
                "payload": payload
            })
    
    # Upload all points to Qdrant
    logger.info(f"[Attributes] Uploading {len(points)} attribute points to Qdrant")
    await qdrant_db.upload_points(
        collection_name=collection_name,
        points=points
    )
    
    logger.info(f"[Attributes] Successfully uploaded {len(points)} attributes to Qdrant")
    return len(points)


# ================================================================
# Folder Operations (Neo4j) - Document Path Hierarchy
# ================================================================

def parse_folder_hierarchy_from_minio_url(minio_url: str) -> List[Dict[str, Any]]:
    """Parse minio URL into folder hierarchy.

    Given: minio:interview/2025/10/30/거제시 데이터포털_빅데이터 분석/관광인구정의(1).hwp
    Returns:
        [
            {"name": "interview", "path": "interview"},
            {"name": "2025", "path": "interview/2025"},
            {"name": "10", "path": "interview/2025/10"},
            {"name": "30", "path": "interview/2025/10/30"},
            {"name": "거제시 데이터포털_빅데이터 분석", "path": "interview/2025/10/30/거제시 데이터포털_빅데이터 분석"}
        ]

    Args:
        minio_url: MinIO URL (e.g., "minio:path/to/folder/file.pdf")

    Returns:
        List of folder dictionaries with name, path
    """
    if not minio_url:
        return []

    # Remove minio: prefix if present
    path = minio_url
    if path.startswith("minio:"):
        path = path[6:]

    # Split by / and remove empty parts
    parts = [p for p in path.split("/") if p]

    if len(parts) <= 1:
        # No folders, just a file at root
        return []

    # Exclude the last part (file name)
    folder_parts = parts[:-1]

    folders = []
    for i, name in enumerate(folder_parts):
        folder_path = "/".join(folder_parts[:i+1])
        folders.append({
            "name": name,
            "path": folder_path
        })

    return folders


def get_minio_bucket_name() -> str:
    """Get MinIO bucket name from environment or config.

    Returns:
        Bucket name (default: gorag-files)
    """
    import os
    return os.getenv("MINIO_BUCKET_NAME", "gorag-files")


async def upload_folder_hierarchy(
    db,
    minio_url: str,
    document_id: str
) -> Tuple[int, int]:
    """Upload folder hierarchy from minio URL to Neo4j.

    Creates:
    - Bucket node (db: minio, name: bucket_name)
    - Folder nodes with name, path
    - HAS_FOLDER relationship from Bucket to root folder
    - HAS_SUBFOLDER relationships between parent-child folders
    - CONTAINS_DOCUMENT relationship from deepest folder to document

    Args:
        db: Neo4j database connection
        minio_url: MinIO URL of the document
        document_id: Document ID to link to

    Returns:
        Tuple of (folders_created, relationships_created)
    """
    if not minio_url:
        logger.info("[Folders] No minio_url provided, skipping folder hierarchy")
        return 0, 0

    folders = parse_folder_hierarchy_from_minio_url(minio_url)
    if not folders:
        logger.info("[Folders] No folder hierarchy found in minio_url")
        return 0, 0

    bucket_name = get_minio_bucket_name()
    logger.info(f"[Folders] Creating folder hierarchy: {len(folders)} folders from {minio_url} (bucket: {bucket_name})")

    # Step 0: Create/Merge Bucket node
    bucket_query = """
    MERGE (b:Bucket {name: $bucket_name})
    ON CREATE SET
        b.db = 'minio',
        b.created_at = datetime()
    ON MATCH SET
        b.updated_at = datetime()
    RETURN count(b) AS bucket_count
    """

    bucket_result = await db.execute_query(
        bucket_query,
        bucket_name=bucket_name,
        result_type="list"
    )
    bucket_count = bucket_result[0]["bucket_count"] if bucket_result else 0
    logger.info(f"[Folders] Bucket node created/updated: {bucket_name}")

    # Step 1: Create/Merge all Folder nodes
    folder_query = """
    UNWIND $folders AS folder
    MERGE (f:Folder {path: folder.path})
    ON CREATE SET
        f.name = folder.name,
        f.created_at = datetime()
    ON MATCH SET
        f.updated_at = datetime()
    RETURN count(f) AS folder_count
    """

    result = await db.execute_query(
        folder_query,
        folders=folders,
        result_type="list"
    )
    folder_count = result[0]["folder_count"] if result else 0
    logger.info(f"[Folders] Created/updated {folder_count} folder nodes")

    rel_count = 0

    # Step 2: Link Bucket to root folder via HAS_FOLDER
    if folders:
        root_folder_path = folders[0]["path"]
        bucket_folder_query = """
        MATCH (b:Bucket {name: $bucket_name})
        MATCH (f:Folder {path: $folder_path})
        MERGE (b)-[:HAS_FOLDER]->(f)
        RETURN count(*) AS link_count
        """

        bucket_folder_result = await db.execute_query(
            bucket_folder_query,
            bucket_name=bucket_name,
            folder_path=root_folder_path,
            result_type="list"
        )
        bucket_folder_count = bucket_folder_result[0]["link_count"] if bucket_folder_result else 0
        rel_count += bucket_folder_count
        logger.info(f"[Folders] Created {bucket_folder_count} HAS_FOLDER relationship: {bucket_name} -> {root_folder_path}")

    # Step 3: Create HAS_SUBFOLDER relationships between parent-child folders
    if len(folders) > 1:
        subfolder_pairs = []
        for i in range(len(folders) - 1):
            subfolder_pairs.append({
                "parent_path": folders[i]["path"],
                "child_path": folders[i + 1]["path"]
            })

        subfolder_query = """
        UNWIND $pairs AS pair
        MATCH (parent:Folder {path: pair.parent_path})
        MATCH (child:Folder {path: pair.child_path})
        MERGE (parent)-[:HAS_SUBFOLDER]->(child)
        RETURN count(*) AS rel_count
        """

        subfolder_result = await db.execute_query(
            subfolder_query,
            pairs=subfolder_pairs,
            result_type="list"
        )
        subfolder_rel_count = subfolder_result[0]["rel_count"] if subfolder_result else 0
        rel_count += subfolder_rel_count
        logger.info(f"[Folders] Created {subfolder_rel_count} HAS_SUBFOLDER relationships")

    # Step 4: Link deepest folder to document via HAS_DOCUMENT
    if folders:
        deepest_folder_path = folders[-1]["path"]
        doc_link_query = """
        MATCH (f:Folder {path: $folder_path})
        MATCH (d:Document {document_id: $document_id})
        MERGE (f)-[:HAS_DOCUMENT]->(d)
        RETURN count(*) AS link_count
        """

        doc_link_result = await db.execute_query(
            doc_link_query,
            folder_path=deepest_folder_path,
            document_id=document_id,
            result_type="list"
        )
        doc_link_count = doc_link_result[0]["link_count"] if doc_link_result else 0
        rel_count += doc_link_count
        logger.info(f"[Folders] Created {doc_link_count} CONTAINS_DOCUMENT relationship: {deepest_folder_path} -> {document_id}")

    return folder_count + bucket_count, rel_count


# ================================================================
# Relationship Operations (Generic)
# ================================================================

async def upload_nodes(label: str, items: List[Dict[str, Any]], db, batch_size: int = 500):
    """배치 처리로 노드를 업로드하여 성능 최적화"""
    id_key, processed_items = _sanitize_items_for_upload(label, items)
    
    if not processed_items:
        logger.warning(f"⚠️ {label} 업로드할 항목 없음")
        return
    
    success_count = 0
    error_count = 0
    
    # TOC인 경우 업로드 전에 모든 TOC 항목 출력
    if label == "TOC" or label == "section_header":
        logger.info(f"📋 [TOC 상세] 업로드할 TOC 목록 ({len(processed_items)}개):")
        for idx, toc_item in enumerate(processed_items, 1):
            title = toc_item.get("title", "N/A")
            level = toc_item.get("level", "N/A")
            logger.info(f"  [{idx}] title={title}, level={level}")
    
    # 배치 단위로 처리
    for i in range(0, len(processed_items), batch_size):
        logger.info(f"for loop 들어왔다고.")
        batch = processed_items[i:i + batch_size]
        
        # UNWIND를 사용한 배치 쿼리 - 실제 처리된 노드 수 반환
        query = f"""
        UNWIND $items AS item
        MERGE (n:{label} {{{id_key}: item.{id_key}}})
        SET n += item
        RETURN n.{id_key} as processed_id
        """
        
        try:
            logger.info(f"✅✅✅✅✅✅")
            result = await db.execute_query(query, items=batch, result_type="list")
            # 실제 처리된 노드 수 = 반환된 결과 행 수
            actual_processed = len(result) if result else len(batch)
            success_count += actual_processed
            batch_num = (i // batch_size) + 1
            total_batches = (len(processed_items) + batch_size - 1) // batch_size
            if actual_processed != len(batch):
                logger.warning(f"⚠️ {label} 배치 {batch_num}/{total_batches}: {len(batch)}개 시도, {actual_processed}개 성공 (일부 실패 가능)")
            else:
                logger.info(f"✅ {label} 배치 {batch_num}/{total_batches}: {actual_processed}개 성공")
        except Exception as e:
            error_count += len(batch)
            logger.error(f"❌ {label} 배치 업로드 실패: {e}")
            # 배치 실패 시 개별 처리로 폴백
            for item in batch:
                try:
                    single_query = f"MERGE (n:{label} {{{id_key}: $item.{id_key}}}) SET n += $item RETURN n.{id_key} as processed_id"
                    result = await db.execute_query(single_query, item=item, result_type="list")
                    if result and len(result) > 0:
                        success_count += 1
                        error_count -= 1
                    else:
                        logger.warning(f"⚠️ {label} 개별 업로드 결과 없음 ({id_key}={item.get(id_key)})")
                except Exception as single_error:
                    logger.error(f"❌ {label} 개별 업로드 실패 ({id_key}={item.get(id_key)}): {single_error}")
    
    logger.info(f"📊 {label} 업로드 완료: 성공 {success_count}개, 실패 {error_count}개")


async def upload_relationships(db, grouped_rels):
    """Upload relationships to Neo4j in batches.
    
    Args:
        db: Neo4j database connection
        grouped_rels: Dictionary of relationships grouped by (from_label, rel_type, to_label)
    """
    total_rels = 0
    start = time.time()
    for (from_label, rel_type, to_label), pairs in grouped_rels.items():
        filtered_pairs = [
            pair for pair in pairs
            if pair["from_id"] is not None and pair["to_id"] is not None
        ]
        if not filtered_pairs:
            continue
        # 라벨별 기본키 매핑
        pk_map = {
            "Document": "document_id",
            "TOC": "toc_id",
            "Section": "section_id",
            "Fact": "fact_id",
            "Keyword": "keyword_id",
            "Table": "table_id",
            "TableColumn": "column_id",
            "TableRow": "row_id",
            "Measure": "measure_id",
        }
        from_field = pk_map.get(from_label, "id")
        to_field = pk_map.get(to_label, "id")
        query = f"""
        UNWIND $pairs AS pair
        MATCH (a:{from_label} {{{from_field}: pair.from_id}}), (b:{to_label} {{{to_field}: pair.to_id}})
        MERGE (a)-[:{rel_type}]->(b)
        """
        try:
            await db.execute_query(query, pairs=filtered_pairs)
            total_rels += len(filtered_pairs)
            logger.info(f"✅ 관계 {rel_type} {len(filtered_pairs)}건 업로드 완료")

            # Extra logging for TOC-Section relationships
            if rel_type == "HAS_SECTION":
                logger.info(f"[Debug] TOC-Section relationships created: {filtered_pairs[:3]}...")  # Log first 3

        except Exception as e:
            logger.error(f"❌ 관계 업로드 실패 ({rel_type}): {e}")
            if rel_type == "HAS_SECTION":
                logger.error(f"[Debug] Failed TOC-Section pairs: {filtered_pairs[:3]}...")  # Log first 3 failed
    logger.info(f"🎯 총 {total_rels}개 관계 업로드 완료 — 소요 시간: {time.time() - start:.2f}초")


# ================================================================
# High-Level Orchestration
# ================================================================

async def upload_to_neo4j(document_package: dict, db=None) -> str:
    """Neo4j 노드/관계 업로드 
    - Table 업로드는 개선된 스키마 경로에서 수행되므로, 중복/임시 노드 생성을 방지하기 위해 여기서는 제외합니다.
    
    Args:
        document_package: Packaged document data
        db: Neo4j database connection
        
    Returns:
        Final document_id (may be reconciled with existing document)
    """
    # Import regex pattern from text_upload_node
    import re
    _SECTION_PATTERN = re.compile(r"^(.+)_sec_(\d+(?:_\d+)*)$")
    
    doc_id = document_package["doc_id"]
    toc_items = document_package["toc_items"]
    sections = document_package["sections"]
    facts = document_package["facts"]
    keywords = document_package["keywords"]
    # tables는 문서 업로드 단계에서 사용하지 않음 (개선된 스키마 업로드에서 처리)
    tables = []
    section_relations = document_package["section_relations"]

    # --- 사전 정합: 테이블 업로더가 먼저 만든 Document를 그대로 재사용 ---
    try:
        document_title = (document_package.get("document") or {}).get("document_title")
        if db is not None and document_title:
            # 하나의 쿼리로 ID와 제목 모두 확인 (최적화)
            try:
                rows = await db.execute_query_with_retry(
                    """
                    MATCH (d:Document)
                    WHERE d.document_id = $doc_id OR d.document_title = $doc_title
                    RETURN d.document_id AS id
                    ORDER BY d.document_id = $doc_id DESC
                    LIMIT 1
                    """,
                    {"doc_id": doc_id, "doc_title": document_title},
                    result_type="list",
                )
            except Exception:
                rows = []
            if rows:
                existing_id = rows[0].get("id")
                if existing_id and existing_id != doc_id:
                    logger.info(
                        f"[Upload] Existing Document detected (by id/title). Reusing document_id: {existing_id} (was {doc_id})"
                    )
                    doc_id = existing_id
                    # 패키지 내 id 및 문서 객체의 id를 정합
                    document_package["doc_id"] = doc_id
                    if "document" in document_package and isinstance(document_package["document"], dict):
                        document_package["document"]["document_id"] = doc_id
    except Exception as e:
        logger.warning(f"[Upload] Document id/title reconciliation skipped due to error: {e}")

    # --- 보강: Document 노드의 document_id 보장 ---
    try:
        if "document" not in document_package or not isinstance(document_package["document"], dict):
            document_package["document"] = {"document_id": doc_id}
        else:
            document_package["document"].setdefault("document_id", doc_id)
    except Exception:
        # 안전하게 기본 Document 구성
        document_package["document"] = {"document_id": doc_id}

    # --- 보강: TOC/Section ID 정합성 보장 ---
    try:
        # TOC: toc_id가 없으면 순번 기반으로 생성
        fixed_toc_items = []
        for idx, toc in enumerate(toc_items or []):
            if not isinstance(toc, dict):
                continue
            if not toc.get("toc_id"):
                toc["toc_id"] = f"{doc_id}_toc_{idx}"
            fixed_toc_items.append(toc)
        toc_items = fixed_toc_items

        # Section: toc_item_id가 없으면 section_id에서 유추(doc_sec_N_M 패턴)
        # Handles both flat (doc_sec_27_0 → doc_toc_27) and hierarchical (doc_sec_1_2_3_0 → doc_toc_1_2_3)
        fixed_sections = []
        for sec in sections or []:
            if not isinstance(sec, dict):
                fixed_sections.append(sec)
                continue
            sid = sec.get("section_id")
            if not sec.get("toc_item_id") and isinstance(sid, str) and "_sec_" in sid:
                try:
                    # Extract TOC ID from section_id using regex
                    # Examples:
                    #   "doc_sec_27_0" → "doc_toc_27"
                    #   "doc_sec_1_2_3_0" → "doc_toc_1_2_3"
                    match = _SECTION_PATTERN.match(sid)
                    if match:
                        doc_prefix = match.group(1)
                        toc_hierarchical = match.group(2)
                        # Remove last split index: "1_2_3_0" → "1_2_3"
                        toc_parts = toc_hierarchical.split("_")
                        if len(toc_parts) > 1:
                            toc_num = "_".join(toc_parts[:-1])
                        else:
                            toc_num = toc_hierarchical
                        sec["toc_item_id"] = f"{doc_prefix}_toc_{toc_num}"
                except Exception:
                    pass
            fixed_sections.append(sec)
        sections = fixed_sections
        # 반영
        document_package["toc_items"] = toc_items
        document_package["sections"] = sections
    except Exception:
        pass

    # 진단 로깅: 비어있을 때 경고
    if not toc_items:
        logger.warning(f"[Upload] TOC 비어있음 (doc_id={doc_id})")
    if not sections:
        logger.warning(f"[Upload] Sections 비어있음 (doc_id={doc_id})")
    if not facts:
        logger.warning(f"[Upload] Facts 비어있음 (doc_id={doc_id})")
    # Keywords check removed - handled by canonical extraction pipeline
 
    # 데이터 구조 상세 로깅 (Table 개수는 0으로 고정)
    logger.info(f"📊 [Upload] 데이터 구조: TOC={len(toc_items)}, Sections={len(sections)}, Facts={len(facts)}, Tables=0, Section Relations={len(section_relations)}")

    # 요청: 이 시점 toc_items를 별도 JSON으로 저장
    _dump_toc_items(doc_id, toc_items, stage="before_neo4j")

    # 필수 키 보정 및 누락 로그
    def _ensure_keys(name: str, items: list[dict], required: list[str]):
        bad = []
        for it in items:
            missing = [k for k in required if k not in it or it.get(k) in (None, "")]
            if missing:
                bad.append({"item": it, "missing": missing})
        if bad:
            logger.warning(f"[Upload] {name} 누락 필드 감지: {len(bad)} 건 (예: {bad[:2]})")

    _ensure_keys("Facts", facts, ["fact_id", "fact", "section_id"]) 
    # Keywords validation removed - handled by canonical extraction pipeline
    
    # TOC 업로드 전 상태 스냅샷 저장
    _save_toc_snapshot(doc_id, toc_items, stage="before_neo4j")

    # 노드 업로드 (Table 제외)
    logger.info("📦 [Neo4j] 노드 업로드 시작 (Table 제외)...")
    try:
        if db is None:
            raise RuntimeError("Neo4j DB instance must be provided (no global singleton)")
        await upload_nodes("Document", [document_package["document"]], db)
        await upload_nodes("TOC", toc_items, db)
        await upload_nodes("Section", sections, db)
        await upload_nodes("Fact", facts, db)
        # Keyword nodes removed - now handled by canonical extraction pipeline

        # Folder hierarchy upload from document_minio_url
        minio_url = document_package.get("document", {}).get("document_minio_url")
        if minio_url:
            folder_count, folder_rel_count = await upload_folder_hierarchy(db, minio_url, doc_id)
            logger.info(f"📁 [Neo4j] Folder hierarchy: {folder_count} folders, {folder_rel_count} relationships")
    except Exception as e:
        logger.error(f"❌ [Neo4j] 노드 업로드 중 오류 발생: {e}", exc_info=True)
        raise
    
    # 관계 생성 (Table 관련 관계 제외)
    logger.info("🔗 [Neo4j] 관계 생성 시작 (Table 제외)...")
    try:
        fact_map_by_section = defaultdict(list)
        for fact in facts:
            sid = fact.get("section_id") if isinstance(fact, dict) else None
            if not sid:
                logger.warning(f"[Upload] section_id 없는 Fact 스킵: {fact}")
                continue
            fact_map_by_section[sid].append(fact)
        
        # keyword_map_by_fact removed - handled by canonical extraction pipeline

        rels = []
        rels += [{"from_label": "Document", "from_id": doc_id, "rel": "HAS_TOC", "to_label": "TOC", "to_id": toc["toc_id"]} for toc in toc_items]

        # Debug TOC-Section relationship creation
        toc_section_rels = []

        # DIAGNOSTIC: Check for duplicate section_ids with different toc_ids before creating relationships
        section_toc_check = {}
        for section in sections:
            toc_item_id = section.get("toc_item_id")
            section_id = section.get("section_id")
            if toc_item_id and section_id:
                if section_id in section_toc_check:
                    if section_toc_check[section_id] != toc_item_id:
                        logger.error(f"❌ [UPLOAD DUPLICATE] Section {section_id} has multiple TOCs in upload: {section_toc_check[section_id]} AND {toc_item_id}")
                section_toc_check[section_id] = toc_item_id
                toc_section_rels.append({"from_label": "TOC", "from_id": toc_item_id, "rel": "HAS_SECTION", "to_label": "Section", "to_id": section_id})
            else:
                logger.warning(f"[Relationships] Skipping TOC-Section relationship for section_id='{section_id}', toc_item_id='{toc_item_id}' (missing field)")

        rels += toc_section_rels
        logger.info(f"[Relationships] Created {len(toc_section_rels)} TOC-Section relationships out of {len(sections)} sections")
        logger.info(f"[Relationships] Unique section_ids in relationships: {len(section_toc_check)}")
        
        for section in sections:
            for fact in fact_map_by_section.get(section["section_id"], []):
                rels.append({"from_label": "Section", "from_id": section["section_id"], "rel": "HAS_FACT", "to_label": "Fact", "to_id": fact["fact_id"]})
        
        rels += [{"from_label": "Section", "from_id": r.source_section_id, "rel": "REFERS_TO", "to_label": "Section", "to_id": r.target_section_id} for r in section_relations]
        
        # Fact-Keyword relationships removed - handled by canonical extraction pipeline
        rels += [{"from_label": "TOC", "from_id": toc.get("parent_toc_id"), "rel": "HAS_SUBTOC", "to_label": "TOC", "to_id": toc["toc_id"]} for toc in toc_items if toc.get("parent_toc_id")]
        
        logger.info(f"🔗 [Neo4j] 총 {len(rels)}개의 관계 생성됨 (Table 제외)")
        
        grouped_rels = defaultdict(list)
        for rel in rels:
            if rel["from_id"] and rel["to_id"]:
                key = (rel["from_label"], rel["rel"], rel["to_label"])
                grouped_rels[key].append({"from_id": rel["from_id"], "to_id": rel["to_id"]})
        
        if db is None:
            raise RuntimeError("Neo4j DB instance must be provided (no global singleton)")
        await upload_relationships(db, grouped_rels)
        logger.info("✅ [Neo4j] 모든 작업 완료 (Table 제외)")
        
    except Exception as e:
        logger.error(f"❌ [Neo4j] 관계 생성 중 오류 발생: {e}", exc_info=True)
        raise
    # 최종 사용된 document_id 반환 (reconcile 반영)
    return doc_id


async def upload_to_qdrant(document_package, encoder, db, qdrant_collections=None):
    """Upload document data to Qdrant.
    
    Args:
        document_package: Packaged document data
        encoder: Text encoder for embeddings
        db: Qdrant database connection
        qdrant_collections: QdrantCollectionNames (optional, falls back to default)
    """
    doc_title = document_package["document"].get("document_title", "")
    sections = document_package["sections"]
    facts = document_package["facts"]
    # Keywords removed - handled by canonical extraction pipeline
    tables = document_package["tables"]
    
    # Determine collection names
    if qdrant_collections is None:
        from shared.utils.service_router import get_default_qdrant_collection_names
        qdrant_collections = get_default_qdrant_collection_names()
    
    await asyncio.gather(
        upload_sections_to_qdrant(sections, doc_title, qdrant_collections.sections, encoder, db),
        upload_facts_to_qdrant(facts, doc_title, qdrant_collections.facts, encoder, db),
        # Keywords upload removed - handled by canonical extraction pipeline
        return_exceptions=True,
    )
    logger.info("✅ [Qdrant] 모든 작업 완료")
