"""
Section header cleaning functions using LLM for Docling structure extraction

Enhanced with batched processing (50 headers per batch) and structured output:
- Prevents token limit issues with large documents
- Maintains hierarchical context across batches
- Uses structured output (BatchedSectionHeaders) for each batch
- Fallback to single-pass processing if batching fails
- Preserves chronological order when merging batch results
"""

import asyncio
import logging
import re
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from shared.factory.llm_factory import get_llm
from shared.config.config import llm_config

logger = logging.getLogger(__name__)


class SectionHeaderAssignment(BaseModel):
    """Represents a valid section header with hierarchical index assigned by LLM"""
    toc_id: int = Field(default=0, description="Sequential TOC ID number starting from 1")
    hierarchical_index: str = Field(..., description="Hierarchical index like '1', '1.1', '2.3.1'")
    section_text: str = Field(default="", description="Cleaned section title", alias="header")
    position: int = Field(default=0, description="Position in docling_body")

    class Config:
        populate_by_name = True  # Allow both 'section_text' and 'header'


class CleanedSectionHeaders(BaseModel):
    """Result of LLM processing of section headers"""
    assignments: List[SectionHeaderAssignment] = Field(default_factory=list)


class BatchedSectionHeaders(BaseModel):
    """Result of LLM processing a batch of section headers with context"""
    batch_assignments: List[SectionHeaderAssignment] = Field(default_factory=list)
    last_hierarchical_index: str = Field("", description="Last valid hierarchical index from this batch for next batch context")


def _next_number(hierarchical_index: str) -> str:
    """Calculate expected next sequential number from a hierarchical index"""
    if not hierarchical_index:
        return "1"

    # Handle hierarchical (e.g., "1.2.3")
    if '.' in hierarchical_index:
        parts = hierarchical_index.split('.')
        # Could be next sibling (1.2.3 → 1.2.4) or parent sibling (1.2.3 → 1.3)
        # Return sibling increment as hint
        parts[-1] = str(int(parts[-1]) + 1)
        return '.'.join(parts)

    # Simple number (e.g., "13" → "14")
    try:
        return str(int(hierarchical_index) + 1)
    except ValueError:
        return hierarchical_index  # Can't parse, return as-is


async def process_header_batch(
    batch_headers: List[Dict[str, Any]],
    batch_idx: int,
    previous_context: str = ""
) -> BatchedSectionHeaders:
    """Process a batch of section headers with LLM using structured output"""

    if not batch_headers:
        return BatchedSectionHeaders()

    # Build context-aware system prompt
    system_prompt = (
        "You are a document structure expert. Process this BATCH of section headers and FILTER for valid TOC entries. "
        "CRITICAL: Analyze the ENTIRE batch to understand document hierarchy based on CONTENT, not input order.\n\n"

        "FILTERING CRITERIA - KEEP headers that are:\n"
        "- **Standard insurance policy legal headings**:\n"
        "  - Examples: '제 1절 ...', '제 1관 ...', '제1관 ...', '제 1 조 ...', '제1조 ...'\n"
        "- Roman TOC-like headings:\n"
        "  - Examples: 'I. 약관 안내', 'II. 보장 내용'\n"
        "- Appendices / tables / supplementary sections:\n"
        "  - Examples: '【별표】', '별표 1', '별표', '부록', '분류표'\n"
        "- **Specific(possibly long) rider/policy titles even WITHOUT numbering/prefix**:\n"
        "  - Examples:\n"
        "    - 'NGS유전자패널검사비용지원S특약(갱신형) 무배당 약관'\n"
        "    - '단체취급특약 약관'\n"
        "  - Heuristic: if it looks like a product/rider name (keywords like '특약', '약관'), treat it as a TOC entry.\n\n"
        "  - CRITICAL RESET RULE:\n"
        "    - **Long rider/policy titles (often ending with '특약', '약관')**, even if that appear after a chain of 절/관/조 headings, **MUST start a NEW TOP-LEVEL number**.\n"
        "    - **Do NOT nest these titles under the preceding 절/관/조 index. They represent a new document block.**\n"
        "    - Example: if the current deepest index is '15.32.2.6.2.12.7', then '스마트연금전환특약(K3.1) 무배당 약관' should be '16' (NOT '15.32.2.6.2.12.8' nor '15.32.2.6.2.12.7.1').\n\n"
        
        "FILTERING CRITERIA - REJECT headers that are:\n"
        "✗ Generic/vague phrases: '내용', '그 내용', '기타', '참고', '비고', '【사례】', '【주의사항】', '【유의사항】'\n"
        "✗ Bracketed examples/notes that should NOT be TOC (especially BETWEEN 조 headings):\n"
        "  - Reject things like: '【...예시...】', '【...사례...】', '【...주의사항...】', '【...유의사항...】', '【제 n조의 ... 예시】', and any other bracketed examples/notes that are not appendix/table markers.\n"
        "  - Exception: KEEP true appendices/tables like '【별표】', '【별표 1】', '별표', '부록', '분류표' (those are TOC).\n"
        "✗ Transitional text: '다음과 같다', '아래와 같이'\n"
        "✗ Page headers/footers or document metadata\n"
        "✗ List item markers without substance (e.g., just 'ㅇ', '①', '-')\n"
        "✗ Single words without context (e.g., '현황', '대책' alone)\n"
        "✗ Fragments or incomplete sentences\n\n"

        "HIERARCHICAL INDEX ASSIGNMENT (CRITICAL - READ CAREFULLY):\n"
        "Step 1: FIRST, scan ALL headers in the batch to identify explicit patterns:\n"
        "   - hierarchical_index must be a string using dots for hierarchy, e.g., '1', '1.2', '3.1.4'\n"
        "   - Convert Roman numerals (I, II, III, …) into Arabic numbers in order (I→1, II→2, III→3).\n"

        "Step 2: DETECT nested heading patterns (CRITICAL EDGE CASE):\n"
        "- For Korean legal headings, treat unit levels as different hierarchy depths (IMPORTANT):\n"
        "  - 절 (largest) > 관 > 조 (smallest)\n"
        "  - Do NOT assign the same index to '제 1관' and '제 1조'. Even if both have '1', they are different levels.\n"
        "  - If 'n 조' follows 'n 관' with the same number, nest it one level deeper under that 관.\n"
        "    - Example: '제 n 관 ...' = '15.32.2' ⇒ subsequent '제 n 조 ...' = '15.32.2.1'\n"
        "  - Example mapping (recommended):\n"
        "    - '제 1관 ...' → '1'\n"
        "    - '제 1조 ...' → '1.1'\n"
        "    - '제 2조 ...' → '1.2'\n"

        "\n"
        "  - CONTINUITY RULE (조 numbering must stay connected to the SAME 관):\n"
        "    - If you are inside a 관 block (e.g., '제 1관 ...' already started) and you see '제 1조', '제 2조', ...,\n"
        "      then the subsequent '제 3조' MUST continue as the NEXT sibling 조 under that SAME 관 UNLESS the long rider/policy titles starts a new major block.\n"
        
        "    - If OTHER valid TOC entries appear BETWEEN consecutive 조 items,\n"
        "      you MUST NEST those intervening entries UNDER the MOST RECENT 조 (as its child), NOT as a new sibling of the 관.\n"
        "      **NO BACKTRACKING**: you MUST NOT jump to a higher-level sibling index (e.g., 2.2.2.7) and then later return to a previous prefix such as 2.2.2.6.x.\n"
        
        "    - Example:\n"
        "      - '제 1관 ...' → 1\n"
        "      - '제 1조 ...' → 1.1\n"
        "      - '제 2조 ...' → 1.2\n"
        "      - 'other valid toc between 조 ...' → 1.2.1 (MUST be a child of the most recent TOC item: 1.2)\n"
        "      - '【예시】' → REJECT\n"
        "      - '제 3조 ...' → 1.3 (MUST be sibling of 1.2; NOT 1.2.2)\n"
        
        "- For long rider/policy titles without numbering:\n"
        "  - If it starts a new major block, assign a new top-level number (reset to the NEXT ROOT, e.g., previous root 15 → new root 16), regardless of prior 절/관/조 depth.\n"
        "  - Only if it is clearly labeled as a subsection of an existing rider block should it be nested; default to NEW TOP-LEVEL when unsure.\n\n"
        "  - Example mapping if rider/policy titles exist (recommended):\n"
        "    - 'NGS유전자패널검사비용지원S특약(갱신형) 무배당 약관' → '1'\n"
        "    - '제 1절 ...' → '1.1'\n"
        "    - '제 1관 ...' → '1.1.1'\n"
        "    - '제 2절 ...' → '1.2'\n"
        "    - '제 1관 ...' → '1.2.1'\n"
        "    - '제 1-1조 ...' → '1.2.1.1'\n"
        "    - '제 1-2조 ...' → '1.2.1.2'\n"
        "    - **'단체취급특약 약관' → '2'**\n"
        "    - '제 1관 ...' → '2.1'\n"
        "    - '제 1조 ...' → '2.1.1'\n"
        "    - **'【부록】 법령내용' → '3'** (BACKTRACKING MUST NOT HAPPEN HERE: after you advance to 3, a later TOC item MUST NOT revert to 2.x or lower)\n"
        "    - **'제 4-1조 ...' → '3.1'**\n"
        "    - **'제 9조 ...' → '3.2'**\n"
        "Do not blindly assign 1,2,3… by input order. Prioritize explicit patterns: '절/관/조', Roman numerals, and long rider/policy titles.\n"

        "EDGE CASE - Nested numbering that restarts:\n"
        "Batch: ['1. 주제', '1. 소주제1', '2. 소주제2', '3. 소주제3', '2. 주제2']\n"
        "WRONG assignment: ['1', '1', '2', '3', '2'] ❌ (no duplicates or backtracking!)\n"
        "CORRECT assignment: ['1', '1.1', '1.2', '1.3', '2'] ✅\n"
        "Reasoning: Repeated '1' after main '1' indicates subsection. Convert to hierarchical.\n\n"

        "IMPORTANT: DO NOT just assign 1, 2, 3, 4, 5... based on input order. LOOK AT THE TEXT CONTENT!"
    )

    if previous_context:
        # Enhanced context with counting guidance
        system_prompt += (
            f"\n\n{'='*60}\n"
            f"CRITICAL BATCH CONTINUATION CONTEXT:\n"
            f"Previous batch ended with hierarchical_index: '{previous_context}'\n\n"
            f"CONTINUATION RULES (STRICTLY ENFORCE):\n"
            f"1. NEXT sequential number MUST be: '{_next_number(previous_context)}'\n"
            f"   - Example: if previous='13' → next='14' (NOT 23, NOT 20, EXACTLY 14)\n"
            f"   - Example: if previous='5.3' → next='5.4' OR '6' (depending on hierarchy)\n\n"
            f"2. COUNT CAREFULLY as you assign:\n"
            f"   - If you assign 14, 15, 16 → last will be '16' (count: 3 headers)\n"
            f"   - Your last_hierarchical_index MUST match the LAST number you actually assigned\n\n"
            f"3. NO SKIPPING NUMBERS:\n"
            f"   - Sequential must be continuous: 14, 15, 16, 17...\n"
            f"   - Hierarchical must be logical: 1.1, 1.2, 1.3 OR 1.1, 2, 3...\n"
            f"4. RIDER RESET EXCEPTION (IMPORTANT):\n"
            f"   - If you see a long rider/policy title (keywords: '특약', '약관', product name) that clearly starts a new document block, immediately start a NEW top-level number (previous root 15 → new root 16), even if the prior index was deep like '15.32.2.6'.\n"
            f"   - Do NOT continue deep chains for such titles; they are new roots by default unless explicitly marked as a subsection of an existing rider.\n"
            f"{'='*60}\n"
        )

    # Build header list for this batch
    header_list = []
    for h in batch_headers:
        header_list.append(f"toc_id={h['toc_id']}, position={h['position']}: \"{h['section_text']}\"")

    user_prompt = (
        f"Process batch {batch_idx + 1} with {len(batch_headers)} headers:\n\n"
        "Below is a list of section_header candidates detected by Docling.\n"
        "Keep only the items that should become TOC entries for insurance policy/rider documents, and assign hierarchical_index to each kept item.\n\n"
        + "\n".join(header_list) + "\n\n"
        "PROCESSING WORKFLOW:\n\n"
        "1) Scan all candidates to detect patterns:\n"
        "   - 제X절 / '제X관 / 제X조'\n"
        "   - Roman numerals\n"
        "   - Long rider/policy titles without numbering (usually ends with '약관')\n"
        "2) FILTER invalid headers\n"
        "   - REMOVE: repeated headers/footers, TOC page label ('목차'), noise phrases, marker-only items\n"
        "   - KEEP: meaningful structural headings including 절/관/조, roman headings, and long rider/policy titles\n"
        "3) Assign hierarchical_index with dot hierarchy:\n"
        "   - Preserve explicit numbering when possible (convert Roman numerals to Arabic numbers)\n"
        "   - Ensure 절/관/조 become different hierarchy depths\n"
        "   - IMPORTANT: '제 1관' and '제 1조' must NOT share the same index; use dot hierarchy.\n"
        "   - IMPORTANT CONTINUITY (NO BACKTRACKING): If '제 n조' continues within the same 관, keep 조 as siblings (…1, …2, …3...).\n"
        "     If other valid TOC entries appear BETWEEN 조 items, NEST them under the MOST RECENT 조 as children (…1.1, …1.2, etc.).\n"
        "     Never assign an intervening item a higher-level sibling index that would force later items to return to the prior 관 prefix.\n"
        "4) If previous_context is provided, continue top-level numbering naturally when appropriate.\n\n"
        
        "FEW-SHOT EXAMPLES (Learn from these):\n\n"
        "EXAMPLE 1 - Batch Continuation:\n"
        "Context: Previous batch ended with '13'\n"
        "Headers: ['【별표 1】 보험금 지급기준', '【별표 2】 장해분류표', '부칙']\n"
        "CORRECT:\n"
        "  - '【별표 1】 보험금 지급기준' → hierarchical_index='14' (previous 13 + 1)\n"
        "  - '【별표 2】 장해분류표' → hierarchical_index='15' (continue)\n"
        "  - '부칙' → hierarchical_index='16' (continue)\n"
        "  - last_hierarchical_index='16' ✓\n\n"
        "WRONG:\n"
        "  - '【별표 1】 보험금 지급기준' → hierarchical_index='23' ✗ (skipped 14-22!)\n"
        "  - '【별표 2】 장해분류표' → hierarchical_index='24' ✗ (skipped!)\n\n"
        "EXAMPLE 2 - First Batch (절/관/조 계층 분리 + 필터링):\n"
        "Context: (none - first batch)\n"
        "Headers: ['목차', '제 1관 목적 및 용어의 정의', '제 1조 (용어의 정의)', '【연단위 복리 예시】', '【피보험자의 범위】', 'NGS유전자패널검사비용지원S특약(갱신형) 무배당 약관', '【별표 1】 보험금 지급기준', '【별표 2】 장해분류표']\n"
        "CORRECT:\n"
        "  - '제 1관 목적 및 용어의 정의' → hierarchical_index='1'\n"
        "  - '제 1조 (용어의 정의)' → hierarchical_index='1.1' (조는 관의 하위 레벨)\n"
        "  - '【연단위 복리 예시】' → REJECT since it is not a standard insurance policy legal heading\n"
        "  - '【피보험자의 범위】' → REJECT since it is not a standard insurance policy legal heading\n"
        "  - 'NGS유전자패널검사비용지원S특약(갱신형) 무배당 약관' → hierarchical_index='2' (new major block)\n"
        "  - '【별표 1】 보험금 지급기준' → hierarchical_index='3' (new major block)\n"
        "  - '【별표 2】 장해분류표' → hierarchical_index='4' (new major block)\n"
        "  - last_hierarchical_index='4' ✓\n\n"

        "OUTPUT REQUIREMENTS:\n"
        "- batch_assignments: List of VALID headers with extracted hierarchical_index\n"
        "- last_hierarchical_index: The ACTUAL LAST number you assigned (verify it!)\n"
        "- Empty batch_assignments if ALL headers are invalid (acceptable)\n\n"

        "REMEMBER:\n"
        "- Ignore toc_id values - they are just input indices. Focus on TEXT CONTENT!\n"
        "- Count carefully! If you assign 3 headers starting from 14 → last is 16, NOT 13, NOT 17!\n"
        "- VERIFY your last_hierarchical_index matches your last assignment!"
    )

    # Call LLM with structured output for this batch
    llm = get_llm(llm_config)
    structured_llm = llm.with_structured_output(BatchedSectionHeaders, method="function_calling")

    max_attempts = 2  # 1 initial try + 1 retry for transient LLM/network issues

    try:
        result = None
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                result = structured_llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ])
                break
            except Exception as e:
                last_exc = e
                if attempt < max_attempts:
                    # Common transient failure: "incomplete chunked read"
                    logger.warning(
                        f"[DoclingStructure] Batch {batch_idx + 1} LLM attempt {attempt}/{max_attempts} failed: {e} "
                        f"(retrying once after short backoff)"
                    )
                    await asyncio.sleep(1.0 * attempt)
                else:
                    raise

        if result is None:
            # Defensive: should not happen unless invoke returns None without raising
            raise RuntimeError(f"LLM returned no result for batch {batch_idx + 1}") from last_exc

        # Build lookup from input batch for toc_id and position
        input_lookup: Dict[str, Dict[str, Any]] = {}
        for h in batch_headers:
            # Normalize text for matching (strip whitespace)
            key = (h['section_text'] or "").strip()
            input_lookup[key] = h
            # Also add with first 50 chars as key (for partial matches)
            if len(key) > 50:
                input_lookup[key[:50]] = h

        # Post-process batch results
        cleaned_assignments: List[SectionHeaderAssignment] = []
        last_index = previous_context

        # Track sequential counter for non-numbered headers
        next_sequential = 1
        if previous_context and '.' not in previous_context:
            try:
                next_sequential = int(previous_context) + 1
            except ValueError:
                next_sequential = 1

        # EDGE CASE DETECTION: Track seen hierarchical_index to detect duplicates (nested numbering pattern)
        seen_indices: Dict[str, int] = {}  # hierarchical_index -> first occurrence position
        current_parent: str = ""  # Track current parent section for nesting

        for assignment in result.batch_assignments:
            # Fill in missing toc_id and position from input batch
            text_key = (assignment.section_text or "").strip()
            matched_input = input_lookup.get(text_key) or input_lookup.get(text_key[:50] if len(text_key) > 50 else text_key)

            if matched_input:
                if assignment.toc_id == 0:
                    assignment.toc_id = matched_input['toc_id']
                if assignment.position == 0:
                    assignment.position = matched_input['position']
                # Also fill section_text if empty (LLM might use 'header' alias)
                if not assignment.section_text:
                    assignment.section_text = matched_input['section_text']
                logger.debug(f"[DoclingStructure] Matched assignment to input: toc_id={assignment.toc_id}, position={assignment.position}")
            # Validate and clean hierarchical indices
            if not assignment.hierarchical_index:
                # LLM didn't assign - try to extract from text
                text = assignment.section_text
                # Match patterns like: "1.", "1.1", "1)", "1-", "1 ", "(1)", etc.
                match = re.match(r'^(?:\()?(\d+(?:\.\d+)*)(?:\)|\.|-|:)?\s*', text)
                if match and match.group(1):
                    assignment.hierarchical_index = match.group(1)
                    logger.debug(f"[DoclingStructure] Extracted hierarchical_index='{match.group(1)}' from text: '{text[:50]}'")
                else:
                    # No numbering in text - assign sequential
                    assignment.hierarchical_index = str(next_sequential)
                    logger.debug(f"[DoclingStructure] Assigned sequential hierarchical_index='{next_sequential}' for unnumbered: '{text[:50]}'")
                    next_sequential += 1
            else:
                # # LLM assigned - verify if it matches text numbering
                # text = assignment.section_text
                # # Always prioritize text numbering over LLM assignment
                # match = re.match(r'^(?:\()?(\d+(?:\.\d+)*)(?:\)|\.|-|:)?\s*', text)
                # if match and match.group(1):
                #     # Use the numbering from text (more reliable)
                #     extracted = match.group(1)
                #     if extracted != assignment.hierarchical_index:
                #         logger.info(
                #             f"[DoclingStructure] Correcting LLM assignment: '{assignment.hierarchical_index}' → '{extracted}' "
                #             f"(from text: '{text[:50]}')"
                #         )
                #     assignment.hierarchical_index = extracted
                # # else: keep LLM-assigned sequential 
                
                pass    # LLM이 부여한 hierarchical_index 그대로 유지

            # EDGE CASE FIX: Detect nested numbering pattern (only fix clear duplicates)
            final_index = assignment.hierarchical_index

            # Only fix duplicates - let LLM handle complex semantic understanding
            if final_index in seen_indices:
                # Duplicate detected! This is a subsection with restarted numbering
                if current_parent and '.' not in final_index:
                    # Simple number that's a duplicate - make it a child of current parent
                    final_index = f"{current_parent}.{final_index}"
                    logger.info(
                        f"[DoclingStructure] Nested numbering detected (duplicate): '{assignment.hierarchical_index}' → '{final_index}' "
                        f"(subsection of parent '{current_parent}')"
                    )
                    assignment.hierarchical_index = final_index
                else:
                    # Complex duplicate - append subsection counter
                    subsection_count = sum(1 for idx in seen_indices.keys() if idx.startswith(f"{final_index}."))
                    final_index = f"{final_index}.{subsection_count + 1}"
                    logger.warning(
                        f"[DoclingStructure] Unexpected duplicate index: '{assignment.hierarchical_index}' → '{final_index}'"
                    )
                    assignment.hierarchical_index = final_index

            # Track current parent for next iteration
            if '.' not in final_index:
                current_parent = final_index
            else:
                # Update current parent to the root of this hierarchical index
                current_parent = final_index.split('.')[0]

            # Track this index to detect future duplicates
            seen_indices[final_index] = len(cleaned_assignments)

            last_index = assignment.hierarchical_index
            cleaned_assignments.append(assignment)

            # Update sequential counter based on highest root-level number seen
            if '.' not in assignment.hierarchical_index:
                try:
                    root_num = int(assignment.hierarchical_index.split('.')[0])
                    if root_num >= next_sequential:
                        next_sequential = root_num + 1
                except ValueError:
                    pass

        logger.info(f"[DoclingStructure] Batch {batch_idx + 1}: Processed {len(batch_headers)} → {len(cleaned_assignments)} valid headers")

        return BatchedSectionHeaders(
            batch_assignments=cleaned_assignments,
            last_hierarchical_index=last_index
        )

    except Exception as e:
        logger.warning(f"[DoclingStructure] Batch {batch_idx + 1} LLM processing failed after {max_attempts} attempt(s): {e}")
        # Fallback: generate basic assignments for this batch
        fallback_assignments = []
        for h in batch_headers:
            text = h['section_text']
            # Use improved regex pattern matching various numbering formats
            match = re.match(r'^(?:\()?(\d+(?:\.\d+)*)(?:\)|\.|-|:)?\s*', text)
            if match and match.group(1):
                fallback_assignments.append(SectionHeaderAssignment(
                    toc_id=h['toc_id'],
                    hierarchical_index=match.group(1),
                    section_text=text,
                    position=h['position']
                ))
                logger.debug(f"[DoclingStructure] Fallback extracted: '{match.group(1)}' from '{text[:50]}'")

        return BatchedSectionHeaders(
            batch_assignments=fallback_assignments,
            last_hierarchical_index=fallback_assignments[-1].hierarchical_index if fallback_assignments else previous_context
        )


def generate_assignments_from_docling_body(
    docling_body: List[Dict[str, Any]],
    max_toc_depth: int = 5
) -> List[SectionHeaderAssignment]:
    """Generate fallback section header assignments from docling body

    Args:
        docling_body: Docling 파싱 결과
        max_toc_depth: TOC에 포함할 최대 깊이 (기본 2 = 1, 1.1까지만)
                       예: 2이면 1, 1.1은 포함, 1.1.1은 제외

    Returns:
        상위 레벨 section header만 포함한 TOC 항목 리스트
    """
    assignments: List[SectionHeaderAssignment] = []
    sequential_counter = 1

    for idx, item in enumerate(docling_body):
        if item.get("label") != "section_header":
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue

        # Check if it has a numbered format (improved pattern)
        match = re.match(r'^(?:\()?(\d+(?:\.\d+)*)(?:\)|\.|-|:)?\s*', text)
        if match and match.group(1):
            hierarchical_index = match.group(1)

            # 깊이 체크: max_toc_depth 이하만 포함
            depth = len(hierarchical_index.split('.'))
            if depth > max_toc_depth:
                logger.debug(f"[DoclingStructure] Skipping deep header (depth={depth}): '{text[:50]}'")
                continue

            # Update sequential counter to continue after this number
            try:
                base_num = int(hierarchical_index.split('.')[0])
                if base_num >= sequential_counter:
                    sequential_counter = base_num + 1
            except (ValueError, IndexError):
                pass

            assignments.append(
                SectionHeaderAssignment(
                    toc_id=idx + 1,
                    hierarchical_index=hierarchical_index,
                    section_text=text,
                    position=idx,
                )
            )
        else:
            # 번호가 없는 section_header는 fallback에서 TOC로 포함하지 않음
            # (LLM이 처리할 때만 의미있는 제목으로 판단하여 포함)
            logger.debug(f"[DoclingStructure] Skipping unnumbered header in fallback: '{text[:50]}'")
            continue

    logger.info(f"[DoclingStructure] Generated {len(assignments)} fallback TOC items (max_depth={max_toc_depth}, numbered only)")
    return assignments


async def clean_docling_section_headers(docling_body: List[Dict[str, Any]], batch_size: int = 50) -> CleanedSectionHeaders:
    """Extract section headers from docling body and assign hierarchical indices via LLM with batched processing"""

    # Extract section headers in chronological order
    section_headers = []
    for i, item in enumerate(docling_body):
        label = item.get("label", "")

        if label == "section_header":
            section_text = item.get("text", "").strip()
            if section_text:
                section_headers.append({
                    "position": i,  # Position in docling_body
                    "toc_id": i + 1,  # Sequential ID for LLM reference
                    "section_text": section_text
                })
                logger.info(f"[DoclingStructure] Found section header {len(section_headers)}: '{section_text}' at position {i}")

    if not section_headers:
        logger.info("[DoclingStructure] No section headers found in docling body")
        return CleanedSectionHeaders(assignments=[])

    total_headers = len(section_headers)
    logger.info(f"[DoclingStructure] Found {total_headers} section headers, processing in batches of {batch_size}...")

    # BATCHED PROCESSING: Split headers into batches of 100
    all_assignments: List[SectionHeaderAssignment] = []
    previous_context = ""

    try:
        # Process headers in batches
        for batch_idx in range(0, total_headers, batch_size):
            batch_headers = section_headers[batch_idx:batch_idx + batch_size]

            logger.info(f"[DoclingStructure] Processing batch {batch_idx // batch_size + 1}: headers {batch_idx + 1}-{min(batch_idx + batch_size, total_headers)}")

            # Process this batch with structured output
            batch_result = await process_header_batch(
                batch_headers,
                batch_idx // batch_size,
                previous_context
            )

            # VERIFICATION: Check for number jumps between batches
            if batch_result.batch_assignments and previous_context:
                first_assigned = batch_result.batch_assignments[0].hierarchical_index
                expected_next = _next_number(previous_context)

                # Only warn for simple sequential numbers (not hierarchical like 1.1)
                if '.' not in first_assigned and '.' not in previous_context:
                    try:
                        first_num = int(first_assigned)
                        expected_num = int(expected_next)

                        if first_num != expected_num:
                            logger.warning(
                                f"[DoclingStructure] NUMBER JUMP DETECTED in batch {batch_idx // batch_size + 1}! "
                                f"Previous batch ended: '{previous_context}', "
                                f"Expected next: '{expected_next}', "
                                f"But LLM assigned: '{first_assigned}' "
                                f"(jumped {first_num - expected_num} numbers)"
                            )
                        else:
                            logger.info(
                                f"[DoclingStructure] ✓ Batch {batch_idx // batch_size + 1} continuation verified: "
                                f"'{previous_context}' → '{first_assigned}' (correct)"
                            )
                    except (ValueError, AttributeError):
                        # Can't parse as integers, skip verification
                        pass

            # Merge batch results
            all_assignments.extend(batch_result.batch_assignments)
            previous_context = batch_result.last_hierarchical_index

            logger.info(f"[DoclingStructure] Batch {batch_idx // batch_size + 1} completed: {len(batch_result.batch_assignments)} valid headers")

        # Final result - TOC 항목 수가 너무 많으면 단순화 => 보험약관 특성을 고려하여 비활성화
        # MAX_TOC_ITEMS = 100  # 최대 TOC 항목 수
        #
        # if len(all_assignments) > MAX_TOC_ITEMS:
        #     logger.warning(
        #         f"[DoclingStructure] Too many TOC items ({len(all_assignments)} > {MAX_TOC_ITEMS}), "
        #         "filtering to top-level items only"
        #     )
        #     # 상위 레벨(깊이 1-5)만 유지
        #     filtered = [
        #         a for a in all_assignments
        #         if len(a.hierarchical_index.split('.')) <= 5
        #     ]
        #     if filtered:
        #         all_assignments = filtered
        #         logger.info(f"[DoclingStructure] Filtered to {len(all_assignments)} top-level TOC items")
        #     else:
        #         # 모든 항목이 깊은 레벨이면 처음 MAX_TOC_ITEMS개만 유지
        #         all_assignments = all_assignments[:MAX_TOC_ITEMS]
        #         logger.info(f"[DoclingStructure] Truncated to first {MAX_TOC_ITEMS} TOC items")

        final_result = CleanedSectionHeaders(assignments=all_assignments)

        logger.info(f"[DoclingStructure] BATCHED PROCESSING COMPLETE: {total_headers} → {len(all_assignments)} valid headers across {(total_headers + batch_size - 1) // batch_size} batches")

        # Log sample results
        for i, assignment in enumerate(all_assignments[:5]):
            logger.info(f"[DoclingStructure] Final Result #{i}: toc_id={assignment.toc_id}, hierarchical_index='{assignment.hierarchical_index}', section_text='{assignment.section_text}', position={assignment.position}")

        if not all_assignments:
            logger.warning(f"[DoclingStructure] No valid headers from batched processing; using fallback")
            fallback = generate_assignments_from_docling_body(docling_body)
            return CleanedSectionHeaders(assignments=fallback)

        return final_result

    except Exception as e:
        logger.error(f"[DoclingStructure] Batched processing failed: {e}")
        logger.info(f"[DoclingStructure] Falling back to single-pass processing for {total_headers} headers")

        # FALLBACK: Use original single-pass method if batching fails
        fallback = generate_assignments_from_docling_body(docling_body)
        return CleanedSectionHeaders(assignments=fallback)
