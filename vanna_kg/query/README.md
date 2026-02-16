# GraphRAG Query Pipeline

## Executive Summary

The GraphRAG Query Pipeline is a question-answering system that retrieves structured knowledge from a Neo4j graph database and synthesizes natural language answers. The architecture follows a four-phase design: question decomposition, parallel research, context assembly, and final synthesis.

### Core Design Principles

1. **Always Decompose**: Every question, regardless of complexity, undergoes structured decomposition to extract entities, topics, relationships, and generate targeted sub-queries.

2. **Wide-Net Resolution**: Entity and topic hints map one-to-many in the knowledge graph. A single hint like "tech companies" can resolve to multiple graph nodes (Apple, Microsoft, Google, etc.).

3. **Chunk-Centric Retrieval**: The primary retrieval target is EpisodicNodes (document chunks), not just FactNodes. Chunks provide rich contextual content for synthesis.

4. **Parallel Execution**: Sub-queries run concurrently with semaphore-controlled limits, and retrieval operations within each sub-query also execute in parallel.

5. **Question-Type-Aware Synthesis**: The final answer is formatted according to the classified question type (FACTUAL, COMPARISON, CAUSAL, TEMPORAL, ENUMERATION).

### System Components

| Component | Responsibility |
|-----------|----------------|
| GraphRAGPipeline | Main orchestrator coordinating all phases |
| QueryDecomposer | Breaks questions into structured retrieval plans |
| GraphStore | Knowledge graph interface with resolution and retrieval |
| Researcher | Per-sub-query research agent |
| ContextBuilder | Assembles and prioritizes retrieved content |

---

## Pipeline Phases Overview

The V7 pipeline processes questions through four distinct phases, each with timing instrumentation for performance analysis.

### Phase 1: Question Decomposition

**Input**: Natural language question
**Output**: QueryDecomposition containing sub-queries, entity hints, topic hints, temporal scope, and question type

The decomposition phase uses a chain-of-thought approach via an LLM (gemini-3-flash-preview) to extract structured information from the question. Even simple questions undergo decomposition to ensure consistent processing.

### Phase 2: Parallel Research

**Input**: QueryDecomposition from Phase 1
**Output**: List of SubAnswer objects, one per sub-query

Each sub-query is researched independently by a Researcher instance. Sub-queries execute in parallel with a configurable concurrency limit (default: 5). Within each research task, entity resolution, topic resolution, and retrieval operations also run in parallel.

### Phase 3: Context Assembly

**Input**: Raw retrieved content (chunks, facts, entities)
**Output**: StructuredContext with organized, deduplicated content

The ContextBuilder takes retrieved content from multiple sources and organizes it into a structured format with:
- Deduplication by chunk_id and fact_id
- Relevance-based splitting (high vs. low relevance)
- Category-specific limits
- Proper ordering for LLM consumption

### Phase 4: Final Synthesis

**Input**: Original question, list of SubAnswer objects, question type
**Output**: Final answer with confidence score

An LLM (gpt-5.2) merges all sub-answers into a coherent final response. The synthesis is guided by question-type-specific formatting instructions.

---

## Question Decomposition

The QueryDecomposer performs structured extraction following a six-step chain-of-thought process.

### Step-by-Step Decomposition Algorithm

**Step 1 - Extract Entities with Definitions**
- Identify named entities explicitly mentioned in the question text
- For each entity, generate a contextual definition to aid semantic matching
- Only extract what appears in the question; do not enumerate from external knowledge
- Definitions describe the entity's category or type

**Step 2 - Extract Topics with Definitions (Topic Chain-of-Thought)**
- First, extract concepts directly stated in the question (Mentioned Topics)
- Then, infer related contexts where relevant data might be stored (Context Topics)
- The guiding question: "Where in a document would this information be written?"
- Include 1-2 context topics beyond explicit mentions
- Each topic gets a contextual definition

**Step 3 - Extract Relationships**
- Extract action verbs and their modifiers as relationship phrases
- Include qualifiers: "increased slightly", "declined modestly", "remained unchanged"
- Include manner: "reported", "experienced", "saw"
- Relationships capture HOW entities relate to topics

**Step 4 - Identify Temporal Scope**
- Extract any time references (specific dates, quarters, months, years)
- Handle relative references: "recent", "previous", "last"

**Step 5 - Classify Question Type**
- FACTUAL: Questions about specific facts or states
- COMPARISON: Questions comparing multiple entities
- CAUSAL: Questions about cause and effect
- TEMPORAL: Questions about change over time
- ENUMERATION: Questions asking which/what entities match criteria

**Step 6 - Generate Sub-Queries**
- Create keyword-focused search queries combining extracted elements
- Combine entities + topics + relationship keywords
- For comparison questions, generate separate queries per entity
- Each sub-query includes its own entity_hints and topic_hints

### Sub-Query Generation

Each SubQuery contains:
- **query_text**: Search query text (keyword phrase, not full sentence)
- **target_info**: Description of what this sub-query aims to find
- **entity_hints**: Entity names to resolve for this sub-query
- **topic_hints**: Topic names to resolve for this sub-query

For comparison questions like "Compare inflation in Boston vs NY", the decomposer generates combinatorial sub-queries:
- Sub-query 1: "inflation Boston" with entity_hints=["Boston"]
- Sub-query 2: "inflation New York" with entity_hints=["New York"]

This enables parallel retrieval and cross-query context building.

### Entity Hint Structure

EntityHints include contextual definitions:
- **name**: The entity/topic name as mentioned in the question
- **definition**: Brief contextual definition to aid semantic matching

Example:
- name: "Boston"
- definition: "Federal Reserve regional banking district"

### Question Type Classification

| Type | Description | Detection Keywords |
|------|-------------|-------------------|
| FACTUAL | Simple fact lookup | (default when no other type matches) |
| COMPARISON | Compare multiple entities | compare, versus, vs, differ, difference |
| CAUSAL | Cause and effect relationships | why, cause, because, led to, affect, effect, result |
| TEMPORAL | Change over time | change, trend, over time, since, from, to |
| ENUMERATION | List items matching criteria | which, list, what are, how many |

### Fallback Decomposition

If LLM decomposition fails, a fallback algorithm:
1. Extracts capitalized words (excluding stop words) as potential entities
2. Detects question type from keywords
3. Creates a single sub-query with the original question text
4. Sets confidence to 0.3 to indicate degraded quality

---

## Entity and Topic Resolution

Resolution is the process of matching decomposition hints to actual nodes in the knowledge graph. The system employs a "wide-net" principle where one hint can resolve to many nodes.

### Wide-Net Principle

The resolution strategy prioritizes recall over precision:
- **One-to-Many Matching**: A single query term may map to multiple entities
  - "tech companies" resolves to Apple, Microsoft, Google, Amazon, etc.
  - "Federal Reserve officials" resolves to Jerome Powell and multiple Fed governors
- **Partial Matches Included**: "Apple" matches "Apple Inc." and "Apple Park"
- **Alias Handling**: "Fed" matches "Federal Reserve", "Federal Reserve System", regional Fed banks

The philosophy is: it's better to include marginally relevant entities than to miss important ones. The retrieval and synthesis phases will filter appropriately.

### Resolution Algorithm

For each hint (entity or topic):

1. **Cache Check**: Look up the hint (lowercase, stripped) in the resolution cache
   - If found, return cached results immediately
   - Cache entries map hint to list of (name, summary) tuples

2. **Vector Search for Candidates**: If not cached, query the appropriate vector index
   - Entity candidates: `entity_name_embeddings` index
   - Topic candidates: `topic_embeddings` index
   - Parameters: top_k candidates (30 for entities, 20 for topics), similarity threshold

3. **LLM Verification**: Present candidates to an LLM (gemini-3-flash-preview) with structured output
   - System prompt emphasizes wide-net matching
   - LLM returns list of ResolvedNode objects (name + reason) or no_match=true
   - Case-insensitive matching handles minor discrepancies

4. **Cache Results**: Store the resolved (name, summary) pairs in cache

5. **Deduplication**: After resolving all hints, deduplicate by resolved_name

### Resolution Caching

The GraphStore maintains two resolution caches:
- **Entity Cache**: Maps lowercase hint to list of (name, summary) tuples
- **Topic Cache**: Maps lowercase hint to list of (name, definition) tuples

Caching prevents redundant LLM calls when the same hint appears across multiple sub-queries. Cache entries include empty results to avoid re-querying hints that yielded no matches.

### Vector Search Parameters

| Parameter | Entity Resolution | Topic Resolution |
|-----------|-------------------|------------------|
| Vector Index | entity_name_embeddings | topic_embeddings |
| Top-K Candidates | 30 | 20 |
| Similarity Threshold | 0.35 (configurable) | 0.35 (configurable) |

### LLM Resolution Prompts

**Entity Resolution System Prompt Key Points:**
- Cast a wide net - find ALL potentially relevant entities
- One-to-many matching expected
- Include partial matches
- Prefer recall over precision
- Consider aliases and variations

**Topic Resolution System Prompt Key Points:**
- Wide net for topics providing relevant context
- One-to-many matching for thematic breadth
- Hierarchical matching (specific and general topics)
- Thematic connections that provide understanding

---

## Retrieval Operations

Once entities and topics are resolved, the Researcher executes multiple retrieval operations in parallel to gather comprehensive context.

### Retrieval Strategy Overview

For each resolved entity:
1. Entity chunks retrieval
2. Entity facts retrieval (subject and object queries)
3. 1-hop neighbor expansion (if enabled)

For each resolved topic:
1. Topic chunks retrieval

Additionally:
- Global chunk search (if enabled) as fallback coverage

### Entity Chunks Retrieval

Retrieves EpisodicNodes (document chunks) connected to an entity via relationship edges.

**Algorithm:**
1. Query pattern: (EntityNode) -[r {fact_id}]-> (EpisodicNode)
2. Optionally join to DocumentNode for metadata
3. Score chunks by embedding similarity to the query
4. Filter by threshold (default: 0.35)
5. Return chunks with: chunk_id, content, header_path, doc_id, document_date, score

**Fallback:** If the complex query with vector scoring fails, a simpler query retrieves chunks without relevance ranking (assigns default score of 0.5).

### Topic Chunks Retrieval

Retrieves chunks discussing a specific topic via the DISCUSSES relationship.

**Algorithm:**
1. Query pattern: (EpisodicNode) -[:DISCUSSES]-> (TopicNode)
2. Match by topic name
3. Return chunks with metadata
4. Default score: 0.6 (topics provide thematic context)

### 1-Hop Neighbor Expansion

Expands context by finding entities connected to resolved entities and retrieving their chunks.

**Get Neighbors Algorithm:**
1. Find entities connected via shared fact_ids on relationship edges
2. Check both directions:
   - Where resolved entity is subject
   - Where resolved entity is object
3. Exclude the original entity from neighbors
4. Order by connection count (more connections = more relevant)
5. Limit to max_neighbors (default: 10)

**Get Neighbor Chunks Algorithm:**
1. For each neighbor entity name
2. Call get_entity_chunks with lower threshold (0.25)
3. Limit to top_k_per_neighbor chunks (default: 5)
4. Deduplicate across neighbors, keeping highest-scoring version

### Entity Facts Retrieval

Retrieves structured facts (FactNodes) involving a specific entity.

**Algorithm:**
1. Vector search against `fact_embeddings` index
2. Two-part query:
   - As subject: (EntityNode {name}) -[r1 {fact_id}]-> (EpisodicNode) -[r2 {fact_id}]-> (target)
   - As object: (source) -[r1 {fact_id}]-> (EpisodicNode) -[r2 {fact_id}]-> (EntityNode {name})
3. Filter by threshold (default: 0.35)
4. Return: fact_id, content, subject, edge_type, object, chunk_id, score

This captures facts where the entity appears as either subject or object of relationships.

### Global Chunk Search

A fallback mechanism that searches across all chunks regardless of entity/topic connections.

**Algorithm:**
1. Vector search against `fact_embeddings` index with query embedding
2. For each matching fact, find its containing EpisodicNode
3. Aggregate by chunk, keeping maximum score per chunk
4. Parameters: top_k (default: 50), threshold (0.25)

Global search provides broad coverage when entity/topic resolution yields few results.

### Retrieval Parallelization

Within a single Researcher.research() call:

1. **Resolution Phase**: Entity and topic resolution run in parallel via asyncio.gather()

2. **Retrieval Phase**: All retrieval tasks are collected and executed via asyncio.gather():
   - N entity chunk tasks (one per resolved entity)
   - N entity fact tasks (one per resolved entity)
   - N neighbor tasks (one per resolved entity, if enabled)
   - M topic chunk tasks (one per resolved topic)
   - 1 global search task (if enabled)

3. **Neighbor Chunks Phase**: After neighbors are identified, their chunks are retrieved in parallel

---

## Context Assembly

The ContextBuilder transforms raw retrieved content into a structured, deduplicated, and prioritized format for LLM synthesis.

### Input Sources

The builder receives:
- **entity_chunks**: Chunks connected to resolved entities
- **neighbor_chunks**: Chunks from 1-hop neighbor entities
- **facts**: Retrieved FactNodes with relationships
- **resolved_entities**: Entities with summaries from resolution
- **topic_chunks**: Chunks connected to resolved topics
- **global_chunks**: Chunks from global vector search

### Deduplication Strategy

**Chunk Deduplication:**
1. Combine entity_chunks + neighbor_chunks + global_chunks
2. Build a map of chunk_id to chunk
3. For duplicates, keep the version with highest vector_score
4. Topic chunks are deduplicated separately against the combined chunk set

**Fact Deduplication:**
1. Build a map of fact_id to fact
2. For duplicates, keep the version with highest vector_score
3. Sort facts by score descending

### Relevance Threshold Splitting

Chunks are split into high and low relevance categories based on vector_score.

**Default Threshold: 0.45**

- vector_score >= 0.45: High relevance (primary evidence)
- vector_score < 0.45: Low relevance (supporting context)

This split allows the synthesis prompt to prioritize high-confidence content while still providing additional context.

### Context Ordering

The StructuredContext organizes content in this order:

1. **Entities Section**: Entity summaries from resolved entities
   - Format: "- {name} ({entity_type}): {summary}"
   - Only entities with summaries are included

2. **High Relevance Context Section**: Top-scoring chunks
   - Full metadata: document name, date, header path
   - Content follows metadata

3. **Facts Section**: Structured facts with relationships
   - Format: "- {subject} {edge_type} {object}: {content}"

4. **Topic Context Section**: Topic-linked chunks
   - Full metadata and content

5. **Additional Context Section**: Lower-relevance chunks
   - Same format as high relevance
   - Provides supplementary information

### Per-Category Limits

To prevent context overflow, each category has configurable limits:

| Category | Default Limit |
|----------|---------------|
| High Relevance Chunks | 30 |
| Facts | 40 |
| Topic Chunks | 15 |
| Low Relevance Chunks | 20 |

Limits are applied after deduplication and sorting.

### Entity Summary Filtering

Only resolved entities with non-empty summaries are included in the context. Entities without summaries provide little value for synthesis and are excluded to save context space.

---

## Final Synthesis

The final synthesis phase merges all sub-answers into a coherent response using an LLM (gpt-5.2).

### Sub-Answer Synthesis (Per Sub-Query)

Each Researcher produces a SubAnswer via its own synthesis step:

**Input:**
- sub_query text and target_info
- StructuredContext formatted as prompt text

**Process:**
1. Convert context to text via StructuredContext.to_prompt_text()
2. Check for empty context - return fallback if no content
3. Format the SUB_ANSWER_USER_PROMPT with query and context
4. Invoke LLM with structured output (SubAnswerSynthesis schema)
5. Return answer, confidence, and entities_mentioned

**Fallback:** If synthesis fails or context is empty, return:
- Answer: "Insufficient information available to answer: {target_info}"
- Confidence: 0.1 (empty context) or 0.0 (error)

### Final Answer Merging

The GraphRAGPipeline._merge_answers() method combines sub-answers:

**Input:**
- Original question
- List of SubAnswer objects
- Question type classification

**Process:**
1. Check for valid answers (confidence > 0.0)
2. If no valid answers, return "No information was found" with confidence 0.0
3. Format sub-answers using format_sub_answers_for_final():
   - Each sub-answer includes: query, target, answer, confidence, entities
4. Get question-type-specific instructions
5. Build final prompt with FINAL_SYNTHESIS_USER_PROMPT
6. Invoke LLM with structured output (FinalSynthesis schema)
7. Return final answer and confidence

**Fallback:** If final synthesis fails:
1. Create simple concatenated answer with headers
2. Average confidence of valid sub-answers, multiplied by 0.8 (penalty)

### Question-Type-Specific Formatting

The synthesis prompt includes type-specific instructions:

**COMPARISON:**
- Structure as side-by-side comparison
- For each aspect, show how entities/periods differ
- Use parallel structure
- Example: "In Boston, X [Source]. In contrast, New York showed Y [Source]."

**ENUMERATION:**
- Use bullet points or numbered lists
- Ensure comprehensive coverage
- Group related items logically
- Indicate if list may be incomplete

**CAUSAL:**
- Show cause-effect relationships explicitly
- Use connective phrases: "This was caused by...", "As a result...", "Leading to..."
- Distinguish correlation from causation

**TEMPORAL:**
- Organize chronologically
- Include specific dates and time periods
- Show progression and changes
- Highlight key turning points

**FACTUAL:**
- Lead with direct answer
- Follow with supporting evidence
- Include specific numbers and details
- Cite sources for key claims

### Source Citation Format

Sub-answer synthesis uses the format:
```
[Source: doc_name, date]
```

Citations should appear immediately after each factual claim. The context chunks provide document metadata (doc_id, document_date, header_path) for constructing citations.

### Confidence Scoring

Confidence flows through the pipeline:

1. **Sub-Answer Confidence**: LLM-assigned 0-1 score based on evidence quality
2. **Final Confidence**: LLM-assigned 0-1 score for the merged answer
3. **Fallback Penalty**: If fallback concatenation is used, confidence is reduced by 20%

The system previously had an abstention threshold (0.3) that would reject low-confidence answers, but this was removed to let confidence "speak for itself."

---

## Fallback Mechanisms

The pipeline includes multiple fallback paths to ensure graceful degradation.

### Decomposition Failures

**Trigger:** LLM decomposition throws exception or returns parsing error

**Fallback Algorithm:**
1. Extract capitalized words as potential entity hints (excluding stop words)
2. Detect question type from keywords:
   - COMPARISON: compare, versus, vs, differ, difference
   - CAUSAL: why, cause, because, led to, affect, effect, result
   - ENUMERATION: which, list, what are, how many
   - TEMPORAL: change, trend, over time, since, from, to
   - FACTUAL: (default)
3. Create single sub-query with original question text (truncated to 100 chars)
4. Set confidence to 0.3
5. Add reasoning: "Fallback decomposition. {error}"

### Resolution Failures

**Entity/Topic Resolution Errors:**
- If vector search or LLM verification fails for a hint, log error and continue
- Return empty list for that hint
- Other hints still process normally

**LLM Resolution Error Fallback:**
- If structured LLM call fails, return top 3 candidates by vector score
- Provides some coverage even without LLM verification

**Empty Resolution:**
- If no entities or topics resolve, global search becomes the primary retrieval method

### Sub-Query Research Failures

**Trigger:** asyncio.gather() catches exception from a sub-query research task

**Fallback:**
- Create SubAnswer with:
  - answer: "Error during research: {exception}"
  - confidence: 0.0
- Continue processing other sub-queries normally

### Empty Result Handling

**No Sub-Queries from Decomposition:**
- Create default sub-query using:
  - query_text: Original question (truncated to 100 chars)
  - target_info: "Answer to the question"
  - entity_hints: From decomposition.entity_hints
  - topic_hints: From decomposition.topic_hints

**No Resolved Entities or Topics:**
- If resolution yields nothing, fall back to global search only
- Build context from global_chunks

**Empty Context:**
- Return fallback answer: "Insufficient information available to answer: {target_info}"
- Confidence: 0.1

**No Valid Sub-Answers:**
- If all sub-answers have confidence 0.0 (all errors)
- Return: "All research attempts failed. Please try again."
- Confidence: 0.0

### Synthesis Failures

**Sub-Answer Synthesis Error:**
- Return: "Unable to synthesize answer: {exception}"
- Confidence: 0.0
- Entities mentioned: []

**Final Synthesis Error:**
- Use _create_fallback_answer() to concatenate valid sub-answers
- Format: "**Finding {i}** ({sub_query}):\n{answer}"
- Confidence: Average of valid sub-answer confidences * 0.8

---

## Configuration Parameters

The V7Config dataclass centralizes all tunable parameters.

### Resolution Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| entity_threshold | 0.35 | Minimum vector similarity for entity resolution |
| topic_threshold | 0.35 | Minimum vector similarity for topic resolution |

Lower thresholds cast a wider net but may include less relevant candidates.

### Relevance Threshold

| Parameter | Default | Description |
|-----------|---------|-------------|
| high_relevance_threshold | 0.45 | Split point for high vs low relevance chunks |

Chunks scoring at or above this threshold go to high_relevance_chunks; below go to low_relevance_chunks.

### Retrieval Limits

| Parameter | Default | Description |
|-----------|---------|-------------|
| max_high_relevance_chunks | 30 | Maximum chunks in primary evidence section |
| max_facts | 40 | Maximum structured facts to include |
| max_topic_chunks | 15 | Maximum topic-related chunks |
| max_low_relevance_chunks | 20 | Maximum supporting context chunks |
| global_search_top_k | 50 | Maximum results from global vector search |

### Feature Toggles

| Parameter | Default | Description |
|-----------|---------|-------------|
| search_definitions | true | Search topic definitions (currently unused in V7) |
| enable_1hop_expansion | true | Enable 1-hop neighbor entity expansion |
| enable_global_search | true | Enable global vector search as fallback |

### Concurrency Control

| Parameter | Default | Description |
|-----------|---------|-------------|
| max_concurrent (pipeline) | 5 | Maximum parallel sub-query research tasks |

Controlled via asyncio.Semaphore in the parallel research phase.

### Model Assignments

| Parameter | Default | Description |
|-----------|---------|-------------|
| decomposition_model | gemini-3-flash-preview | LLM for question decomposition (fast) |
| resolution_model | gemini-3-flash-preview | LLM for entity/topic resolution (fast) |
| synthesis_model | gpt-5.2 | LLM for answer synthesis (accurate) |

### Abstention

| Parameter | Default | Description |
|-----------|---------|-------------|
| abstention_threshold | 0.3 | Confidence below which to consider abstaining |

Note: The abstention threshold was previously used to reject low-confidence answers but is currently not enforced (removed to let confidence speak for itself).

### Multi-Tenancy

| Parameter | Default | Description |
|-----------|---------|-------------|
| group_id | "default" | Multi-tenant group identifier for data isolation |

All graph queries filter by group_id to ensure tenant separation.

---

## Appendix: Data Flow Summary

```
Question (string)
    |
    v
[Phase 1: Decomposition]
    |
    +---> QueryDecomposition
    |         |
    |         +---> sub_queries: list[SubQuery]
    |         +---> entity_hints: list[EntityHint]
    |         +---> topic_hints: list[EntityHint]
    |         +---> question_type: QuestionType
    |         +---> temporal_scope: Optional[str]
    |
    v
[Phase 2: Parallel Research] (one Researcher per sub_query)
    |
    +---> Resolution
    |         +---> resolve_entities(hints) -> list[ResolvedEntity]
    |         +---> resolve_topics(hints) -> list[ResolvedTopic]
    |
    +---> Retrieval (parallel)
    |         +---> get_entity_chunks() -> list[RetrievedChunk]
    |         +---> get_entity_facts() -> list[RetrievedFact]
    |         +---> get_1hop_neighbors() -> list[RetrievedEntity]
    |         +---> get_neighbor_chunks() -> list[RetrievedChunk]
    |         +---> get_topic_chunks() -> list[RetrievedChunk]
    |         +---> global_chunk_search() -> list[RetrievedChunk]
    |
    +---> Context Assembly
    |         +---> ContextBuilder.build() -> StructuredContext
    |
    +---> Sub-Answer Synthesis
              +---> SubAnswer (per sub_query)
    |
    v
[Phase 3: Final Synthesis]
    |
    +---> Merge SubAnswers + Question Type Instructions
    |
    v
PipelineResult
    +---> answer: string
    +---> confidence: float
    +---> sub_answers: list[SubAnswer]
    +---> question_type: string
    +---> timing breakdown (decomposition, resolution, retrieval, synthesis)
```

---

## Appendix: Key Type Definitions

### ResolvedEntity
- original_hint: The hint from decomposition
- resolved_name: Canonical name in the graph
- summary: Entity summary from the graph
- confidence: Resolution confidence (1.0 for LLM-verified)

### ResolvedTopic
- original_hint: The hint from decomposition
- resolved_name: Canonical topic name
- definition: Topic definition
- confidence: Resolution confidence

### RetrievedChunk
- chunk_id: Unique identifier
- content: Chunk text content
- header_path: Hierarchical path for context
- doc_id: Source document identifier
- document_date: Document timestamp
- vector_score: Similarity score
- source: Retrieval source (e.g., "entity:Apple", "topic:M&A", "global")

### RetrievedFact
- fact_id: Unique identifier
- content: Fact text content
- subject: Subject entity name
- edge_type: Relationship type
- object: Object entity name
- chunk_id: Source chunk for provenance
- vector_score: Similarity score

### SubAnswer
- sub_query: The sub-question text
- target_info: What this sub-query sought
- answer: Synthesized answer
- confidence: 0-1 quality signal
- context: StructuredContext used
- entities_found: Entities mentioned in answer
- timing: resolution_time_ms, retrieval_time_ms, synthesis_time_ms

### PipelineResult
- question: Original question
- answer: Final synthesized answer
- confidence: Overall confidence
- sub_answers: List of SubAnswer objects
- question_type: Classification string
- timing: decomposition_time_ms, resolution_time_ms, retrieval_time_ms, synthesis_time_ms
