# Entity and Topic Extraction System

## Phase 1 of the ZommaKG Ingestion Pipeline

---

## 1. Executive Summary

The Entity and Topic Extraction system is the first phase of the ZommaKG knowledge graph ingestion pipeline. It transforms unstructured financial text into structured facts that flow into the embedded Parquet + DuckDB + LanceDB runtime.

### Core Philosophy: Chain-of-Thought Extraction

The extraction system uses a **two-step chain-of-thought approach** that forces the LLM to think systematically:

1. **Entity Enumeration First**: Before generating any relationships, the LLM must explicitly list every entity in the text
2. **Constrained Relationship Generation**: Relationships can only be formed between entities that were enumerated in step one

This architecture addresses a common failure mode where single-pass extractors miss entities that appear in complex sentences or subordinate clauses. By separating enumeration from relationship generation, the system ensures comprehensive entity coverage.

### Hypothesis Behind the Design

The chain-of-thought approach was developed to solve the "Chronicle Security problem" - a real-world case where single-pass extractors failed to identify a subsidiary mentioned in passing within a larger sentence about its parent company. By forcing explicit enumeration, the system catches entities that would otherwise be overlooked.

---

## 2. Two-Step Extraction Process

### Step 1: Entity Enumeration

The first step requires the LLM to scan the entire text chunk and list **every** named entity it finds. This includes entities from both the chunk body and the section header (which provides hierarchical context).

**What the LLM produces in Step 1:**

For each entity found, the LLM outputs:
- **name**: The exact name as it appears in the text
- **entity_type**: One of the six canonical types (see Section 3)
- **summary**: A 1-2 sentence description based on context

**Critical Instruction**: The LLM is explicitly told to think like a portfolio manager and ask: "What themes in this text would a trader want to monitor or filter by?"

### Step 2: Relationship Generation

Only after completing entity enumeration does the LLM proceed to generate relationships. This step has a hard constraint: **subjects and objects must come from the Step 1 entity list**.

**What the LLM produces in Step 2:**

For each relationship, the LLM outputs an `ExtractedFact` containing:
- The fact text (a self-contained proposition)
- Subject and object entities (must exist in the enumeration)
- A free-form relationship description
- Temporal context (required for every fact)
- Associated topics

The two-step structure within a single LLM call ensures the model cannot "forget" entities discovered in step one, as the structured output schema enforces both sections.

---

## 3. Entity Types and Definitions

The system recognizes six canonical entity types. Each serves a distinct purpose in the knowledge graph.

### Company

Represents corporate entities, including publicly traded companies, private companies, and startups.

**Critical Rule - Subsidiary Awareness**: Subsidiaries are treated as **separate entities** from their parent companies. AWS is not Amazon; YouTube is not Alphabet. This distinction is essential for accurate relationship tracking and prevents conflation of subsidiary-level events with parent company facts.

**Examples**: Apple Inc., AWS, Google Cloud, Tesla

### Person

Represents named individuals mentioned in financial documents.

**Extraction Rule**: Always use the full name as it appears in the text. Do not standardize or infer missing name components.

**Examples**: Sundar Pichai, Warren Buffett, Janet Yellen

### Organization

Represents non-corporate institutions and bodies. This includes government agencies, regulatory bodies, international organizations, and standards bodies.

**Distinction from Company**: Organizations are typically non-profit or governmental entities. The Federal Reserve is an Organization; JPMorgan Chase is a Company.

**Examples**: Federal Reserve, SEC, World Bank, United Nations, NATO

### Location

Represents geographic entities at any scale - countries, states, cities, or regions.

**Usage**: Locations are extracted when they are meaningful to the financial context, not merely incidental mentions.

**Examples**: United States, Silicon Valley, European Union, China

### Product

Represents specific products, services, or offerings from companies.

**Examples**: iPhone, ChatGPT, Azure OpenAI Service, Model S

### Topic

Represents abstract concepts, metrics, themes, and categories that are not physical entities but are essential for knowledge graph queries.

**Critical Function**: Topics enable semantic filtering and discovery. A portfolio manager searching for "all facts related to inflation" needs Topic nodes to find relevant facts across different companies and time periods.

**Topic Categories**:
- Financial metrics: Revenue, Market Valuation, EBITDA, Market Share
- Economic concepts: Inflation, Interest Rates, GDP Growth
- Business themes: M&A, IPO, Restructuring, Layoffs
- Policy areas: Trade Policy, Antitrust Regulation, Climate Policy
- Sector categories: Technology, Healthcare, Energy

---

## 4. Extraction Rules and Constraints

### Clean Entity Names

Subject and object fields must contain **clean entity names only** - no descriptors, qualifiers, adjectives, or contextual information.

**Correct**: "Apple" as subject, "Revenue" as object
**Incorrect**: "Apple Inc. (technology company)" or "Apple's Q3 2024 Revenue"

Descriptors and qualifiers belong in the relationship field or the fact text, not in entity names. This ensures consistent entity resolution and deduplication downstream.

### No Parentheses Rule

Entity names and relationship descriptions must not contain parentheses. Any parenthetical information should be incorporated into the fact text or omitted if redundant.

### Date Context Requirement (REQUIRED)

**Every extracted fact must have date_context populated. This field is never optional.**

The system uses a fallback hierarchy:
1. **Specific date from text**: If the fact mentions a specific date, quarter, or time period ("Q3 2024", "August 5, 2024"), use that
2. **Document date fallback**: If no specific date is mentioned, use "Document date: YYYY-MM-DD"

Date context enables temporal queries, allowing users to ask questions like "What happened to Apple in Q3 2024?" or "Show me all M&A activity from the past year."

### Subsidiary Awareness

The system explicitly distinguishes subsidiaries from parent companies:
- AWS and Amazon are separate entities
- Google Cloud and Alphabet are separate entities
- Instagram and Meta are separate entities

Facts about AWS should link to the AWS entity node, not to Amazon. This preserves accuracy and enables subsidiary-specific queries.

### What to Exclude

The following should **NOT** be extracted as entities:

**URLs and Links**
- https://www.example.com
- www.company.com

**Specific Numeric Values**
- $1.5 billion (goes in fact text only)
- 25% increase (goes in fact text only)
- #3 ranking (goes in fact text only)

**Dates and Time Periods**
- 2024, Q3 2024, August 5, 2024 (goes in date_context field)

**Citation Metadata**
- Article titles from reference sections
- Archive URLs
- Retrieval dates
- "(PDF)" markers
- Author names from citations (unless the author is a subject of the main text)

**Page Titles**
- "Company Name on Forbes"
- "Company Name on Wikipedia"

**Bibliographic References**
- Any text that is clearly reference or citation metadata

---

## 5. Fact Structure

Each extracted fact follows a standardized schema designed for knowledge graph ingestion.

### fact (text)

A complete, self-contained proposition that can be understood without external context.

**Requirements**:
- No pronouns without antecedents (use full entity names)
- Include specific values, percentages, and dates
- Should be a single atomic statement

**Good Example**: "Apple reported revenue of $94.8 billion in Q3 2024, exceeding analyst expectations of $92.5 billion."

**Bad Example**: "They beat expectations." (missing subject, values, and context)

### subject and subject_type

The primary entity performing the action or being described.

- **subject**: Clean entity name (e.g., "Apple")
- **subject_type**: One of Company, Person, Organization, Location, Product, Topic

### object and object_type

The entity being acted upon or related to the subject.

- **object**: Clean entity name (e.g., "Revenue")
- **object_type**: One of Company, Person, Organization, Location, Product, Topic

### relationship

A free-form description of how the subject relates to the object.

**Key Design Decision**: Relationships are NOT enumerated. The system uses free-form text descriptions to preserve nuance that would be lost with a fixed taxonomy.

**Examples**:
- "acquired"
- "partnered with"
- "reported increase in"
- "filed lawsuit against"
- "reached milestone"

The relationship field can include qualifiers that would be inappropriate in entity names: "reported strong growth in" rather than just "reported".

### date_context

Temporal context for the fact. **This field is required for every fact.**

**Format Options**:
- Specific: "Q3 2024", "August 5, 2024", "FY2023"
- Fallback: "Document date: 2024-08-15"

### topics

A list of topic strings that categorize the fact for semantic retrieval.

**Examples**: ["Earnings", "Revenue Growth", "Technology Sector"]

Topics enable cross-cutting queries like "Show me all facts about inflation across all companies."

---

## 6. Critique/Reflexion System

The extraction pipeline includes a **critique step** that reviews extraction results before finalizing them. This implements a reflexion pattern where the system evaluates its own output.

### Purpose of the Critique Step

The critique system serves as quality control, catching errors that the initial extraction may have made. It operates as a "senior analyst" reviewing a junior analyst's work.

### Critique Checklist

The critic evaluates extractions against seven criteria:

**1. Entity Coverage**
Were ALL named entities captured in the enumeration? This includes:
- Every company, person, and organization mentioned
- Subsidiaries, divisions, and products
- All Topic entities that a portfolio manager would care about

**2. Sentence-Level Completeness (Critical)**
This is the most important check. The critic goes through **each sentence** in the original text and verifies:
- Does every clause have a corresponding extracted fact?
- Compound sentences often contain multiple facts
- Example: "Revenue increased while expenses decreased" requires TWO facts

**3. Relationship Completeness**
Did we capture all material interactions?
- Every action, acquisition, partnership, filing
- Every financial metric or event
- Every relationship between entities

**4. Accuracy**
Are extractions correct?
- Entity names match the text exactly
- Relationships accurately describe the interaction
- No hallucinated information

**5. Self-Containment**
Can each fact be understood alone?
- No pronouns without antecedents
- Complete propositions with all necessary context

**6. Clean Structure**
Are subjects, objects, and relationships properly formed?
- Entity names are clean (no descriptors)
- Numbers and values appear only in fact text
- No parentheses in entity or relationship names

**7. Exclusion Check**
Flag any entities that should NOT have been extracted:
- URLs
- Citation metadata
- Page titles
- Bibliographic references

### Critique Output

The critique produces a structured result:
- **is_approved**: Boolean - true if extraction passes all checks
- **critique**: Text description of issues found
- **missed_facts**: List of facts that should have been extracted but were not
- **corrections**: Specific corrections to entity names, types, or relationships

### Re-extraction on Failure

If the critique identifies issues (is_approved = false), the system triggers a **re-extraction**. The re-extraction prompt includes:
- The original text
- The critique feedback
- Specific corrections needed
- List of missed facts to add

The re-extraction is bounded to a single retry to prevent infinite loops. If re-extraction also fails, the system proceeds with the best available result.

---

## 7. Topic Extraction Strategy

### Portfolio Manager Perspective

Topic extraction is guided by a simple heuristic: **"What would a portfolio manager or trader want to filter by?"**

The system extracts topics for:
- Government policies, regulations, or political factors affecting markets
- Macroeconomic conditions, trends, or indicators
- Sectors or industries being impacted
- Business events and corporate actions

### Cause and Effect Topics

When text describes causal relationships, the system extracts **both** cause topics and effect topics.

**Example**: If a policy causes labor shortages, extract:
- The policy topic (e.g., "Immigration Policy")
- The effect topic (e.g., "Labor Market")

This enables queries from either direction - finding all effects of a policy, or finding all causes of a labor market condition.

### Numeric Metrics Handling

When text mentions specific numbers about an entity, the system follows a specific pattern:

- **Subject**: The entity being measured (e.g., "Apple")
- **Object**: The metric CATEGORY as a Topic (e.g., "Market Valuation") - NOT the specific number
- **Relationship**: Normalized verb describing the action (e.g., "reached milestone in")
- **Fact text**: The complete statement including specific values

**Example Text**: "Apple's market value reached $3 trillion."

**Extraction**:
- Subject: "Apple" (Company)
- Object: "Market Valuation" (Topic)
- Relationship: "reached milestone in"
- Fact: "Apple's market value reached $3 trillion."

This pattern ensures that numeric values are preserved in searchable fact text while enabling semantic queries through Topic nodes.

---

## 8. Prompt Engineering Details

### System Prompt Structure

The extraction system prompt follows a structured format:

**1. Role Definition**
Establishes the persona: "You are a financial analyst building a knowledge graph."

**2. Two-Step Instructions**
Clearly delineates Step 1 (Entity Enumeration) and Step 2 (Relationship Generation).

**3. Entity Type Definitions**
Lists the six entity types with examples.

**4. Topic Extraction Guidance**
Includes the "portfolio manager perspective" heuristic and cause/effect instructions.

**5. Numeric Handling Rules**
Explicit instructions for separating metric values from metric categories.

**6. Date Context Requirements**
Emphasizes that date_context is required for every fact.

**7. Clean Structure Rules**
Lists all the formatting requirements for entity names and relationships.

**8. Exclusion List**
Explicit list of what NOT to extract.

### User Prompt Template

The user prompt provides chunk-specific context:

```
DOCUMENT CONTEXT:
Document Date: {document_date}

CHUNK TEXT:
Section: {header_path}

{chunk_text}

First enumerate ALL entities (including any entities mentioned in the Section header -
the ">" denotes subheader hierarchy), then generate relationships between them.
Remember: ALL facts MUST have date_context. Use "{document_date}" as fallback if no
specific date in text.
```

The section header uses a ">" delimiter to show hierarchy (e.g., "Financials > Revenue > Q3 2024").

### Structured Output Enforcement

The system uses LangChain's `with_structured_output()` to enforce schema compliance. The LLM must produce output conforming to the `ChainOfThoughtResult` Pydantic model, which contains:
- `entities`: List[EnumeratedEntity]
- `facts`: List[ExtractedFact]

Structured output eliminates parsing errors and ensures all required fields are present.

### Few-Shot Examples

The current implementation does not include few-shot examples in the prompt. The detailed instructions and schema definitions serve as sufficient guidance. However, the critique system provides implicit few-shot learning through its feedback loop.

---

## 9. Concurrency and Batching

### Async Parallel Extraction

Document chunks are processed in parallel using Python's asyncio. The pipeline creates extraction tasks for all chunks and runs them concurrently.

```
tasks = [extract_chunk(i, chunk) for i, chunk in enumerate(chunks)]
results = await asyncio.gather(*tasks)
```

This parallelization dramatically reduces total extraction time for large documents.

### Semaphore-Based Rate Limiting

To prevent overwhelming the LLM API, extraction uses an asyncio Semaphore to limit concurrent requests:

```
sem = asyncio.Semaphore(concurrency)

async def extract_chunk(chunk_idx, chunk):
    async with sem:
        # Extraction happens here
        result = await asyncio.to_thread(extractor.extract, ...)
```

The default concurrency is 5, configurable via the `concurrency` parameter.

### Derived Concurrency Settings

The single `concurrency` parameter controls multiple parallel operations:
- **extraction_concurrency**: Equal to `concurrency` (default: 5)
- **embedding_concurrency**: `concurrency // 2` with minimum of 2
- **resolve_concurrency**: `concurrency * 10`
- **dedup_concurrency**: `concurrency * 4`
- **embedding_batch_size**: Fixed at 100

### Error Handling Per Chunk

Each chunk extraction is wrapped in try/except. If a chunk fails:
- The error is logged
- An empty ChainOfThoughtResult is returned for that chunk
- Other chunks continue processing

This isolation prevents single-chunk failures from stopping the entire pipeline.

### Thread Pool for Synchronous LLM Calls

Since many LLM clients are synchronous, the system uses `asyncio.to_thread()` to run extractions in a thread pool without blocking the event loop:

```
result = await asyncio.to_thread(
    extractor.extract,
    chunk_text=text,
    header_path=header_path,
    document_date=document_date
)
```

### LLM Call Budget Per Chunk

The system makes 2-3 LLM calls per chunk:
1. **Extraction call**: Initial chain-of-thought extraction (always)
2. **Critique call**: Review of extraction results (always, unless empty result)
3. **Re-extraction call**: Only if critique finds issues (0-1 times)

With max_retries=2 for parsing errors, the worst case is 4-6 LLM calls per chunk, but typical operation is 2-3.

---

## 10. Integration with Pipeline Phases

Entity and Topic Extraction (Phase 1) produces raw extraction results that flow into subsequent pipeline phases:

**Phase 2a-c: Entity Deduplication**
- Uses embedding-based clustering to identify duplicate entity mentions
- LLM verification confirms matches
- Produces canonical UUIDs for each distinct entity

**Phase 2d-e: Resolution**
- Entities are resolved against existing KB entities via embedded vector indices
- Topics are resolved against the curated topic ontology
- New entities/topics get UUIDs; existing ones get matched

**Phase 3: Assembly**
- Resolved entities and facts are written to Parquet storage
- Embeddings are generated for semantic search
- Relationships are created between nodes

The extraction phase's output quality directly impacts downstream phases. Clean entity names enable accurate deduplication. Complete fact coverage ensures no information is lost. Proper topic extraction enables rich semantic queries.

---

## Appendix: Schema Definitions

### EnumeratedEntity

```
name: str         # Entity name as it appears in text
entity_type: str  # Company, Person, Organization, Location, Product, Topic
summary: str      # 1-2 sentence description (optional)
```

### ExtractedFact

```
fact: str            # Complete, self-contained proposition
subject: str         # Primary entity name
subject_type: str    # Entity type
subject_summary: str # Description of subject (optional)
object: str          # Related entity name
object_type: str     # Entity type
object_summary: str  # Description of object (optional)
relationship: str    # Free-form relationship description
date_context: str    # Temporal context (REQUIRED)
topics: List[str]    # Related topic names
```

### ChainOfThoughtResult

```
entities: List[EnumeratedEntity]  # Step 1 output
facts: List[ExtractedFact]        # Step 2 output
```

### CritiqueResult

```
is_approved: bool        # True if extraction passes review
critique: str            # Description of issues (optional)
missed_facts: List[str]  # Facts that should have been extracted
corrections: List[str]   # Specific corrections needed
```
