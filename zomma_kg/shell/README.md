# Filesystem Navigation Architecture for Knowledge Graph Exploration

## Executive Summary

This document describes a novel architecture for AI agent interaction with knowledge graphs using familiar Unix filesystem commands. Rather than requiring agents to construct complex queries, this approach presents the knowledge graph as a virtual filesystem that can be navigated using standard CLI commands: `ls`, `cd`, `cat`, `grep`, `find`, `head`, `tail`, `wc`, and `pwd`.

### The Core Insight

Large Language Models are extensively trained on CLI interactions. Terminal sessions comprise a significant portion of training data, meaning LLMs have deep, intuitive understanding of:

- Directory navigation patterns (`cd`, `pwd`, relative vs absolute paths)
- Listing and filtering (`ls`, `ls -la`, `ls *.txt`)
- Content retrieval (`cat`, `head`, `tail`)
- Search patterns (`grep`, `find`, regex)
- Pagination and counting (`head -n 10`, `wc -l`)
- Piping and composition (`ls | grep pattern | head`)

By mapping knowledge graph operations to these familiar commands, we create an interface that LLMs can use naturally without special training. The agent explores iteratively, deciding where to navigate next based on what it discovers, rather than constructing a single complex query upfront.

### Key Benefits

1. **Zero-shot usability**: Any LLM can immediately use the interface
2. **Iterative exploration**: Agent discovers and follows leads naturally
3. **Intuitive semantics**: Commands mean what LLMs expect them to mean
4. **Local execution**: All queries translate to fast DuckDB/LanceDB operations
5. **Context building**: Navigation history builds understanding progressively

---

## Design Philosophy

### Why Filesystem Metaphors Work for LLMs

The filesystem metaphor succeeds because it maps abstract graph operations to concrete, well-understood patterns:

| Graph Concept | Filesystem Metaphor | LLM Understanding |
|---------------|---------------------|-------------------|
| Entity collection | Directory | "Contains things I can list" |
| Entity instance | File or subdirectory | "Has content I can read" |
| Entity properties | File contents | "Information about this thing" |
| Relationships | Symbolic links / cross-references | "Points to related things" |
| Topics/Categories | Directory hierarchy | "Organizational structure" |
| Search | grep/find | "Find things matching criteria" |

LLMs understand these metaphors deeply because:

1. **Training prevalence**: Terminal sessions are heavily represented in training data
2. **Consistent semantics**: CLI commands have stable, predictable behavior
3. **Compositional**: Commands can be combined (pipes, redirects)
4. **Self-documenting**: Command structure implies capability (`ls` = list, `cd` = change directory)

### Iterative vs Single-Query Exploration

Traditional knowledge graph interfaces require the agent to construct a complete query upfront:

```
"Find all entities of type Organization that have a relationship ACQUIRED
 with another Organization where the acquisition occurred after 2023 and
 the target organization has topic 'Artificial Intelligence'"
```

This approach has significant problems:

1. **Requires full knowledge of schema**: Agent must know node types, relationship types, property names
2. **Brittle**: Small errors invalidate the entire query
3. **No discovery**: Agent cannot explore what exists before querying
4. **High cognitive load**: Complex queries are hard to construct and debug

The filesystem approach enables iterative exploration:

```bash
$ cd /kg/entities/organizations/
$ ls
Alphabet_Inc/  Apple_Inc/  Microsoft/  ...

$ cd Alphabet_Inc/
$ ls
summary.txt  facts/  relationships/  topics/  chunks/

$ cat summary.txt
Alphabet Inc. is a multinational technology conglomerate...

$ ls relationships/
ACQUIRED/  PARTNERED_WITH/  INVESTED_IN/  ...

$ ls relationships/ACQUIRED/
DeepMind/  Fitbit/  Mandiant/  ...

$ cat relationships/ACQUIRED/DeepMind/
Acquisition date: 2014
Price: $500M
Description: Acquired UK AI research lab...
```

This iterative approach:

1. **Progressive discovery**: Agent learns schema by exploring
2. **Self-correcting**: Wrong paths are easily abandoned
3. **Context accumulation**: Each step adds understanding
4. **Natural reasoning**: Agent thinks "I should look at acquisitions" not "construct MATCH query"

### Agent Autonomy in Navigation

The filesystem interface gives agents genuine autonomy in exploration:

1. **The agent decides what to explore**: Not limited to predefined query templates
2. **Exploration adapts to findings**: Discovery of unexpected connections changes the path
3. **Context drives decisions**: Current location influences what makes sense to explore next
4. **Backtracking is natural**: `cd ..` or `cd /kg/entities/` to change direction

This autonomy is crucial for complex research tasks where the agent needs to:
- Follow chains of relationships
- Compare entities across categories
- Discover unexpected connections
- Build understanding incrementally

---

## Virtual Filesystem Structure

The knowledge graph is presented as a hierarchical filesystem with multiple root directories optimized for different exploration patterns.

### Root Structure

```
/kg/
├── entities/           # Entity-centric exploration
│   ├── organizations/
│   ├── people/
│   ├── locations/
│   └── concepts/
├── topics/             # Topic/theme-centric exploration
│   ├── inflation/
│   ├── labor_markets/
│   └── economic_growth/
├── chunks/             # Source text exploration
│   └── by_document/
├── documents/          # Document-centric exploration
│   └── [document_id]/
├── facts/              # Fact-centric exploration
│   └── by_type/
└── @search/            # Special search endpoints
    ├── semantic/
    ├── entities/
    └── facts/
```

### /kg/entities/ Hierarchy

The entities directory organizes all EntityNodes by type, enabling type-scoped exploration.

```
/kg/entities/
├── organizations/
│   ├── Alphabet_Inc/
│   │   ├── summary.txt              # Entity summary text
│   │   ├── metadata.json            # Type, aliases, identifiers
│   │   ├── facts/                   # Facts where entity is subject
│   │   │   ├── by_type/
│   │   │   │   ├── ACQUIRED/
│   │   │   │   ├── PARTNERED_WITH/
│   │   │   │   └── REPORTED_FINANCIALS/
│   │   │   └── all.jsonl            # All facts (one per line)
│   │   ├── mentioned_in/            # Facts where entity is object
│   │   │   └── ...
│   │   ├── relationships/           # Outgoing edges
│   │   │   ├── ACQUIRED/ -> [entities]
│   │   │   ├── SUBSIDIARY_OF/ -> [entities]
│   │   │   └── ...
│   │   ├── related_to/              # Incoming edges (reverse relationships)
│   │   │   ├── ACQUIRED_BY/ -> [entities]
│   │   │   └── ...
│   │   ├── topics/                  # Associated topics
│   │   │   └── [topic_name] -> /kg/topics/[topic]
│   │   └── chunks/                  # Source chunks mentioning entity
│   │       └── [chunk_id] -> /kg/chunks/[id]
│   ├── Apple_Inc/
│   │   └── ...
│   └── _index.txt                   # Alphabetical listing
├── people/
│   ├── Sundar_Pichai/
│   │   ├── summary.txt
│   │   ├── roles/                   # Special: positions held
│   │   │   └── CEO_of/ -> [organizations]
│   │   └── ...
│   └── ...
├── locations/
│   ├── Boston/
│   │   ├── summary.txt
│   │   ├── economic_data/           # Special: regional data
│   │   └── ...
│   └── ...
└── concepts/
    └── ...
```

**Key Design Decisions**:

1. **Directories per entity**: Each entity gets its own namespace for organizing related information
2. **Relationship subdirectories**: Outgoing relationships grouped by type
3. **Bidirectional visibility**: Both `relationships/` (outgoing) and `related_to/` (incoming)
4. **Symlinks for cross-references**: Topics and chunks link to canonical locations
5. **Special files**: `summary.txt` always contains human-readable description

### /kg/topics/ Hierarchy

Topics provide thematic entry points into the knowledge graph.

```
/kg/topics/
├── inflation/
│   ├── definition.txt               # Topic definition
│   ├── subtopics/                   # Hierarchical topics
│   │   ├── wage_inflation/
│   │   └── price_inflation/
│   ├── entities/                    # Entities discussing this topic
│   │   └── [entity_name] -> /kg/entities/...
│   ├── chunks/                      # Chunks tagged with topic
│   │   └── [chunk_id] -> /kg/chunks/...
│   └── facts/                       # Facts related to topic
│       └── [fact_id].json
├── labor_markets/
│   ├── definition.txt
│   ├── subtopics/
│   │   ├── unemployment/
│   │   ├── hiring/
│   │   └── wages/
│   └── ...
├── economic_growth/
│   └── ...
└── _ontology.json                   # Full topic hierarchy
```

**Key Design Decisions**:

1. **Hierarchical topics**: Subtopics as subdirectories enable drill-down
2. **Bidirectional links**: Topics link to entities, entities link to topics
3. **Chunk aggregation**: All source text for a topic in one place
4. **Definition files**: Clear, searchable topic descriptions

### /kg/chunks/ Hierarchy

Chunks are the source text units providing provenance for all extracted information.

```
/kg/chunks/
├── by_document/
│   ├── beigebook_2024_10/
│   │   ├── chunk_0001/
│   │   │   ├── content.txt          # Raw chunk text
│   │   │   ├── metadata.json        # header_path, position, dates
│   │   │   ├── entities/            # Entities mentioned
│   │   │   │   └── [entity] -> /kg/entities/...
│   │   │   ├── facts/               # Facts extracted from chunk
│   │   │   │   └── [fact_id].json
│   │   │   └── topics/              # Topics discussed
│   │   │       └── [topic] -> /kg/topics/...
│   │   ├── chunk_0002/
│   │   │   └── ...
│   │   └── _toc.txt                 # Table of contents (header paths)
│   └── earnings_call_goog_q3/
│       └── ...
├── by_topic/                        # Alternative organization
│   └── [topic_name]/
│       └── [chunk_id] -> by_document/...
└── by_date/                         # Temporal organization
    └── 2024/
        └── 10/
            └── [chunk_id] -> by_document/...
```

**Key Design Decisions**:

1. **Multiple organization schemes**: by_document, by_topic, by_date for different access patterns
2. **Symlinks for deduplication**: One canonical location, multiple entry points
3. **Table of contents**: Quick navigation aid for documents
4. **Provenance tracking**: Clear path from extracted data back to source

### /kg/documents/ Hierarchy

Documents provide the top-level organizational structure.

```
/kg/documents/
├── beigebook_2024_10/
│   ├── metadata.json                # Title, date, type, source
│   ├── summary.txt                  # Document-level summary
│   ├── chunks/                      # All chunks in order
│   │   ├── 0001/ -> /kg/chunks/by_document/.../chunk_0001
│   │   ├── 0002/ -> /kg/chunks/by_document/.../chunk_0002
│   │   └── ...
│   ├── entities/                    # All entities mentioned
│   │   └── [entity_name] -> /kg/entities/...
│   ├── topics/                      # All topics covered
│   │   └── [topic_name] -> /kg/topics/...
│   └── structure.txt                # Header hierarchy outline
├── earnings_call_goog_q3/
│   └── ...
└── _index.json                      # Document listing with metadata
```

### @search Special Endpoints

The `@search` directory provides query-like functionality through filesystem semantics.

```
/kg/@search/
├── semantic/
│   └── "what caused inflation in Boston"    # Query as path
│       └── results.jsonl                     # Semantic search results
├── entities/
│   └── "Federal Reserve"/                   # Entity name search
│       └── matches.jsonl                     # Matching entities
├── facts/
│   └── "interest rate changes"/             # Fact content search
│       └── results.jsonl                     # Matching facts
└── chunks/
    └── "labor market conditions"/           # Chunk content search
        └── results.jsonl                     # Matching chunks
```

**Usage Pattern**:
```bash
$ cat "/kg/@search/semantic/what economic trends affected Boston/results.jsonl"
{"type": "chunk", "score": 0.92, "path": "/kg/chunks/by_document/beigebook.../chunk_0042"}
{"type": "fact", "score": 0.89, "path": "/kg/entities/Boston/facts/..."}
...
```

The `@search` endpoints enable hybrid exploration:
1. Start with semantic search to find entry points
2. Navigate from results to explore context
3. Use findings to formulate more specific searches

---

## Command Semantics

Each CLI command maps to intuitive knowledge graph operations.

### ls - List Entities, Facts, Connections

The `ls` command lists contents of the current context, with options for filtering and detail.

**Basic Usage**:
```bash
$ ls /kg/entities/organizations/
Alphabet_Inc/    Apple_Inc/    Amazon/    Microsoft/    ...

$ ls /kg/entities/organizations/Alphabet_Inc/
summary.txt    metadata.json    facts/    relationships/    topics/    chunks/

$ ls /kg/entities/organizations/Alphabet_Inc/relationships/
ACQUIRED/    PARTNERED_WITH/    INVESTED_IN/    SUBSIDIARY_OF/
```

**Options**:
```bash
$ ls -l /kg/entities/organizations/
drwxr-xr-x  Alphabet_Inc/     # 47 facts, 12 relationships, 23 chunks
drwxr-xr-x  Apple_Inc/        # 38 facts, 9 relationships, 18 chunks
drwxr-xr-x  Amazon/           # 52 facts, 15 relationships, 31 chunks
...

$ ls -la /kg/entities/organizations/Alphabet_Inc/relationships/ACQUIRED/
total 15
drwxr-xr-x  DeepMind/         -> /kg/entities/organizations/DeepMind/
drwxr-xr-x  Fitbit/           -> /kg/entities/organizations/Fitbit/
drwxr-xr-x  Mandiant/         -> /kg/entities/organizations/Mandiant/
...

$ ls -t /kg/chunks/by_date/2024/     # Sort by time
10/    09/    08/    07/    ...

$ ls -S /kg/entities/organizations/  # Sort by size (fact count)
Amazon/    Alphabet_Inc/    Microsoft/    Apple_Inc/    ...
```

**Filtering**:
```bash
$ ls /kg/entities/organizations/A*/
Alphabet_Inc/    Apple_Inc/    Amazon/    Adobe/    ...

$ ls /kg/topics/*/subtopics/
inflation/subtopics:
wage_inflation/    price_inflation/    ...

labor_markets/subtopics:
unemployment/    hiring/    wages/    ...
```

### cd - Navigate Context

The `cd` command changes the current exploration context.

**Basic Navigation**:
```bash
$ pwd
/kg/

$ cd entities/organizations/Alphabet_Inc/
$ pwd
/kg/entities/organizations/Alphabet_Inc/

$ cd relationships/ACQUIRED/DeepMind/
$ pwd
/kg/entities/organizations/Alphabet_Inc/relationships/ACQUIRED/DeepMind/

$ cd ..
$ pwd
/kg/entities/organizations/Alphabet_Inc/relationships/ACQUIRED/

$ cd /kg/topics/inflation/
$ pwd
/kg/topics/inflation/

$ cd -    # Return to previous directory
$ pwd
/kg/entities/organizations/Alphabet_Inc/relationships/ACQUIRED/
```

**Symlink Resolution**:
```bash
$ cd /kg/entities/organizations/Alphabet_Inc/topics/artificial_intelligence/
$ pwd
/kg/topics/artificial_intelligence/    # Symlink resolved to canonical path
```

**Context Implications**:
When inside an entity directory, commands are scoped:
```bash
$ cd /kg/entities/organizations/Alphabet_Inc/
$ ls facts/            # Facts about Alphabet
$ grep "acquisition"   # Search within Alphabet's context
```

### cat - Read Content

The `cat` command retrieves content from files.

**Basic Usage**:
```bash
$ cat /kg/entities/organizations/Alphabet_Inc/summary.txt
Alphabet Inc. is a multinational technology conglomerate headquartered in
Mountain View, California. It was created through a restructuring of Google
on October 2, 2015, and became the parent company of Google and several
former Google subsidiaries...

$ cat /kg/chunks/by_document/beigebook_2024_10/chunk_0042/content.txt
Labor market conditions in the First District remained tight but showed
signs of easing. Employers reported difficulty filling positions in
healthcare, hospitality, and skilled trades. Wage growth moderated
compared to earlier in the year but remained elevated...

$ cat /kg/entities/organizations/Alphabet_Inc/metadata.json
{
  "type": "Organization",
  "aliases": ["Google", "Alphabet", "GOOGL"],
  "founded": "2015-10-02",
  "headquarters": "Mountain View, California",
  "sector": "Technology"
}
```

**Multiple Files**:
```bash
$ cat /kg/entities/organizations/Alphabet_Inc/facts/all.jsonl
{"fact_id": "f001", "type": "ACQUIRED", "object": "DeepMind", "date": "2014", "description": "..."}
{"fact_id": "f002", "type": "REPORTED_FINANCIALS", "date": "2024-Q3", "description": "..."}
...
```

### grep - Semantic Search

The `grep` command performs semantic and keyword search, combining vector similarity with traditional pattern matching.

**Semantic Search**:
```bash
$ grep "economic impact of interest rate changes" /kg/chunks/
/kg/chunks/by_document/beigebook.../chunk_0015/content.txt: [0.94] Rising interest...
/kg/chunks/by_document/fed_minutes.../chunk_0042/content.txt: [0.91] The committee...
...

$ grep "companies that were acquired" /kg/entities/
/kg/entities/organizations/DeepMind/summary.txt: [0.89] DeepMind was acquired...
/kg/entities/organizations/Fitbit/summary.txt: [0.87] Fitbit, acquired by...
...
```

**Keyword Search**:
```bash
$ grep -F "Federal Reserve" /kg/chunks/        # Literal string match
/kg/chunks/by_document/beigebook.../chunk_0001/content.txt: The Federal Reserve...
/kg/chunks/by_document/beigebook.../chunk_0023/content.txt: ...Federal Reserve Bank...
...

$ grep -E "interest rate[s]?" /kg/facts/       # Regex pattern
/kg/facts/by_type/RAISED_POLICY_RATE/f_001.json: raised interest rates
/kg/facts/by_type/LOWERED_POLICY_RATE/f_042.json: cut interest rate
...
```

**Options**:
```bash
$ grep -i "INFLATION" /kg/topics/              # Case insensitive
$ grep -c "Boston" /kg/chunks/                 # Count matches only
$ grep -l "unemployment" /kg/chunks/           # List files only
$ grep -v "2023" /kg/facts/                    # Invert match
$ grep -A 2 -B 1 "wage growth" /kg/chunks/     # Context lines
```

**Scoped Search**:
```bash
$ cd /kg/entities/organizations/Alphabet_Inc/
$ grep "artificial intelligence"               # Search within entity context
facts/by_type/INVESTED_IN/f_042.json: invested in AI research...
chunks/chunk_0015.txt: ...artificial intelligence initiatives...
```

### find - Filtered Search with Patterns

The `find` command locates files matching specific criteria.

**By Name Pattern**:
```bash
$ find /kg/entities/ -name "*Google*"
/kg/entities/organizations/Google_Cloud/
/kg/entities/organizations/Google_DeepMind/
/kg/entities/people/Google_employees/

$ find /kg/chunks/ -name "*.txt"
/kg/chunks/by_document/beigebook.../chunk_0001/content.txt
/kg/chunks/by_document/beigebook.../chunk_0002/content.txt
...
```

**By Type**:
```bash
$ find /kg/entities/ -type d                   # Directories only (entities)
$ find /kg/facts/ -type f                      # Files only (fact records)
```

**By Metadata**:
```bash
$ find /kg/chunks/ -date "2024-10"            # Chunks from October 2024
$ find /kg/entities/ -entity_type "Organization"
$ find /kg/facts/ -relationship "ACQUIRED"
$ find /kg/topics/ -has_subtopics true
```

**Combined Criteria**:
```bash
$ find /kg/entities/organizations/ -name "*Inc*" -has_facts_count ">10"
/kg/entities/organizations/Alphabet_Inc/       # 47 facts
/kg/entities/organizations/Apple_Inc/          # 38 facts
/kg/entities/organizations/Meta_Platforms_Inc/ # 29 facts
```

**By Content Properties**:
```bash
$ find /kg/facts/ -subject "Federal Reserve" -type "RAISED_POLICY_RATE"
/kg/facts/by_type/RAISED_POLICY_RATE/f_001.json
/kg/facts/by_type/RAISED_POLICY_RATE/f_015.json

$ find /kg/chunks/ -mentions_entity "Boston" -topic "labor_markets"
/kg/chunks/by_document/beigebook.../chunk_0042/
/kg/chunks/by_document/beigebook.../chunk_0089/
```

### head/tail - Pagination

The `head` and `tail` commands enable paginated access to large result sets.

**Basic Usage**:
```bash
$ ls /kg/entities/organizations/ | head -10
Alphabet_Inc/
Amazon/
Apple_Inc/
Bank_of_America/
Berkshire_Hathaway/
Citigroup/
Goldman_Sachs/
JPMorgan_Chase/
Meta_Platforms/
Microsoft/

$ cat /kg/entities/organizations/Alphabet_Inc/facts/all.jsonl | tail -5
{"fact_id": "f043", "type": "PARTNERED_WITH", ...}
{"fact_id": "f044", "type": "REPORTED_FINANCIALS", ...}
{"fact_id": "f045", "type": "LAUNCHED_PRODUCT", ...}
{"fact_id": "f046", "type": "INVESTED_IN", ...}
{"fact_id": "f047", "type": "ACQUIRED", ...}
```

**Pagination Pattern**:
```bash
$ grep "inflation" /kg/chunks/ | head -20              # First page
$ grep "inflation" /kg/chunks/ | head -40 | tail -20   # Second page
```

**With Counts**:
```bash
$ find /kg/entities/ -type d | head -50 -c    # Show 50 and total count
Showing 50 of 1,247 entities
```

### wc - Counting

The `wc` command counts items in the knowledge graph.

**Basic Counts**:
```bash
$ wc -l /kg/entities/organizations/_index.txt
247 organizations

$ wc -l /kg/topics/_ontology.json
89 topics

$ ls /kg/entities/organizations/Alphabet_Inc/facts/ | wc -l
47 facts
```

**Aggregations**:
```bash
$ wc /kg/entities/                             # Summary stats
   1,247 entities
     247 organizations
     412 people
     156 locations
     432 concepts

$ wc /kg/facts/by_type/
     89 ACQUIRED
     56 PARTNERED_WITH
    124 REPORTED_FINANCIALS
    ...
   1,842 total facts
```

### pwd - Current Context

The `pwd` command shows the current navigation context with optional metadata.

**Basic Usage**:
```bash
$ pwd
/kg/entities/organizations/Alphabet_Inc/
```

**With Context Info**:
```bash
$ pwd -v
Current: /kg/entities/organizations/Alphabet_Inc/
Entity:  Alphabet Inc.
Type:    Organization
Facts:   47
Chunks:  23
Topics:  8
```

---

## Command to Query Translation

Under the hood, each command translates to efficient DuckDB and LanceDB queries. This section details the mapping.

### Storage Architecture

The virtual filesystem is backed by two databases:

1. **DuckDB**: Relational storage for structured data
   - `entities` table: Entity metadata, summaries, embeddings
   - `facts` table: Fact records with subject/object references
   - `chunks` table: Chunk content, metadata, document references
   - `documents` table: Document metadata
   - `topics` table: Topic definitions and hierarchy
   - `entity_chunks` junction: Entity-chunk associations
   - `fact_topics` junction: Fact-topic associations

2. **LanceDB**: Vector storage for semantic search
   - `entity_embeddings`: Entity name + summary embeddings
   - `fact_embeddings`: Fact content embeddings
   - `chunk_embeddings`: Chunk content embeddings
   - `topic_embeddings`: Topic definition embeddings

### ls Translation

**List entities in a type directory**:
```bash
$ ls /kg/entities/organizations/
```
Translates to:
```sql
SELECT name,
       (SELECT COUNT(*) FROM facts WHERE subject_id = e.id) as fact_count
FROM entities e
WHERE entity_type = 'Organization'
  AND group_id = $current_group
ORDER BY name
LIMIT 100;
```

**List entity subdirectories**:
```bash
$ ls /kg/entities/organizations/Alphabet_Inc/
```
Translates to:
```sql
-- Static structure, no query needed
-- Returns: ['summary.txt', 'metadata.json', 'facts/', 'relationships/', 'topics/', 'chunks/']
```

**List relationships**:
```bash
$ ls /kg/entities/organizations/Alphabet_Inc/relationships/
```
Translates to:
```sql
SELECT DISTINCT relationship_type
FROM facts
WHERE subject_id = (SELECT id FROM entities WHERE name = 'Alphabet Inc.')
  AND group_id = $current_group;
```

**List relationship targets**:
```bash
$ ls /kg/entities/organizations/Alphabet_Inc/relationships/ACQUIRED/
```
Translates to:
```sql
SELECT e.name
FROM facts f
JOIN entities e ON f.object_id = e.id
WHERE f.subject_id = (SELECT id FROM entities WHERE name = 'Alphabet Inc.')
  AND f.relationship_type = 'ACQUIRED'
  AND f.group_id = $current_group;
```

### cd Translation

`cd` commands primarily update the agent's current context state. Path resolution involves:

```python
def resolve_path(path: str, current_context: Context) -> Context:
    """Resolve a path and return new context."""

    if path.startswith('/'):
        # Absolute path
        segments = path.split('/')[1:]  # Remove empty first element
    else:
        # Relative path
        segments = current_context.path.split('/') + path.split('/')

    # Handle .. and .
    resolved = []
    for seg in segments:
        if seg == '..':
            resolved.pop() if resolved else None
        elif seg == '.':
            continue
        else:
            resolved.append(seg)

    # Validate path exists
    new_path = '/' + '/'.join(resolved)
    if not path_exists(new_path):
        raise FileNotFoundError(f"No such directory: {new_path}")

    return Context(path=new_path, entity=extract_entity(new_path), ...)
```

Path existence validation queries:
```sql
-- For entity paths
SELECT 1 FROM entities WHERE name = $entity_name AND group_id = $group_id;

-- For topic paths
SELECT 1 FROM topics WHERE name = $topic_name AND group_id = $group_id;

-- For chunk paths
SELECT 1 FROM chunks WHERE chunk_id = $chunk_id AND group_id = $group_id;
```

### cat Translation

**Read entity summary**:
```bash
$ cat /kg/entities/organizations/Alphabet_Inc/summary.txt
```
Translates to:
```sql
SELECT summary FROM entities
WHERE name = 'Alphabet Inc.' AND group_id = $current_group;
```

**Read entity metadata**:
```bash
$ cat /kg/entities/organizations/Alphabet_Inc/metadata.json
```
Translates to:
```sql
SELECT entity_type, aliases, metadata
FROM entities
WHERE name = 'Alphabet Inc.' AND group_id = $current_group;
```

**Read chunk content**:
```bash
$ cat /kg/chunks/by_document/beigebook_2024_10/chunk_0042/content.txt
```
Translates to:
```sql
SELECT content FROM chunks
WHERE chunk_id = 'beigebook_2024_10_chunk_0042' AND group_id = $current_group;
```

**Read all facts**:
```bash
$ cat /kg/entities/organizations/Alphabet_Inc/facts/all.jsonl
```
Translates to:
```sql
SELECT f.id, f.relationship_type, e2.name as object_name, f.description, f.date_context
FROM facts f
LEFT JOIN entities e2 ON f.object_id = e2.id
WHERE f.subject_id = (SELECT id FROM entities WHERE name = 'Alphabet Inc.')
  AND f.group_id = $current_group
ORDER BY f.date_context DESC;
```

### grep Translation

**Semantic search**:
```bash
$ grep "economic impact of interest rate changes" /kg/chunks/
```
Translates to:
1. Generate embedding for query text
2. Vector search in LanceDB:
```python
results = chunk_embeddings.search(query_embedding).limit(50).to_list()
```
3. Format results with similarity scores

**Keyword search**:
```bash
$ grep -F "Federal Reserve" /kg/chunks/
```
Translates to:
```sql
SELECT chunk_id, content,
       POSITION('Federal Reserve' IN content) as match_pos
FROM chunks
WHERE content LIKE '%Federal Reserve%'
  AND group_id = $current_group
ORDER BY match_pos
LIMIT 50;
```

**Scoped semantic search** (within entity context):
```bash
$ cd /kg/entities/organizations/Alphabet_Inc/
$ grep "artificial intelligence investments"
```
Translates to:
```python
# Get chunks associated with entity
entity_chunks = db.execute("""
    SELECT c.chunk_id, c.content
    FROM chunks c
    JOIN entity_chunks ec ON c.id = ec.chunk_id
    WHERE ec.entity_id = (SELECT id FROM entities WHERE name = 'Alphabet Inc.')
""").fetchall()

# Generate embeddings for entity chunks
chunk_embeddings = embed([c.content for c in entity_chunks])

# Vector search within entity scope
results = cosine_similarity(query_embedding, chunk_embeddings)
```

### find Translation

**By name pattern**:
```bash
$ find /kg/entities/ -name "*Google*"
```
Translates to:
```sql
SELECT 'entities/' || entity_type || '/' || name as path
FROM entities
WHERE name LIKE '%Google%'
  AND group_id = $current_group;
```

**By metadata**:
```bash
$ find /kg/facts/ -subject "Federal Reserve" -type "RAISED_POLICY_RATE"
```
Translates to:
```sql
SELECT 'facts/by_type/' || relationship_type || '/' || id as path
FROM facts f
JOIN entities e ON f.subject_id = e.id
WHERE e.name = 'Federal Reserve'
  AND f.relationship_type = 'RAISED_POLICY_RATE'
  AND f.group_id = $current_group;
```

**By date**:
```bash
$ find /kg/chunks/ -date "2024-10"
```
Translates to:
```sql
SELECT 'chunks/by_document/' || document_id || '/' || chunk_id as path
FROM chunks
WHERE valid_at >= '2024-10-01' AND valid_at < '2024-11-01'
  AND group_id = $current_group;
```

### Semantic Search Integration

The `grep` command's semantic search leverages hybrid retrieval:

```python
def semantic_grep(query: str, scope: str, options: GrepOptions) -> list[GrepResult]:
    """Execute semantic grep with hybrid retrieval."""

    # 1. Generate query embedding
    query_embedding = embed(query)

    # 2. Determine search scope
    if scope == '/kg/chunks/':
        table = 'chunk_embeddings'
    elif scope == '/kg/entities/':
        table = 'entity_embeddings'
    elif scope == '/kg/facts/':
        table = 'fact_embeddings'
    else:
        # Scoped search within entity/topic context
        table = get_scoped_embeddings(scope)

    # 3. Vector search
    vector_results = lancedb.search(
        table=table,
        query=query_embedding,
        limit=options.limit or 50,
        filter=options.filter
    )

    # 4. Optional: Combine with keyword search
    if options.hybrid:
        keyword_results = duckdb.execute(f"""
            SELECT id, ts_rank(content_tsv, query) as rank
            FROM {table}
            WHERE content_tsv @@ to_tsquery($query)
        """, [query]).fetchall()

        results = reciprocal_rank_fusion(vector_results, keyword_results)
    else:
        results = vector_results

    # 5. Format output
    return [
        GrepResult(
            path=result.path,
            score=result.score,
            snippet=extract_snippet(result.content, query)
        )
        for result in results
    ]
```

### Pattern Matching with find

The `find` command supports complex pattern matching:

```python
def find_entities(
    path: str,
    name: str = None,
    entity_type: str = None,
    has_relationship: str = None,
    min_facts: int = None,
    date_range: tuple = None
) -> list[str]:
    """Find entities matching criteria."""

    conditions = ["group_id = $group_id"]
    params = {"group_id": current_group}

    if name:
        conditions.append("name LIKE $name_pattern")
        params["name_pattern"] = name.replace("*", "%")

    if entity_type:
        conditions.append("entity_type = $entity_type")
        params["entity_type"] = entity_type

    if has_relationship:
        conditions.append("""
            EXISTS (
                SELECT 1 FROM facts
                WHERE subject_id = entities.id
                  AND relationship_type = $rel_type
            )
        """)
        params["rel_type"] = has_relationship

    if min_facts:
        conditions.append("""
            (SELECT COUNT(*) FROM facts WHERE subject_id = entities.id) >= $min_facts
        """)
        params["min_facts"] = min_facts

    query = f"""
        SELECT entity_type || '/' || name as path
        FROM entities
        WHERE {' AND '.join(conditions)}
        ORDER BY name
    """

    return duckdb.execute(query, params).fetchall()
```

---

## Agent Interaction Flow

This section illustrates how an AI agent would use the filesystem interface to answer complex questions.

### Example Exploration Session

**User Question**: "What acquisitions has Alphabet made in the AI space and how have they impacted their AI capabilities?"

**Agent Reasoning and Navigation**:

```
THOUGHT: I need to find information about Alphabet's acquisitions, specifically
those related to AI. Let me start by exploring Alphabet's entity.

$ cd /kg/entities/organizations/Alphabet_Inc/
$ pwd
/kg/entities/organizations/Alphabet_Inc/

THOUGHT: Good, I found the entity. Let me see what information is available.

$ ls
summary.txt    metadata.json    facts/    relationships/    topics/    chunks/

THOUGHT: I should look at the acquisitions. Let me check the relationships.

$ ls relationships/
ACQUIRED/    PARTNERED_WITH/    INVESTED_IN/    SUBSIDIARY_OF/    ...

THOUGHT: ACQUIRED is exactly what I need. Let me see what companies they acquired.

$ ls relationships/ACQUIRED/
DeepMind/    Fitbit/    Mandiant/    Looker/    Nest_Labs/    ...

THOUGHT: DeepMind is clearly AI-related. Let me investigate it first.

$ cd DeepMind/
$ pwd
/kg/entities/organizations/DeepMind/

$ cat summary.txt
DeepMind is a British artificial intelligence company founded in 2010.
It was acquired by Google (now Alphabet) in 2014 for approximately $500 million.
DeepMind is known for developing AlphaGo, AlphaFold, and other breakthrough
AI systems...

THOUGHT: This confirms DeepMind is a major AI acquisition. Let me find specific
facts about the acquisition and its impact.

$ ls facts/
by_type/    all.jsonl

$ ls facts/by_type/
GOT_ACQUIRED/    DEVELOPED_PRODUCT/    PARTNERED_WITH/    ...

$ cat facts/by_type/GOT_ACQUIRED/
{"fact_id": "f_dm_001", "subject": "Google", "description": "Google acquired
DeepMind in January 2014 for approximately $500 million, marking its largest
European acquisition at the time", "date": "2014-01"}

THOUGHT: Now let me search for impacts on AI capabilities. I'll use semantic
search within Alphabet's context.

$ cd /kg/entities/organizations/Alphabet_Inc/
$ grep "AI capabilities impact DeepMind"
chunks/chunk_0089/content.txt: [0.91] Following the DeepMind acquisition,
  Google significantly enhanced its AI research capabilities...
chunks/chunk_0124/content.txt: [0.88] DeepMind's technology has been
  integrated into Google's data centers, reducing cooling costs by 40%...
facts/f_042.json: [0.85] Alphabet reported that DeepMind's contributions
  led to major improvements in Google Search and Assistant...

THOUGHT: Let me check the source chunks for more context.

$ cat /kg/chunks/by_document/alphabet_10k_2023/chunk_0089/content.txt
Following the DeepMind acquisition, Google significantly enhanced its AI
research capabilities. The acquisition brought world-class AI researchers
including Demis Hassabis to the company. DeepMind's innovations have been
applied across Google products, from improving search results to developing
new healthcare applications...

THOUGHT: I should check if there are other AI-related acquisitions. Let me
search more broadly.

$ find /kg/entities/ -name "*AI*" -related_to "Alphabet_Inc"
/kg/entities/organizations/Google_AI/
/kg/entities/concepts/Artificial_Intelligence/

$ grep "Alphabet acquired.*artificial intelligence" /kg/facts/
/kg/facts/f_alph_042.json: [0.92] Alphabet acquired UK AI lab DeepMind...
/kg/facts/f_alph_089.json: [0.87] Google acquired AI startup DNNresearch...
/kg/facts/f_alph_102.json: [0.84] Alphabet acquired Kaggle, an AI competition platform...

THOUGHT: I found more AI acquisitions. Let me explore DNNresearch and Kaggle.

$ cat /kg/entities/organizations/DNNresearch/summary.txt
DNNresearch was a startup founded by Geoffrey Hinton, a pioneer in deep
learning. Google acquired the company in 2013, bringing Hinton and his
research on neural networks to Google...

$ cat /kg/entities/organizations/Kaggle/summary.txt
Kaggle is a platform for data science competitions and machine learning.
Google acquired Kaggle in 2017, gaining access to its large community of
data scientists and extensive collection of datasets...

CONCLUSION: I now have comprehensive information about Alphabet's AI
acquisitions (DeepMind, DNNresearch, Kaggle) and their impacts on AI
capabilities. I can synthesize this into an answer.
```

### How Agent Decides Next Navigation Step

The agent's navigation decisions follow natural reasoning patterns:

1. **Goal-oriented exploration**: "I need to find X, so I should look in Y"
   ```
   THOUGHT: I need acquisition information
   ACTION: ls relationships/ACQUIRED/
   ```

2. **Discovery-driven branching**: "I found something interesting, let me explore it"
   ```
   THOUGHT: DeepMind appears AI-related
   ACTION: cd DeepMind/
   ```

3. **Search when unsure**: "I'm not sure where to find this, let me search"
   ```
   THOUGHT: There might be other AI acquisitions
   ACTION: grep "artificial intelligence" /kg/facts/
   ```

4. **Backtracking on dead ends**: "This isn't what I need, let me try another path"
   ```
   THOUGHT: This topic isn't relevant
   ACTION: cd ..
   ```

5. **Aggregation**: "I need to combine information from multiple sources"
   ```
   THOUGHT: Let me check the chunks for more context
   ACTION: cat /kg/chunks/.../content.txt
   ```

### Building Context Iteratively

Each navigation step adds to the agent's understanding:

| Step | Action | Context Gained |
|------|--------|----------------|
| 1 | `cd /kg/entities/organizations/Alphabet_Inc/` | Entity exists, has standard structure |
| 2 | `ls relationships/` | Available relationship types |
| 3 | `ls relationships/ACQUIRED/` | List of acquired companies |
| 4 | `cat DeepMind/summary.txt` | DeepMind details, AI focus confirmed |
| 5 | `grep "AI capabilities"` | Impact information with relevance scores |
| 6 | `cat chunk_0089/content.txt` | Primary source text with full context |
| 7 | `find ... -related_to "Alphabet_Inc"` | Additional AI-related entities |
| 8 | `cat DNNresearch/summary.txt` | Another AI acquisition |

This iterative building creates:
- **Verified information**: Each step confirms or refutes hypotheses
- **Source provenance**: Path back to original documents
- **Comprehensive coverage**: Multiple exploration paths ensure completeness
- **Natural reasoning trace**: The exploration path documents the agent's logic

---

## Benefits

### Leverages LLM CLI Training

LLMs have extensive exposure to terminal sessions in their training data:

1. **GitHub repositories**: Millions of repos with terminal examples in READMEs
2. **Stack Overflow**: Countless Q&A involving CLI commands
3. **Documentation**: Man pages, tutorials, guides
4. **Chat logs**: Developer conversations with command examples

This training means LLMs can:
- Predict appropriate command syntax
- Understand option flags (-l, -a, -r)
- Compose commands naturally (pipes, redirects)
- Handle errors and adjust approach

### Intuitive Exploration

The filesystem metaphor enables exploration without query language knowledge:

| Traditional Query Interface | Filesystem Interface |
|---------------------------|---------------------|
| "What is the MATCH syntax?" | "I'll just ls and see what's here" |
| "Which properties does EntityNode have?" | "Let me cat the summary" |
| "How do I join facts to entities?" | "I'll cd into relationships/" |
| "What's the regex for relationship types?" | "ls shows me the options" |

### Fast Local Queries

All operations translate to local database queries:

1. **DuckDB for structure**: Microsecond response times for metadata
2. **LanceDB for vectors**: Sub-100ms for semantic search
3. **No network roundtrips**: Everything runs locally
4. **Efficient caching**: Repeated queries hit cache

Typical response times:
- `ls`: <1ms
- `cat`: <5ms
- `grep` (semantic): 50-100ms
- `find` (complex): 10-50ms

### No Complex Query Construction

The agent never needs to:
- Learn Cypher or SQL syntax
- Understand graph schema upfront
- Construct multi-join queries
- Handle query optimization

Instead, the agent:
- Explores naturally using familiar commands
- Discovers schema through `ls` and `cat`
- Searches semantically with `grep`
- Filters with intuitive `find` options

---

## Implementation Considerations

### Path Resolution

Path resolution must handle several complexities:

**Absolute vs Relative Paths**:
```python
def resolve_path(path: str, cwd: str) -> str:
    if path.startswith('/'):
        return normalize(path)
    return normalize(join(cwd, path))
```

**Symlink Resolution**:
```python
def resolve_symlinks(path: str) -> str:
    """Resolve symlinks to canonical paths."""
    segments = path.split('/')
    resolved = []

    for seg in segments:
        current = '/'.join(resolved + [seg])
        if is_symlink(current):
            target = read_symlink(current)
            if target.startswith('/'):
                resolved = target.split('/')
            else:
                resolved.append(target)
        else:
            resolved.append(seg)

    return '/'.join(resolved)
```

**Virtual Directory Generation**:
Some directories are generated on-the-fly:
```python
def list_directory(path: str) -> list[str]:
    if path == '/kg/entities/':
        return ['organizations/', 'people/', 'locations/', 'concepts/']
    elif path.matches('/kg/entities/{type}/'):
        return query_entities_by_type(path.type)
    elif path.matches('/kg/entities/{type}/{entity}/'):
        return ['summary.txt', 'metadata.json', 'facts/', 'relationships/', ...]
```

### Caching and Context

**Session Context**:
```python
@dataclass
class NavigationContext:
    cwd: str                          # Current working directory
    history: list[str]                # Navigation history for cd -
    entity_cache: dict[str, Entity]   # Recently accessed entities
    search_cache: dict[str, list]     # Recent search results

    def push_directory(self, path: str):
        self.history.append(self.cwd)
        self.cwd = path

    def pop_directory(self) -> str:
        if self.history:
            self.cwd = self.history.pop()
        return self.cwd
```

**Result Caching**:
```python
class ResultCache:
    def __init__(self, ttl_seconds: int = 300):
        self.cache = {}
        self.ttl = ttl_seconds

    def get(self, command: str) -> Optional[Any]:
        if command in self.cache:
            result, timestamp = self.cache[command]
            if time.time() - timestamp < self.ttl:
                return result
            del self.cache[command]
        return None

    def set(self, command: str, result: Any):
        self.cache[command] = (result, time.time())
```

### Error Handling

**Path Errors**:
```python
class PathError(Exception):
    pass

class FileNotFoundError(PathError):
    def __init__(self, path: str):
        super().__init__(f"No such file or directory: {path}")
        self.path = path

class NotADirectoryError(PathError):
    def __init__(self, path: str):
        super().__init__(f"Not a directory: {path}")

class PermissionDeniedError(PathError):
    def __init__(self, path: str, operation: str):
        super().__init__(f"Permission denied: {operation} on {path}")
```

**Graceful Degradation**:
```python
def handle_command_error(error: Exception, command: str) -> str:
    """Return helpful error messages in CLI style."""

    if isinstance(error, FileNotFoundError):
        suggestions = find_similar_paths(error.path)
        return f"""
{command}: {error}
Did you mean one of these?
  {chr(10).join(suggestions[:5])}
"""

    if isinstance(error, QueryTimeoutError):
        return f"""
{command}: query timed out
Try a more specific path or use 'head' to limit results:
  {command} | head -20
"""

    return f"{command}: {error}"
```

### Pagination Strategies

**Cursor-based Pagination**:
```python
class PaginatedResult:
    def __init__(self, items: list, page_size: int = 50):
        self.items = items
        self.page_size = page_size
        self.cursor = 0

    def next_page(self) -> list:
        start = self.cursor
        end = min(self.cursor + self.page_size, len(self.items))
        self.cursor = end
        return self.items[start:end]

    def has_more(self) -> bool:
        return self.cursor < len(self.items)
```

**Streaming Large Results**:
```python
def stream_cat(path: str) -> Iterator[str]:
    """Stream large file contents."""
    content = read_file(path)

    # Stream in chunks for large content
    if len(content) > 10000:
        chunk_size = 2000
        for i in range(0, len(content), chunk_size):
            yield content[i:i + chunk_size]
    else:
        yield content
```

**Head/Tail Integration**:
```python
def apply_head_tail(results: list, head: int = None, tail: int = None) -> list:
    """Apply head/tail filtering to results."""
    if head and tail:
        # head -N | tail -M pattern for offset
        return results[:head][-tail:]
    elif head:
        return results[:head]
    elif tail:
        return results[-tail:]
    return results
```

### Performance Optimizations

**Lazy Loading**:
```python
class LazyDirectory:
    """Directory that only loads contents when accessed."""

    def __init__(self, path: str, loader: Callable):
        self.path = path
        self.loader = loader
        self._contents = None

    @property
    def contents(self) -> list:
        if self._contents is None:
            self._contents = self.loader(self.path)
        return self._contents
```

**Parallel Queries**:
```python
async def ls_with_stats(path: str) -> list[DirEntry]:
    """List directory with parallel stat calls."""
    entries = await list_directory(path)

    # Parallel stat for all entries
    stats = await asyncio.gather(*[
        stat_entry(join(path, entry))
        for entry in entries
    ])

    return [
        DirEntry(name=entry, stats=stat)
        for entry, stat in zip(entries, stats)
    ]
```

**Index Pre-warming**:
```python
async def warm_caches(common_paths: list[str]):
    """Pre-warm caches for commonly accessed paths."""
    await asyncio.gather(*[
        list_directory(path)
        for path in common_paths
    ])
```

---

## Conclusion

The filesystem navigation architecture provides a powerful, intuitive interface for AI agents to explore knowledge graphs. By mapping graph operations to familiar CLI commands, it leverages extensive LLM training on terminal interactions to enable zero-shot exploration without complex query languages.

Key architectural principles:

1. **Metaphor alignment**: Graph concepts map naturally to filesystem concepts
2. **Iterative discovery**: Agents explore progressively, building context
3. **Hybrid search**: Semantic and keyword search through unified `grep` interface
4. **Local execution**: Fast DuckDB/LanceDB queries with aggressive caching
5. **Graceful errors**: CLI-style error messages guide recovery

This architecture transforms knowledge graph querying from a complex query construction task into a natural exploration experience that any LLM can perform effectively.
