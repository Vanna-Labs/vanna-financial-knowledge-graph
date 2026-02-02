# ZommaKG Query Strategies

You have access to a knowledge graph with two query modes. This skill helps you choose the right approach for each question.

---

## Two Query Modes

### Mode 1: Direct Query (`kg.query()`)

Structured pipeline that automatically:
1. Decomposes question into sub-queries
2. Resolves entity/topic hints via vector search
3. Retrieves relevant chunks and facts
4. Synthesizes answer

**Characteristics:**
- Fast (single pipeline execution)
- Deterministic (same question → same retrieval)
- Best for well-defined questions with clear entities
- Limited exploration (follows pre-set retrieval patterns)

### Mode 2: Shell Exploration (`kg.query_with_shell()`)

Agent-driven navigation that:
1. Uses shell commands to explore the graph
2. Builds context iteratively through navigation
3. Follows connections discovered during exploration
4. Synthesizes answer from gathered evidence

**Characteristics:**
- Slower (multiple shell commands)
- Adaptive (can change direction based on findings)
- Best for exploratory questions or complex connections
- Can discover unexpected relationships

---

## Decision Matrix

| Question Type | Best Mode | Why |
|---------------|-----------|-----|
| Simple factual | Direct | "What was Apple's Q3 revenue?" - clear entity, clear ask |
| Entity lookup | Direct | "Tell me about Tesla" - straightforward retrieval |
| Comparison | Direct | "Compare Apple and Microsoft's cloud strategies" - known entities |
| Exploratory | Shell | "What's driving inflation concerns?" - need to discover entities |
| Connection finding | Shell | "How are Fed policy and tech layoffs related?" - need to trace paths |
| Verification | Shell | "Is it true that Google acquired DeepMind?" - need to verify with sources |
| Multi-hop | Shell | "Who acquired companies that compete with OpenAI?" - requires traversal |
| Temporal analysis | Either | Depends on specificity |
| Enumeration | Direct | "List all acquisitions in Q3" - structured retrieval |

---

## Decision Tree

```
┌─────────────────────────────────────────────────────────────┐
│                    Incoming Question                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│     Are the key entities clearly specified?                  │
└─────────────────────────────────────────────────────────────┘
                    │                    │
                   YES                   NO
                    │                    │
                    ▼                    ▼
        ┌───────────────────┐    ┌───────────────────┐
        │ Is it a simple    │    │ SHELL             │
        │ factual question? │    │ Need to discover  │
        └───────────────────┘    │ relevant entities │
            │           │        └───────────────────┘
           YES          NO
            │           │
            ▼           ▼
    ┌───────────┐  ┌────────────────────────────────┐
    │ DIRECT    │  │ Does it require tracing        │
    │ Fast path │  │ connections or verifying?      │
    └───────────┘  └────────────────────────────────┘
                        │                    │
                       YES                   NO
                        │                    │
                        ▼                    ▼
                ┌───────────────┐    ┌───────────────┐
                │ SHELL         │    │ DIRECT        │
                │ Need to       │    │ Structured    │
                │ explore paths │    │ retrieval ok  │
                └───────────────┘    └───────────────┘
```

---

## Question Patterns → Strategy

### Use DIRECT Query For:

**1. Factual Questions with Named Entities**
```
"What was Apple's revenue in Q3 2024?"
"Who is the CEO of Microsoft?"
"When did Google acquire YouTube?"
```
- Entities are explicitly named
- Question has clear, bounded answer
- No exploration needed

**2. Comparisons Between Known Entities**
```
"Compare Tesla and Ford's EV strategies"
"How do JPMorgan and Goldman Sachs differ on interest rate outlook?"
```
- Entities are known upfront
- Parallel retrieval works well
- Structured comparison synthesis

**3. List/Enumeration Questions**
```
"What companies did Alphabet acquire in 2023?"
"List all topics related to inflation"
```
- Bounded scope
- Structured retrieval sufficient
- Can filter by date/type

**4. Definition/Summary Questions**
```
"What is quantitative easing?"
"Summarize the Fed's recent policy stance"
```
- Topic-based retrieval
- Existing summaries useful
- No traversal needed

---

### Use SHELL Exploration For:

**1. Discovery Questions (Entities Unknown)**
```
"What's driving concerns about commercial real estate?"
"Which sectors are most affected by supply chain issues?"
```
- Don't know which entities to retrieve
- Need to discover relevant players
- Shell: `grep "commercial real estate concerns" /kg/` → follow results

**2. Connection/Relationship Questions**
```
"How are rising interest rates affecting tech companies?"
"What's the relationship between Fed policy and unemployment?"
```
- Need to trace paths between concepts
- Multiple hops required
- Shell: Navigate from topic → entities → their facts

**3. Verification Questions**
```
"Is it true that Amazon is entering healthcare?"
"Did Apple really acquire a car company?"
```
- Need to find evidence (or lack thereof)
- Must trace to source chunks
- Shell: Find fact → verify with source chunk

**4. Multi-Hop Traversal**
```
"Which companies acquired AI startups that compete with OpenAI?"
"What do former Google executives say about AI regulation?"
```
- Requires: find X → find Y related to X → filter by criteria
- Can't do in single retrieval
- Shell: Navigate step by step

**5. Exploratory/Open-Ended**
```
"What interesting patterns exist in Q3 earnings reports?"
"What are the emerging themes in Fed communications?"
```
- No specific target
- Need to browse and discover
- Shell: ls, grep, follow interesting paths

**6. Debugging/Verification**
```
"What does the KB actually know about Tesla?"
"Show me the source for that claim about inflation"
```
- Need to inspect raw data
- Provenance verification
- Shell: Direct navigation and cat

---

## Hybrid Approach

Sometimes start with one mode, then switch:

### Direct → Shell

```
1. kg.query("What acquisitions did Meta make?")
   → Returns list but missing context

2. Switch to shell to explore:
   cd /kg/entities/companies/Meta/relationships/ACQUIRED/
   ls
   cat WhatsApp/summary.txt
   → Get deeper context
```

**When to switch:** Direct answer feels incomplete or needs verification

### Shell → Direct

```
1. Shell exploration discovers key entities:
   grep "AI chip shortage" /kg/
   → Discovers: NVIDIA, TSMC, Intel

2. Switch to direct for structured comparison:
   kg.query("Compare NVIDIA and Intel's response to chip demand")
   → Structured parallel retrieval
```

**When to switch:** Discovered the entities, now need structured analysis

---

## Speed vs Thoroughness

| Priority | Strategy | Trade-off |
|----------|----------|-----------|
| **Speed** | Direct query | May miss nuanced connections |
| **Thoroughness** | Shell exploration | Takes longer, uses more tokens |
| **Balanced** | Direct first, shell if insufficient | Best of both, but two attempts |

### Time Budget Heuristics

| Available Time | Recommended Approach |
|----------------|---------------------|
| Quick answer needed | Direct query only |
| Normal query | Direct, verify key claims with shell if needed |
| Deep research | Shell exploration with systematic coverage |
| Audit/verification | Shell mandatory (need provenance) |

---

## Question Classification

Before choosing a strategy, classify the question:

### Type 1: FACTUAL
```
"What was X's revenue?"
"When did X happen?"
"Who is the CEO of X?"
```
→ **DIRECT** (unless entity unclear)

### Type 2: COMPARISON
```
"Compare X and Y"
"How does X differ from Y?"
"Which is larger, X or Y?"
```
→ **DIRECT** (parallel retrieval)

### Type 3: CAUSAL
```
"Why did X happen?"
"What caused X?"
"How does X affect Y?"
```
→ **SHELL** (need to trace connections)

### Type 4: TEMPORAL
```
"What changed from Q1 to Q2?"
"How has X evolved over time?"
```
→ **DIRECT** (if entities known) or **SHELL** (if discovering trends)

### Type 5: ENUMERATION
```
"List all X"
"What are the Y that Z?"
"Which companies..."
```
→ **DIRECT** (structured retrieval with filters)

### Type 6: EXPLORATORY
```
"What does the KB know about X?"
"What's interesting about Y?"
"Tell me about Z's relationships"
```
→ **SHELL** (need to browse)

### Type 7: VERIFICATION
```
"Is it true that X?"
"Can you confirm Y?"
"Show evidence for Z"
```
→ **SHELL** (need provenance)

---

## Examples

### Example 1: Factual → Direct

**Question:** "What was Tesla's revenue in Q3 2024?"

**Analysis:**
- Entity: Tesla (clearly specified)
- Ask: Revenue, Q3 2024 (bounded, factual)
- Strategy: **DIRECT**

```python
result = await kg.query("What was Tesla's revenue in Q3 2024?")
```

---

### Example 2: Causal → Shell

**Question:** "How are rising interest rates affecting the housing market?"

**Analysis:**
- Entities: Not specific (which aspects of housing? which players?)
- Ask: Causal chain (rates → effects)
- Strategy: **SHELL**

```bash
# Find the connection
grep "interest rates housing" /kg/
cd /kg/topics/interest_rates/
ls entities/
grep "mortgage" /kg/entities/
cd /kg/topics/housing_market/
ls chunks/ | head -20
cat chunks/fed_report_chunk_042.txt
```

---

### Example 3: Verification → Shell

**Question:** "Is it true that Apple is developing an AR headset?"

**Analysis:**
- Entity: Apple (known)
- Ask: Verify a claim (need evidence)
- Strategy: **SHELL**

```bash
cd /kg/entities/companies/Apple_Inc/
grep "AR headset" facts/
grep "augmented reality" facts/
# If found:
cat facts/DEVELOPING/f_023.txt  # Get chunk_id
cd /kg/chunks/by_document/apple_10k/chunk_089/
cat content.txt  # Verify source says this
```

---

### Example 4: Multi-hop → Shell

**Question:** "Which companies that Alphabet acquired are now competing with OpenAI?"

**Analysis:**
- Requires: Alphabet → acquisitions → filter by AI → check competition
- Multi-hop traversal
- Strategy: **SHELL**

```bash
cd /kg/entities/companies/Alphabet_Inc/
ls relationships/ACQUIRED/
# For each acquisition:
cat relationships/ACQUIRED/DeepMind/summary.txt
grep "OpenAI" relationships/ACQUIRED/DeepMind/
grep "compete" relationships/ACQUIRED/DeepMind/facts/
# Repeat for other AI acquisitions
```

---

### Example 5: Hybrid Approach

**Question:** "What's the competitive landscape in cloud computing?"

**Step 1: Shell exploration to find key players**
```bash
grep "cloud computing market" /kg/
cd /kg/topics/cloud_computing/
ls entities/
# Discovers: AWS, Azure, Google Cloud, etc.
```

**Step 2: Direct query for structured comparison**
```python
result = await kg.query(
    "Compare AWS, Azure, and Google Cloud's market position and strategies"
)
```

---

## Red Flags: Wrong Strategy Choice

### Signs you should have used SHELL:
- Direct query returns "I don't have information about X" but you suspect it exists
- Answer lacks nuance or context
- You can't verify where the information came from
- Question involves "how" or "why" connections

### Signs you should have used DIRECT:
- Shell exploration is going in circles
- You already know the key entities
- Question is straightforward factual
- Time is limited

---

## Implementation Notes

### Choosing at Runtime

```python
async def smart_query(kg: KnowledgeGraph, question: str) -> QueryResult:
    # Classify question
    q_type = classify_question(question)  # FACTUAL, CAUSAL, EXPLORATORY, etc.

    # Check if entities are specified
    entities_clear = has_clear_entities(question)

    # Decide strategy
    if q_type in ["FACTUAL", "COMPARISON", "ENUMERATION"] and entities_clear:
        return await kg.query(question)
    elif q_type in ["CAUSAL", "EXPLORATORY", "VERIFICATION"]:
        return await kg.query_with_shell(question)
    else:
        # Default: try direct, fall back to shell if insufficient
        result = await kg.query(question)
        if result.confidence < 0.5:
            return await kg.query_with_shell(question)
        return result
```

### Confidence Threshold

- **Direct query confidence > 0.7:** Trust the answer
- **Direct query confidence 0.5-0.7:** Consider shell verification
- **Direct query confidence < 0.5:** Definitely use shell

---

## Summary

| Question Characteristic | → Strategy |
|------------------------|------------|
| Named entities + factual | DIRECT |
| Unknown entities | SHELL |
| Need to trace connections | SHELL |
| Need source verification | SHELL |
| Comparison of known things | DIRECT |
| List with clear criteria | DIRECT |
| Open-ended exploration | SHELL |
| "Why" or "how" questions | SHELL |
| Time-sensitive | DIRECT |
| Audit/compliance | SHELL |
