# Examples

Use these as short, realistic query patterns.

## 1) Temporal Comparison

Question: How did the Fed's inflation language shift from October 2024 to January 2025?

```bash
find --entity {"name":"Federal Reserve","definition":"US central banking system"} --topic {"name":"Inflation","definition":"General price increases"}
search -entity "Federal Reserve" -topic Inflation --from 2024-10 --to-date 2024-10
search -entity "Federal Reserve" -topic Inflation --from 2025-01 --to-date 2025-01
cat 1
```

Notes:
- Date filters scope by publication date.
- Result line dates display fact `date_context`.

## 2) Acquisition-Focused Entity Query

Question: What acquisitions are associated with Apple?

```bash
find --entity {"name":"Apple","definition":"Consumer electronics and services company"}
info "Apple Inc"
search -entity "Apple Inc" --query "acquired"
cat 1 2
```

Notes:
- `--query` reranks Apple-related facts by acquisition-like relationships.
- Validate top hits with `cat`.

## 3) Multi-Entity N-to-N Search

Question: How are large tech companies linked to interest rates?

```bash
find --entity {"name":"Apple","definition":"Consumer electronics company"} --entity {"name":"Microsoft","definition":"Enterprise software company"} --entity {"name":"Google","definition":"Internet services company"} --topic {"name":"Interest Rates","definition":"Costs of borrowing money set by lenders"}
search -entity "Apple Inc, Microsoft Corp, Alphabet Inc" -topic "Interest Rates" --mode around --from 2024-01
cat 1 2 3
```

Notes:
- Include all relevant canonical names from `find`.
- `--mode around` returns edges that touch any selected node.

## 4) Exploration-First Pattern

Question: Start broad, then narrow.

```bash
find --entity {"name":"Federal Reserve","definition":"US central banking system"}
search -entity "Federal Reserve"
search -entity "Federal Reserve" -topic "Employment"
search -entity "Federal Reserve" -topic "Employment" --query "tightening" --from 2024-01
cat 1
```
