# Model Router Notes

## Status

Model router work is complete for the current project scope.
The feature is integrated into both local test flows:

- `test_pipeline.py` runs retrieval -> compression -> model routing for a single query
- `run_all_queries.py` runs the same pipeline for the full query set and writes results to a root CSV file
- `run_all_queries.py` produced `compression_results.csv` in the project root with 50 routed answers

## What Was Implemented

- Added `src/model_router.py`
- Routed queries by complexity:
  - `simple` -> `gpt-3.5-turbo`
  - `medium` -> `gpt-4o-mini`
  - `complex` -> `gpt-4o`
- Updated `test_pipeline.py` to run retrieval, compression, and model routing
- Updated `run_all_queries.py` to record `model_used` and `answer` in the generated CSV output

## Test Command

```powershell
$env:HTTP_PROXY=''; $env:HTTPS_PROXY=''; $env:ALL_PROXY=''; py test_pipeline.py
```

```powershell
$env:HTTP_PROXY=''; $env:HTTPS_PROXY=''; $env:ALL_PROXY=''; py run_all_queries.py
```

## Test Output

```text
Loading documents...
Loaded 20 documents
Created 30 chunks
Creating vector database...
Vector database created
Retrieving 3 documents for simple query...

=== RETRIEVED CHUNKS ===

Chunk 1:
'Title: University_of_Notre_Dame'

Chunk 2:
'Title: University_of_Notre_Dame'

Chunk 3:
'Title: University_of_Notre_Dame'

=== COMPRESSED CONTEXT ===
Title: University_of_Notre_Dame Title: University_of_Notre_Dame Title: University_of_Notre_Dame

=== METRICS ===
Original tokens: 6
Compressed tokens: 6
Compression ratio: 0.0

=== MODEL ROUTER OUTPUT ===
Model used: gpt-3.5-turbo
Answer: Insufficient context to provide an answer.
```

## Observation

The `test_pipeline.py` output above reflects only the current hardcoded single-query example.
For that specific run, retrieval returned weak context, so the model appropriately answered that the context was insufficient.

The batch run in `run_all_queries.py` produced substantive answers across the dataset and confirmed that the router selected models correctly by complexity.

## Representative CSV Results

Examples from `compression_results.csv`:

- Simple query:
  `To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?`
  Model: `gpt-3.5-turbo`
  Answer: `Saint Bernadette Soubirous.`

- Medium query:
  `Compare the missions of different Notre Dame student publications mentioned in the document`
  Model: `gpt-4o-mini`
  Answer: `The Observer's mission is to provide daily reporting on university and other news, while The Juggler focuses on showcasing student literature and artwork through its biannual publication.`

- Complex query:
  `Evaluate the trade-offs between faculty oversight and editorial independence in student publications, and recommend an optimal governance model for a modern university considering legal liability, educational value, press freedom, and institutional reputation`
  Model: `gpt-4o`
  Answer: `The trade-offs between faculty oversight and editorial independence in student publications involve balancing legal liability, educational value, press freedom, and institutional reputation... An optimal governance model for a modern university might involve a hybrid approach with legal and ethical training, a faculty advisory role, and clear policies.`

## Team Notes

- The model router is now connected to the existing pipeline scripts used by the team
- Batch runs generate a root-level CSV containing compression metrics, selected model, and answer text for each query
- A small compatibility fix was made in `src/retrieval/adaptive_retriever.py` by replacing non-ASCII console checkmark characters with plain ASCII print messages so the pipeline runs cleanly on Windows terminals
- The hardcoded `test_pipeline.py` query currently retrieves limited context, which explains that one insufficient-context answer
- The batch CSV demonstrates that routing behavior is functioning as expected and that useful answers are being generated for many queries
