# Group Project Update 2
**BSAN 765: AI for Business | Spring 2026 | Group 1**
**Due: March 29th, 2026**

---

## Project Title
Cost-Optimized RAG System: Reducing Enterprise AI Costs Through Adaptive Query Routing

---

## 1. Data Sourcing and Preparation (30 points)

### Dataset Acquired

We used the Stanford Question Answering Dataset (SQuAD) available on HuggingFace as our knowledge base. SQuAD is a reading comprehension dataset consisting of Wikipedia articles paired with questions and answers — making it ideal for testing a RAG pipeline.

From the full SQuAD dataset, we extracted 20 unique paragraphs from the University of Notre Dame Wikipedia article. Each paragraph was saved as a separate .txt document covering distinct topics:

- Campus architecture and religious buildings
- Student media and publications
- Congregation of Holy Cross and seminaries
- College of Engineering programs and degrees
- Undergraduate colleges and First Year Studies program
- Graduate degree programs and academic institutes

### Sample of Raw Data

doc_00.txt:

    Title: University_of_Notre_Dame

    Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection...

doc_04.txt:

    Title: University_of_Notre_Dame

    All of Notre Dame's undergraduate students are a part of one of the five undergraduate colleges at the university. The College of Arts and Letters was established as the university's first college in 1842 with the first degrees awarded in 1849...

test_queries.csv (sample rows):

    query_id | query                                                      | complexity | ground_truth
    0        | To whom did the Virgin Mary allegedly appear in 1858...    | simple     | Saint Bernadette Soubirous
    31       | Compare the missions of different Notre Dame publications  | medium     | Manual evaluation needed
    46       | Design a comprehensive strategy for a new university...    | complex    | Requires multi-step analysis

### Data Preparation Steps

1. Deduplication — Initial data loading pulled the first 20 SQuAD items which were only 4 unique paragraphs each repeated 5 times. We rewrote load_squad.py to extract 20 unique paragraphs by deduplicating on context before saving.

2. Document formatting — Each paragraph was saved as a .txt file with a title header (Title: University_of_Notre_Dame) followed by the paragraph content.

3. Chunking — Documents were split using LangChain's RecursiveCharacterTextSplitter with:
   - chunk_size = 1000 characters
   - chunk_overlap = 200 characters
   - Result: 20 documents split into 38 chunks, stored in ChromaDB

4. Query dataset — We compiled 50 labeled test queries:
   - 30 simple queries (factual, single-answer) from SQuAD with ground truth answers
   - 15 medium queries (comparative, relational) with manual evaluation criteria
   - 5 complex queries (analytical, multi-step) requiring deep reasoning

---

## 2. Prompt Engineering Experiments (40 points)

We experimented with two prompt strategies in the Model Router component, which receives a compressed context and query and calls an LLM for an answer.

### Strategy 1 — Zero-Shot Direct Prompt

Prompt used:

    Answer the following question based on the context provided.

    Context: {compressed_context}
    Question: {query}

Model response for query "What is the oldest structure at Notre Dame?":

    The oldest structure at Notre Dame is the Main Building.

Result: Incorrect. The expected answer is Old College. The model picked the most prominent building mentioned in the compressed context rather than reasoning carefully about the word "oldest." The lack of explicit instruction to rely only on context caused hallucination.

---

### Strategy 2 — Role-Playing + Instruction Prompt (Current Implementation)

Prompt used:

    System: You answer questions using only the provided context. If the context is
    insufficient, say that clearly instead of guessing.

    Human: Question: {query}

            Context:
            {compressed_context}

            Answer the question as concisely as possible.

Model response for query "What is the oldest structure at Notre Dame?":

    Old College is the oldest building on campus, located near the shore of St. Mary Lake.

Result: Correct. By assigning a role (context-only answerer) and explicitly instructing the model not to guess, the response became grounded in the provided context. This strategy also produced cleaner "Insufficient context" responses for queries where the relevant document was not retrieved, rather than hallucinated answers.

### Which Strategy Worked Better and Why

Strategy 2 (Role-Playing + Instruction) consistently outperformed Strategy 1. The key reasons:

- The system role anchors the model to context-only answering, reducing hallucination
- The explicit "say clearly if insufficient" instruction produced honest "I don't know" responses rather than wrong answers
- Conciseness instruction reduced verbose outputs that padded short answers with unnecessary text

This is especially important for our cost-optimization goal — a model that admits it lacks context is preferable to one that produces convincing but incorrect answers, which would undermine the Confidence Checker's ability to validate responses.

---

## 3. Vector Database and Embedding Strategy (30 points)

### Embedding Model

We are using OpenAI's text-embedding-ada-002 model via the LangChain OpenAIEmbeddings wrapper.

Reason for selection:
- Strong semantic understanding for question-answer style retrieval tasks
- Widely benchmarked and proven in RAG applications
- Compatible with our existing OpenAI API setup, avoiding additional integration overhead
- Produces 1536-dimensional vectors with excellent cosine similarity performance

### Vector Database

We are using ChromaDB as our vector store, persisted locally at ./chroma_db.

Reason for selection:
- Lightweight and easy to run locally with no external service required
- Native integration with LangChain, reducing implementation effort
- Supports persistent storage so the database does not need to be rebuilt on every run
- Well-suited for our prototype scale (38 chunks, 20 documents)

For production deployment, we would evaluate Pinecone or Milvus for scalability.

### Chunking Strategy

We split documents using LangChain's RecursiveCharacterTextSplitter with the following configuration:

    chunk_size = 1000 characters
    chunk_overlap = 200 characters

Reasoning:
- chunk_size of 1000 captures enough context per chunk for semantic meaning without exceeding token limits when multiple chunks are combined
- chunk_overlap of 200 ensures sentences that span chunk boundaries are not split and lose meaning
- Recursive splitting prioritizes splitting on paragraph breaks, then sentences, then words — preserving natural language boundaries

After chunking, 20 documents produced 38 chunks. Each chunk is embedded and stored in ChromaDB with its source document metadata.

### Retrieval Strategy

Chunk retrieval is adaptive based on query complexity, implemented in AdaptiveRetriever:

    Simple query  → retrieve top 3 chunks (k=3)
    Medium query  → retrieve top 5 chunks (k=5)
    Complex query → retrieve top 10 chunks (k=10)

This adaptive k-selection is a core part of our cost optimization — simpler queries retrieve fewer chunks, reducing the token count passed to the LLM.

---

## Current Progress Summary

| Component          | Status   | Owner   | File                          |
|--------------------|----------|---------|-------------------------------|
| Query Analyzer     | Done     | Deepa   | src/query_analyzer.py         |
| Adaptive Retrieval | Done     | Karthik | src/adaptive_retriever.py     |
| Context Compression| Done     | Anh     | src/context_compression.py    |
| Model Router       | Done     | Gowri   | src/model_router.py           |
| Confidence Checker | Pending  | TBD     | src/confidence_checker.py     |
| Integrated Pipeline| Done     | Deepa   | pipeline.py                   |

### Pipeline Flow

    User Query
        |
        v
    Query Analyzer       — classifies as simple / medium / complex
        |
        v
    Adaptive Retrieval   — fetches 3 / 5 / 10 chunks from ChromaDB
        |
        v
    Context Compression  — keeps top 2 / 4 / 6 sentences via TF-IDF ranking
        |
        v
    Model Router         — routes to gpt-3.5-turbo / gpt-4o-mini / gpt-4o
        |
        v
    Confidence Checker   — (in progress) validates answer quality
        |
        v
    Final Answer

### Evaluation Results (run_all_queries.py on 50 queries)

- Average compression ratio: 85-93% token reduction for simple/medium queries
- Model routing accuracy: 100% (correct model selected for all 30 simple queries verified)
- Answer accuracy on simple queries with ground truth: 53% (16/30 correct)
- Key issue identified: retrieval returning semantically similar but contextually wrong chunks for some queries — being addressed by improving chunk quality and will be further handled by the Confidence Checker

---

## Next Steps

1. Complete the Confidence Checker to validate and retry low-confidence answers
2. Evaluate cost savings vs GPT-4-only baseline across all 50 queries
3. Build evaluation dashboard using compression_results.csv output
