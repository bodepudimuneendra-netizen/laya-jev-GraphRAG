To build a complete Agentic GraphRAG system, you are essentially building a software assembly line. At every station on this line, data must be evaluated, routed, or discarded.

By using a System One decision model (like local Laya or the TypeSafe Jev API), you map every single graph operation to one of three mathematical primitives: **`Choice`** (categorical routing), **`Score`** (ordinal ranking), and **`Noul`** (yes/no probability).

Here is the exhaustive master list of every function and use case across the entire GraphRAG lifecycle, detailing exactly how nodes, relationships, and queries are evaluated.

---

### Phase 1: Ingestion & Graph Maintenance (Nodes & Relationships)

*This phase happens before a user ever asks a question. The goal is to build a mathematically dense, hallucination-free knowledge graph.*

| Function / Use Case | The Goal | The Engine & Primitive | Execution Logic |
| --- | --- | --- | --- |
| **Semantic Chunking** | Prevent cutting sentences in half when ingesting documents. | **Laya/Jev (`Noul`)** | Pass a sliding window of sentences. Ask: *"Is there a major topic shift here?"* If `Noul` passes the boundary threshold, slice the chunk. |
| **Entity Extraction** | Pull raw nodes and edges from text. | **Generative LLM (System 2)** | You still need an 8B/70B model to read the text and output a JSON list of `[Entity A] - [Relationship] -> [Entity B]`. |
| **Entity Disambiguation** | Prevent "Graph Bloat" (e.g., merging "Apple" and "Apple Inc."). | **Laya/Jev (`Noul`)** | Run vector similarity. If two nodes are highly similar, pass their contexts to the model. Ask: *"Do these describe the exact same entity?"* If `Noul` passes the merge threshold, execute a Neo4j merge. |
| **Edge Verification** | Delete hallucinated relationships extracted by the LLM. | **Laya/Jev (`Score`)** | Run a continuous background loop on new edges. Ask: *"Score the logical validity of this relationship based on the source text."* Delete edges that score below a safety threshold. |
| **Ontology Alignment** | Standardize relationship names (e.g., changing "WORKED_AT" to "EMPLOYED_BY"). | **Laya/Jev (`Choice`)** | Pass a raw edge to the model with your strict database schema as the options. The model uses `Choice` to snap the raw text to your predefined relationship ontology. |
| **Community Detection** | Cluster dense groups of nodes together for global summaries. | **Classical Math (Leiden/Core)** | AI is not used here. Neo4j's Graph Data Science (GDS) library calculates network modularity to group nodes. |

---

### Phase 2: Pre-Traversal (Query Prep)

*The user asks a question. Before diving into the graph, the system must decide the search strategy and find the perfect starting point.*

| Function / Use Case | The Goal | The Engine & Primitive | Execution Logic |
| --- | --- | --- | --- |
| **Intent Routing** | Decide which algorithm to use so you don't waste compute. | **Laya/Jev (`Choice`)** | Pass the user query. Ask the model to make a `Choice` between: `[Local_Lookup, Multi_Hop, Global_Theme]`. This triggers the correct downstream Python function. |
| **Seed Retrieval** | Find candidate starting nodes in the graph. | **Classical Math (ColBERT/BM25)** | Use a hybrid dense/sparse vector search to instantly retrieve the top 20 candidate starting nodes. |
| **Semantic Seed Validation** | Pick the absolute best starting node from the candidates. | **Laya/Jev (`Score`)** | Pass the 20 candidates to the model. Ask: *"Score how relevant this node is as a starting point for the query."* Drop the top 2 highest-scoring nodes into your A* search queue. |

---

### Phase 3: Traversal (In-Flight Evaluation)

*This is the active multi-hop search. The algorithm navigates from node to node, scoring edges dynamically to find the logical chain of evidence.*

| Function / Use Case | The Goal | The Engine & Primitive | Execution Logic |
| --- | --- | --- | --- |
| **Neighborhood Fetch** | See what is connected to the current node. | **Neo4j (Cypher Bolt)** | Fetch all outgoing edges and target nodes in 3 milliseconds via index-free adjacency. |
| **Edge Scoring (Heuristic)** | Decide which path to take. | **Laya/Jev (`Score`)** | Pass the connecting edges to the model. Ask: *"Score how relevant this path is to answering the user query."* |
| **Structural Anchoring** | Prevent wandering into dead-end nodes. | **Neo4j (PageRank)** | Pull the pre-calculated PageRank of the target node. Combine this with the Laya `Score` to determine priority. |
| **Path Pruning** | Keep memory usage low and drop bad paths. | **Classical Math (Beam Cutoff)** | Sort the frontier queue by priority score. Delete any path that falls outside your Beam Width (e.g., keep only the top 3 paths). |
| **Early Termination Check** | Stop the search if you found the answer, saving compute. | **Laya/Jev (`Noul`)** | At each hop, evaluate the accumulated context path. Ask: *"Does this context fully answer the user query?"* If `Noul` passes the stop threshold, terminate the search early. |

---

### Phase 4: Post-Traversal & Evaluation

*The search is over. The system must verify the data, resolve conflicts, and hand it off to the generative LLM to write the final response.*

| Function / Use Case | The Goal | The Engine & Primitive | Execution Logic |
| --- | --- | --- | --- |
| **Context Reranking** | Strip out any tangential nodes collected during the search. | **Laya/Jev (`Score`)** | Pass the final selected subgraph. Score each node's direct utility to the query. Filter out the lowest-scoring nodes to save LLM context tokens. |
| **Conflict Resolution** | Handle contradictory data (e.g., Node A says revenue was $5M, Node B says $6M). | **Laya/Jev (`Choice`)** | Pass both source chunks to the model. Ask: *"Which source is more credible/recent?"* The model uses `Choice` to declare the winner. |
| **Hallucination Gatekeeper** | Ensure the retrieved context is actually safe to use. | **Laya/Jev (`Noul`)** | Ask: *"Is the retrieved context factually sufficient to answer the question without guessing?"* If `Noul` fails the safety check, trigger a fallback to a web search. |
| **Final Answer Synthesis** | Write the human-readable answer. | **Generative LLM (System 2)** | Pass the pruned, verified subgraph text to Llama-3.1-8B to generate the final response. |
| **Citation Verification** | Ensure the LLM didn't hallucinate during synthesis. | **Laya/Jev (`Noul`)** | Pass the LLM's final answer and the subgraph context to the model. Ask: *"Is every claim in the generated answer strictly supported by the context?"* If `Noul` confidence is high, return to the user. |

### The System Architecture Summary

By mapping the system this way, your pipeline is perfectly modular. **Neo4j** handles all the structural math (PageRank, Community Detection, Fetching). **Llama 3.1** handles all the text translation (Entity Extraction, Final Synthesis). **Laya/Jev** acts as the high-speed CPU connecting them—executing every routing, validation, and filtering task in the middle using just three primitives.