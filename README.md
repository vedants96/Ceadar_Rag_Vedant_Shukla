**Hybrid Retrieval-Augmented Generation (RAG) for Technical Document QA**

This project explores how retrieval and generation can work together to answer research-style technical questions accurately. The focus is on building a hybrid retriever that merges semantic understanding (dense embeddings) with exact keyword precision (sparse search), and presenting it through a simple Streamlit UI deployed on Hugging Face Spaces for interactive exploration.

**Project Overview**

At its core, the system reads multiple research papers (for example, the Transformer and DeepSeek papers) and turns them into a searchable knowledge base. When a user asks a question, the retriever locates the most relevant text chunks and passes them to a generator that composes a concise, cited answer.

The Hugging Face Space deployment makes the entire process accessible via browser, showing how hybrid retrieval and generation behave on real technical queries.

**System Design**

- Embedding and Indexing

Each document is chunked and converted into vectors using the E5-small-v2 model. E5 is instruction-tuned, meaning it expects input prefixes like query: and passage:—a small but impactful detail that aligns questions and passages in the same semantic space. Embeddings are L2-normalized and stored in a FAISS index for fast nearest-neighbor search. Parallelly, a BM25 index is built over the raw text to capture rare tokens, numbers, and acronyms that embeddings sometimes overlook.

- Hybrid Retrieval

Both FAISS and BM25 results are min-max normalized and blended
(0.6 × dense + 0.4 × sparse).
For numeric or tabular data, a small additional boost is applied to prioritize factual precision. To avoid redundancy among top candidates, Maximal Marginal Relevance (MMR) re-ranks results, balancing relevance with diversity.

- Generation

The retrieved chunks are then passed to a lightweight language model to compose the answer, ensuring each claim includes a citation reference (e.g., [A: p.8 | table]).

**Query Design & Evaluation Strategy**
  
To evaluate how well the system retrieves and grounds information, I created a small GOLD query set that intentionally covered different types of information needs.
Each query was designed to test a unique retrieval behaviour — from conceptual understanding to factual accuracy and summarization ability — ensuring the system’s robustness across varied question styles.

The first query focused on conceptual understanding:

“What is multi-head attention and why is it useful?”
This type of question evaluates whether the retriever can recognize and fetch explanatory sections that describe underlying mechanisms or theories, even when phrased in natural language rather than as a direct keyword match. It primarily tests the semantic power of the embedding model and how well it captures meaning beyond literal words.

The second query targeted numeric and tabular retrieval:

“Report the BLEU result for WMT14 En-De from the Transformer paper.”
Such queries check if the hybrid approach correctly surfaces factual details like scores, percentages, or metrics often buried in tables or structured text. Dense embeddings alone may miss these, so this query helped validate the contribution of BM25 and the table boost in accurately locating numeric data.

The third query tested reasoning and summarization:

“Summarize how DeepSeek improves reasoning in brief.”
Unlike the others, this one demanded more abstract comprehension. It was meant to observe if the system could gather relevant fragments describing improvement strategies or reasoning mechanisms, even when those phrases were expressed differently in the text. This kind of query mirrors real-world scenarios where users ask for concise insights rather than verbatim facts.

Together, these three queries provided a balanced evaluation—covering conceptual, factual, and reasoning-based information retrieval.
They also demonstrated that the system isn’t tuned for a single query style but is capable of handling a range of question types commonly seen in research or technical analysis.

**Results Snapshot**

<img width="1060" height="138" alt="image" src="https://github.com/user-attachments/assets/c8c858fc-61e1-42fc-bf18-330a9b56d7f3" />

All queries achieved perfect retrieval at top-k, showing the hybrid strategy balances semantic and lexical precision well.
Phrase coverage varied with question type—expected for conceptual paraphrases—but every generated response contained proper citation links.

**Streamlit Interface and Deployment**

A simple Streamlit web interface was built to make experimentation intuitive.
The UI lets users:

Enter a custom query

View the retrieved document snippets with similarity scores

Inspect the generated, citation-aware answer

The frontend communicates with the backend (hybrid retriever + generator) via clean function calls, so the entire system runs seamlessly inside a single app.

The complete application was deployed on Hugging Face Spaces, providing an accessible demo environment without local setup.
Users can open the Space, type their own research-style questions, and watch how the retriever surfaces relevant content in real time.

This deployment not only demonstrates the technical pipeline but also showcases real-world usability—how hybrid RAG can be shared as a reproducible, hosted tool.

**Insights & Reflections**

Working end-to-end—from embeddings to web deployment—highlighted how retrieval design choices affect user perception.
Dense models capture meaning; sparse models catch exact matches; normalization and MMR glue them into a cohesive, explainable system.
Deploying it interactively on Hugging Face reinforced the importance of responsiveness, reproducibility, and transparency (citations visible to users).

The project delivers both strong retrieval metrics and an approachable interface, showing a solid grasp of the underlying information-retrieval concepts, evaluation logic, and deployment workflow.
