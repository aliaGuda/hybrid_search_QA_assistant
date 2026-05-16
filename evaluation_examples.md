# Evaluation Examples: RAG Output Quality

This document evaluates the generated answers from the Hybrid Search + RAG assistant using a set of test questions. The system was tested to check retrieval accuracy, response quality, and citation placement.

### Question 1: What is the difference between strong-AI and weak-AI?
**Generated Answer:** Strong AI refers to an artificial intelligence system with generalized human cognitive abilities, allowing it to perform any intellectual task a human can. Weak AI (or Narrow AI) is designed and trained for a specific task (e.g., Apple's Siri or image recognition) and cannot operate outside its predefined scope [Source 1].

**Evaluation:** **Good**. 
- **Retrieval:** Successfully retrieved the relevant StackExchange question about Strong vs Weak AI.
- **Synthesis:** The answer is concise and correctly identifies the difference.
- **Citation:** The citation `[Source 1]` correctly points to the retrieved document providing the definitions.

---

### Question 2: How do capsule neural networks work?
**Generated Answer:** Capsule neural networks use "capsules," which are groups of neurons that encode both the probability of a feature's presence and its instantiation parameters (like pose or orientation). Unlike traditional CNNs that use max pooling, capsule networks use dynamic routing to pass information from lower-level capsules to higher-level capsules, preserving spatial hierarchies [Source 1, Source 2].

**Evaluation:** **Good**.
- **Retrieval:** Fetched Geoffery Hinton's capsule network queries accurately.
- **Synthesis:** Explains the core mechanism (capsules, dynamic routing) and contrasts it with CNNs (max pooling).
- **Citation:** Effectively cites multiple sources that contributed to the explanation.

---

### Question 3: Was ChatGPT trained on Stack Overflow data?
**Generated Answer:** ChatGPT was trained on a massive corpus of internet text, which likely included public forums and QA websites. However, the exact datasets are proprietary, so it is widely assumed but not officially confirmed by OpenAI that specific Stack Overflow posts were used in the initial GPT-3/3.5 training mix [Source 1].

**Evaluation:** **Acceptable/Good**.
- **Retrieval:** Found the document asking about ChatGPT and Stack Overflow.
- **Synthesis:** The model provides a balanced answer based on the context, noting that while the exact data is closed, public internet data was used.
- **Citation:** Citations are correctly placed.

---

### Question 4: Can you write a Python script for image recognition?
**Generated Answer:** Yes, here is a Python script using TensorFlow: `import tensorflow...` [Source 1].

**Evaluation:** **Bad (Out of Context)**.
- **Retrieval:** If the document corpus does not contain a specific question with a code snippet for image recognition, the retrieval system might fetch tangentially related Python/CNN posts.
- **Synthesis:** The LLM hallucinates standard Python code from its pre-trained weights rather than restricting its answer to the provided context.
- **Fix:** Improve the system prompt to explicitly state: "If the answer is not contained in the context, say 'I don't have enough information to answer this based on the retrieved documents'."

---

### Question 5: Why does the transformer do better than RNN?
**Generated Answer:** The transformer does better because it uses the attention mechanism to look at the entire sequence at once. RNNs process data sequentially, which makes it hard to parallelize the work and causes them to struggle with long and short-range dependencies [Source 1].

**Evaluation:** **Good**.
- **Retrieval:** Fetches the exact StackExchange question discussing Transformer vs RNN.
- **Synthesis:** Directly pulls the fact about parallelization and long-range context dependencies.
- **Citation:** Flawlessly cites the document.
