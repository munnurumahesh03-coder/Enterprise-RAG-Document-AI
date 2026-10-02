import os
import sys
import warnings
from unittest.mock import MagicMock

# 1. Silence warnings and Monkey Patch the VertexAI bug
warnings.filterwarnings("ignore")
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()
sys.modules['langchain_community.llms'] = MagicMock()

import pandas as pd
from datasets import Dataset

from ragas import evaluate
# INCLUDED ALL 3 METRICS: Faithfulness, Recall, and Accuracy (answer_correctness)
from ragas.metrics import faithfulness, context_recall, answer_correctness

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# 2. Use Llama-3 (Better at JSON formatting for Ragas)
raw_llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model_name="openai/gpt-oss-120b",
    temperature=0
)
eval_llm = LangchainLLMWrapper(raw_llm)

# Embeddings are required for answer_correctness (Accuracy)
raw_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
eval_embeddings = LangchainEmbeddingsWrapper(raw_embeddings)

def run_ragas_evaluation():
    print("🚀 RUNNING FULL RAGAS EVALUATION (WITH ACCURACY)...\n")

    data_samples = {
        "question": ["What is self-attention?", "What is the speed of RNNs?"],
        "answer": ["Self-attention connects all words directly.", "RNNs are very fast."],
        "contexts": [
            ["Self-attention connects all words directly, improving speed."],
            ["Self-attention connects all words directly."] # Bad context for RNNs
        ],
        "ground_truth": ["It connects all words directly.", "RNNs process sequentially and are slower."]
    }
    dataset = Dataset.from_dict(data_samples)

    # 3. RUN ALL 3 METRICS
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_recall, answer_correctness],
        llm=eval_llm,
        embeddings=eval_embeddings
    )

    # 4. Safe Pandas Extraction (Bypassing the KeyError)
    df = result.to_pandas()

    print("\n==========================================")
    print("📊 RAGAS EVALUATION RESULTS")
    print("==========================================\n")

    for i in range(len(dataset)):
        # Pull the question from the original dataset so it CANNOT fail
        question = dataset['question'][i]

        # Pull the scores safely from the dataframe
        recall_score = df['context_recall'][i] if 'context_recall' in df.columns else 0
        faith_score = df['faithfulness'][i] if 'faithfulness' in df.columns else 0
        accuracy_score = df['answer_correctness'][i] if 'answer_correctness' in df.columns else 0

        print(f"🔍 Question: {question}")
        print(f"   - Context Recall (Database) : {recall_score:.2f}")
        print(f"   - Faithfulness (LLM)        : {faith_score:.2f}")
        print(f"   - Accuracy (Correctness)    : {accuracy_score:.2f}")

        if recall_score < 0.7:
            print("   ❌ ISSUE: Low Recall (Retrieval Failure - Check ChromaDB)")
        elif faith_score < 0.7:
            print("   🚨 ISSUE: Low Faithfulness (Model Hallucination - Check Prompt)")
        elif accuracy_score < 0.7:
            print("   ⚠️ ISSUE: Low Accuracy (Answer doesn't match ground truth)")
        else:
            print("   ✅ PASS: Accurate and Faithful")
        print("-" * 40)

    df.to_csv("ragas_full_report.csv", index=False)
    print("📁 Evaluation complete! Saved to ragas_full_report.csv")

if __name__ == "__main__":
    run_ragas_evaluation()
