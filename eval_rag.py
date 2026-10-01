import os
import json
import csv
from groq import Groq

# 1. Initialize the Judge LLM
judge_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# 2. Curate Ground Truth Dataset
evaluation_dataset = [
    {
        "id": "Test_001",
        "question": "What is self-attention?",
        "retrieved_context": "Self-attention connects all words directly, improving speed.",
        "bot_answer": "Self-attention connects all words directly.",
        "ground_truth": "It connects words directly."
    },
    {
        "id": "Test_002",
        "question": "What is the speed of RNNs?",
        "retrieved_context": "Self-attention connects all words directly.", # Wrong context!
        "bot_answer": "I don't know the answer based on the context.",
        "ground_truth": "RNNs process sequentially and are slower."
    },
    {
        "id": "Test_003",
        "question": "Who invented self-attention?",
        "retrieved_context": "Self-attention connects all words directly, improving speed.",
        "bot_answer": "Self-attention was invented by Elon Musk.", # Hallucination!
        "ground_truth": "Google researchers invented it."
    }
]

def evaluate_pipeline():
    print("🚀 STARTING AI EVALUATION SET (LLM-AS-A-JUDGE)...\n")
    results_log = []
    
    for item in evaluation_dataset:
        print(f"🔍 Question: {item['question']}")
        
        # 3. Master Prompt requesting 3 distinct metrics
        eval_prompt = f"""
        You are an expert AI QA Tester. Evaluate the RAG pipeline output.
        
        QUESTION: {item['question']}
        RETRIEVED CONTEXT (From DB): {item['retrieved_context']}
        BOT ANSWER: {item['bot_answer']}
        GROUND TRUTH: {item['ground_truth']}
        
        Provide a JSON output with exactly these 4 keys:
        - "context_relevance": 1 if the RETRIEVED CONTEXT contains enough relevant info to answer the QUESTION, else 0.
        - "faithfulness": 1 if the BOT ANSWER is strictly based on the RETRIEVED CONTEXT (no hallucinations), else 0.
        - "correctness": 1 if the BOT ANSWER matches the meaning of the GROUND TRUTH, else 0.
        - "reason": A short explanation of the failures, if any.
        """
        
        try:
            response = judge_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": eval_prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            raw_scores = json.loads(response.choices[0].message.content)
            
            # 4. Graceful JSON Validation (Defaults to 0 if LLM misses a key)
            context_relevance = raw_scores.get("context_relevance", 0)
            faithfulness = raw_scores.get("faithfulness", 0)
            correctness = raw_scores.get("correctness", 0)
            reason = raw_scores.get("reason", "No reason provided")
            
            issues = []
            
            # 5. Independent Checks (Catching multiple issues at once)
            if context_relevance == 0:
                print("   ❌ RETRIEVAL MISS: Context lacks relevant info.")
                print("   🔧 ACTION: Investigate retrieval settings (e.g., chunking, embeddings, top-k).")
                issues.append("Retrieval Miss")
                
            if faithfulness == 0:
                print("   🚨 HALLUCINATION: Answer contains facts not in context.")
                print("   🔧 ACTION: Review prompt guardrails or model constraints.")
                issues.append("Hallucination")
                
            if correctness == 0:
                print("   ⚠️ INCORRECT: Answer does not match ground truth.")
                print("   🔧 ACTION: Investigate LLM reasoning or prompt clarity.")
                issues.append("Incorrect Answer")
                
            if not issues:
                print("   ✅ PASS: Context relevant, answer faithful and correct.")
                status = "Pass"
            else:
                status = " | ".join(issues)
                print(f"   💡 JUDGE REASON: {reason}")
            
            print("") 
                
            # Log all scores independently
            results_log.append({
                "Question": item['question'], 
                "Status": status, 
                "Context_Relevance": context_relevance,
                "Faithfulness": faithfulness,
                "Correctness": correctness, 
                "Reason": reason
            })
                
        except Exception as e:
            print(f"   ⚠️ API Error: {e}\n")

    # 6. Export comprehensive CSV
    csv_file = "evaluation_report.csv"
    if results_log:
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["Question", "Status", "Context_Relevance", "Faithfulness", "Correctness", "Reason"])
            writer.writeheader()
            writer.writerows(results_log)
        print(f"📊 Evaluation complete! Results saved to {csv_file}")

if __name__ == "__main__":
    evaluate_pipeline()
