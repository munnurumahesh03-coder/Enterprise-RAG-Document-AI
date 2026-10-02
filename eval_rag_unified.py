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
        "retrieved_context": "Self-attention connects all words directly, improving speed and processing.",
        "bot_answer": "Self-attention connects words directly and improves speed.",
        "ground_truth": "It connects words directly to improve processing speed."
    },
    {
        "id": "Test_002",
        "question": "What is the speed of RNNs?",
        "retrieved_context": "Self-attention connects all words directly.", # Completely wrong context
        "bot_answer": "I cannot answer this based on the context.",
        "ground_truth": "RNNs process sequentially and are slower."
    },
    {
        "id": "Test_003",
        "question": "Who invented self-attention?",
        "retrieved_context": "Self-attention connects all words directly, improving speed.",
        "bot_answer": "Self-attention was invented by Elon Musk.", # Massive Hallucination
        "ground_truth": "Google researchers invented it in 2017."
    }
]

def evaluate_pipeline():
    print("🚀 STARTING UNIFIED AI EVALUATION (DECIMAL SCORING)...\n")
    results_log = []
    
    for item in evaluation_dataset:
        print(f"🔍 Question: {item['question']}")
        
        # 3. THE UNIFIED PROMPT (All 3 metrics mixed together + Decimal Scoring!)
        eval_prompt = f"""
        You are an expert AI QA Tester. Evaluate the RAG pipeline output.
        
        QUESTION: {item['question']}
        RETRIEVED CONTEXT (From DB): {item['retrieved_context']}
        BOT ANSWER: {item['bot_answer']}
        GROUND TRUTH: {item['ground_truth']}
        
        Provide a JSON output with exactly these 4 keys. Score the metrics as decimals between 0.0 and 1.0 (e.g., 0.85, 0.5, 1.0):
        - "context_relevance": Score 0.0 to 1.0 based on how much of the required information is in the RETRIEVED CONTEXT.
        - "faithfulness": Score 0.0 to 1.0 based on how strictly the BOT ANSWER relies on the CONTEXT (1.0 = no hallucinations).
        - "accuracy": Score 0.0 to 1.0 based on how well the BOT ANSWER matches the GROUND TRUTH (1.0 = perfectly accurate).
        - "reason": A short explanation justifying the decimal scores.
        """
        
        try:
            # Call the Judge
            response = judge_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": eval_prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            scores = json.loads(response.choices[0].message.content)
            
            # Fetch the decimal scores safely
            context_relevance = float(scores.get("context_relevance", 0.0))
            faithfulness = float(scores.get("faithfulness", 0.0))
            accuracy = float(scores.get("accuracy", 0.0))
            reason = scores.get("reason", "No reason provided")
            
            # Print the beautiful unified scores
            print(f"   - Context Relevance : {context_relevance:.2f}")
            print(f"   - Faithfulness      : {faithfulness:.2f}")
            print(f"   - Accuracy          : {accuracy:.2f}")
            
            # Smart Threshold Logic (If it's below 0.7, flag it as an issue!)
            issues = []
            if context_relevance < 0.7:
                issues.append("Retrieval Miss")
            if faithfulness < 0.7:
                issues.append("Hallucination")
            if accuracy < 0.7:
                issues.append("Low Accuracy")
                
            if not issues:
                print("   ✅ PASS: High quality response.")
                status = "Pass"
            else:
                status = " | ".join(issues)
                print(f"   ❌ FAILED: {status}")
                print(f"   💡 JUDGE REASON: {reason}")
            
            print("-" * 50)
                
            # Log all scores for the CSV
            results_log.append({
                "Question": item['question'], 
                "Status": status, 
                "Context_Relevance": context_relevance,
                "Faithfulness": faithfulness,
                "Accuracy": accuracy, 
                "Reason": reason
            })
                
        except Exception as e:
            print(f"   ⚠️ API Error: {e}\n")

    # 4. Export to CSV
    csv_file = "unified_evaluation_report.csv"
    if results_log:
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["Question", "Status", "Context_Relevance", "Faithfulness", "Accuracy", "Reason"])
            writer.writeheader()
            writer.writerows(results_log)
        print(f"📊 Evaluation complete! Results saved to {csv_file}")

if __name__ == "__main__":
    evaluate_pipeline()
