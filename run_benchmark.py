import os
from core.engine import GraphRAGEngine

API_KEY = os.environ.get("GROQ_API_KEY")
engine = GraphRAGEngine(api_key=API_KEY)

test_corpus = [
    "AlphaCorp acquired security firm CypherTech in late 2024.",
    "CypherTech is the sole creator of the ZeroTrust-Mesh security protocol.",
    "The ZeroTrust-Mesh protocol requires hardware chips manufactured by Valeron Micro.",
    "Valeron Micro operates its primary semiconductor foundry in Dresden, Germany.",
    "Dr. Evelyn Reed oversees Project Hyperion at the Department of Applied Physics.",
    "Project Hyperion has been deployed at the Atacama Observational Facility in Chile.",
    "The Atacama Observational Facility relies on the Helium-3 Cryogenic Cooling Unit for its sensors."
]

print("[Setup] Ingesting evaluation corpus into GraphRAG engine...")
engine.ingest_documents(test_corpus)

eval_dataset = [
    {
        "query": "Where is the primary foundry of the chipmaker that supports CypherTech's security protocol?",
        "type": "multi_hop_local",
        "ground_truth_nodes": ["Valeron Micro", "Dresden, Germany", "Zerotrust-Mesh"]
    },
    {
        "query": "What cooling technology is utilized at the facility where Evelyn Reed's project is based?",
        "type": "multi_hop_local",
        "ground_truth_nodes": ["Atacama Observational Facility", "Helium-3 Cryogenic Cooling Unit"]
    },
    {
        "query": "What are the primary operational domains covered across the entire organization portfolio?",
        "type": "global_sensemaking",
        "ground_truth_nodes": ["Semiconductor", "Physics", "Cryogenic"]
    }
]

print("\n" + "=" * 90)
print("COMPREHENSIVE BENCHMARK: DENSE VECTOR RAG vs. HYBRID GRAPHRAG (LOCAL & GLOBAL)")
print("=" * 90)

naive_success = 0
graph_success = 0

for idx, item in enumerate(eval_dataset, 1):
    q = item["query"]
    gt = item["ground_truth_nodes"]
    q_type = item["type"]
    
    print(f"\n[Evaluation Case #{idx}] ({q_type.upper()})")
    print(f"Query: \"{q}\"")
    print(f"Required Ground-Truth Entities: {gt}")

    # 1. Naive Vector RAG Run
    naive_res = engine.query_naive_rag(q, top_k=2)
    naive_text = naive_res["answer"]
    naive_hits = sum(1 for node in gt if node.lower() in naive_text.lower())
    naive_score = (naive_hits / len(gt)) * 100
    if naive_score >= 60.0:
        naive_success += 1

    print(f"\n--- Naive Vector RAG (Context Recall Score: {naive_score:.1f}%) ---")
    print(f"Answer: {naive_text.strip()[:200]}...")

    # 2. GraphRAG Run (Routed Local vs Global)
    if q_type == "global_sensemaking":
        graph_res = engine.query_global(q)
    else:
        graph_res = engine.query_local(q)
        
    graph_text = graph_res["answer"]
    graph_hits = sum(1 for node in gt if node.lower() in graph_text.lower())
    graph_score = (graph_hits / len(gt)) * 100
    if graph_score >= 60.0:
        graph_success += 1

    print(f"\n--- GraphRAG Engine (Context Recall Score: {graph_score:.1f}%) ---")
    print(f"Answer: {graph_text.strip()[:200]}...")

    if "traversed_triples" in graph_res:
        print(f"Verified Evidence Triples ({len(graph_res['traversed_triples'])}):")
        for t in graph_res["traversed_triples"][:3]:
            print(f"   * {t}")
    print("-" * 90)

print(f"\nFINAL EVALUATION SUMMARY:")
print(f"Naive Vector RAG Success Rate : {naive_success}/{len(eval_dataset)} ({(naive_success/len(eval_dataset))*100:.1f}%)")
print(f"Hybrid GraphRAG Success Rate  : {graph_success}/{len(eval_dataset)} ({(graph_success/len(eval_dataset))*100:.1f}%)")
print("=" * 90)