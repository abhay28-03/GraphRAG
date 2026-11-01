import os
import json
import numpy as np
import networkx as nx
from typing import List
from pydantic import BaseModel, Field
from groq import Groq
from fastembed import TextEmbedding

# -------------------------------------------------------------
# 0. Clients Initialization (100% Free)
# -------------------------------------------------------------
# Groq client uses GROQ_API_KEY from the environment
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
# Runs lightweight, local embeddings locally on CPU/GPU (No API needed)
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


# -------------------------------------------------------------
# 1. Pydantic Schemas for Structured Extraction
# -------------------------------------------------------------
class Entity(BaseModel):
    name: str = Field(description="Name of the entity, normalized to Title Case.")
    category: str = Field(description="Category (e.g., Person, Facility, Technology).")

class Relationship(BaseModel):
    source: str = Field(description="Source entity name.")
    target: str = Field(description="Target entity name.")
    relation: str = Field(description="Action/verb describing the connection.")

class KnowledgeGraphExtraction(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]


# -------------------------------------------------------------
# 2. Embedding Utilities
# -------------------------------------------------------------
def get_embedding(text: str) -> np.ndarray:
    """Computes free local vector embedding."""
    return list(embed_model.embed([text]))[0]

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm != 0 else 0.0


# -------------------------------------------------------------
# 3. GraphRAG Engine
# -------------------------------------------------------------
class FreeGraphRAG:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_embeddings = {}
        self.raw_chunks = []
        self.chunk_embeddings = []

    def ingest_documents(self, documents: List[str]):
        """Extracts entities and relationships using Groq's free LLM tier."""
        for doc in documents:
            self.raw_chunks.append(doc)
            self.chunk_embeddings.append(get_embedding(doc))

            # Prompt Groq with JSON Schema enforcement
            schema_json = json.dumps(KnowledgeGraphExtraction.model_json_schema())
            system_prompt = (
                "You are a Knowledge Graph extraction engine. "
                "Extract entities and their explicit relationships from the text. "
                f"Respond ONLY with valid JSON conforming to this schema:\n{schema_json}"
            )

            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Text:\n{doc}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )

            try:
                raw_json = json.loads(response.choices[0].message.content)
                extracted_data = KnowledgeGraphExtraction.model_validate(raw_json)
            except Exception as e:
                print(f"Extraction parsing fallback: {e}")
                continue

            # Populate the graph
            for entity in extracted_data.entities:
                norm_name = entity.name.strip().title()
                if not self.graph.has_node(norm_name):
                    self.graph.add_node(norm_name, category=entity.category)
                    self.entity_embeddings[norm_name] = get_embedding(norm_name)

            for rel in extracted_data.relationships:
                src = rel.source.strip().title()
                tgt = rel.target.strip().title()
                self.graph.add_edge(src, tgt, relation=rel.relation)

    def find_seed_entities(self, query: str, top_k: int = 2) -> List[str]:
        """Finds entry point nodes via local cosine similarity."""
        query_vec = get_embedding(query)
        scored = [
            (node, cosine_similarity(query_vec, emb))
            for node, emb in self.entity_embeddings.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in scored[:top_k]]

    def traverse_subgraph(self, seeds: List[str], max_hops: int = 2) -> List[str]:
        """Multi-hop BFS traversal over extracted relationships."""
        triples = set()
        visited = set()

        for seed in seeds:
            if seed not in self.graph:
                continue
            queue = [(seed, 0)]
            visited.add(seed)

            while queue:
                curr, depth = queue.pop(0)
                if depth >= max_hops:
                    continue

                # Forward hops
                for neighbor in self.graph.successors(curr):
                    rel = self.graph[curr][neighbor].get("relation", "relates to")
                    triples.add(f"({curr}) -[{rel}]-> ({neighbor})")
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))

                # Backward hops
                for predecessor in self.graph.predecessors(curr):
                    rel = self.graph[predecessor][curr].get("relation", "relates to")
                    triples.add(f"({predecessor}) -[{rel}]-> ({curr})")
                    if predecessor not in visited:
                        visited.add(predecessor)
                        queue.append((predecessor, depth + 1))

        return list(triples)

    def answer_query(self, query: str) -> str:
        """Retrieves graph context and synthesizes answer using Groq."""
        seeds = self.find_seed_entities(query, top_k=2)
        graph_triples = self.traverse_subgraph(seeds, max_hops=3)

        graph_context = "\n".join(graph_triples) if graph_triples else "No direct path found."

        prompt = f"""Use the following extracted knowledge graph relations to answer the multi-hop question.

Knowledge Graph Paths:
{graph_context}

User Question:
{query}

Answer clearly with the reasoning path:"""

        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return completion.choices[0].message.content


# -------------------------------------------------------------
# 4. Multi-Hop Test Run
# -------------------------------------------------------------
if __name__ == "__main__":
    docs = [
        "Dr. Evelyn Reed is the lead researcher overseeing Project Hyperion.",
        "Project Hyperion has been deployed at the Atacama Observational Facility in Chile.",
        "The Atacama Observational Facility relies on the Helium-3 Cryogenic Cooling Unit for its sensors."
    ]

    print("Building local graph using Groq + FastEmbed...")
    rag = FreeGraphRAG()
    rag.ingest_documents(docs)

    print(f"\nExtracted Graph: {rag.graph.number_of_nodes()} nodes, {rag.graph.number_of_edges()} edges.")
    for u, v, d in rag.graph.edges(data=True):
        print(f"  ({u}) -> [{d.get('relation')}] -> ({v})")

    query = "What cooling unit is used by the facility hosting Evelyn Reed's project?"
    print(f"\nQuery: {query}")
    print("\nAnswer:")
    print(rag.answer_query(query))