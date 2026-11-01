import os
import json
import numpy as np
import networkx as nx
import community as community_louvain
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from groq import Groq
from fastembed import TextEmbedding
from neo4j import GraphDatabase

PRIMARY_MODEL = "openai/gpt-oss-20b"
FALLBACK_MODEL = "qwen/qwen3.8-27b"

class Entity(BaseModel):
    name: str = Field(description="Normalized name of entity, title case.")
    category: str = Field(description="Category (Person, Facility, Technology, Concept).")

class Relationship(BaseModel):
    source: str = Field(description="Subject entity.")
    target: str = Field(description="Object entity.")
    relation: str = Field(description="Predicated action or relationship.")

class KnowledgeGraphExtraction(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]


class GraphRAGEngine:
    def __init__(self, api_key: str = None, neo4j_uri: str = None, neo4j_user: str = "neo4j", neo4j_password: str = "graphragpassword"):
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY is not set.")
        self.groq_client = Groq(api_key=key)
        self.embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # Dual storage: NetworkX for in-memory graph algorithms + Neo4j for enterprise persistence
        self.graph = nx.DiGraph()
        self.entity_embeddings: Dict[str, np.ndarray] = {}
        self.raw_chunks: List[str] = []
        self.chunk_embeddings: List[np.ndarray] = []
        self.community_summaries: Dict[int, str] = {}

        # Neo4j Connection (Optional for local testing, required for production)
        self.neo4j_driver = None
        uri = neo4j_uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        try:
            self.neo4j_driver = GraphDatabase.driver(uri, auth=(neo4j_user, neo4j_password))
            self.neo4j_driver.verify_connectivity()
            print("[Neo4j] Connected successfully to persistent database.")
        except Exception:
            print("[Neo4j] Running in local in-memory mode (Neo4j standalone server unreachable).")

    def close(self):
        if self.neo4j_driver:
            self.neo4j_driver.close()

    def _call_llm(self, prompt: str, system_prompt: str = "", json_mode: bool = False) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_err = None
        for model in [PRIMARY_MODEL, FALLBACK_MODEL]:
            try:
                kwargs = {"model": model, "messages": messages, "temperature": 0.0}
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = self.groq_client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"All configured Groq models failed. Last error: {last_err}")

    def get_embedding(self, text: str) -> np.ndarray:
        return list(self.embed_model.embed([text]))[0]

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / norm) if norm != 0 else 0.0

    def resolve_entity(self, entity_name: str, threshold: float = 0.88) -> str:
        """Semantic Entity Resolution: Deduplicates and aliases entities using cosine distance."""
        name = entity_name.strip().title()
        if not self.entity_embeddings:
            return name
        if name in self.entity_embeddings:
            return name

        cand_vec = self.get_embedding(name)
        best_match = None
        best_score = -1.0

        for existing_node, emb in self.entity_embeddings.items():
            sim = self.cosine_sim(cand_vec, emb)
            if sim > best_score:
                best_score = sim
                best_match = existing_node

        if best_score >= threshold:
            return best_match
        return name

    def _persist_to_neo4j(self, src: str, tgt: str, rel: str, src_cat: str = "Entity", tgt_cat: str = "Entity"):
        if not self.neo4j_driver:
            return
        query = (
            "MERGE (a:Entity {name: $src}) "
            "ON CREATE SET a.category = $src_cat "
            "MERGE (b:Entity {name: $tgt}) "
            "ON CREATE SET b.category = $tgt_cat "
            "MERGE (a)-[r:RELATION {type: $rel}]->(b)"
        )
        try:
            with self.neo4j_driver.session() as session:
                session.run(query, src=src, tgt=tgt, rel=rel, src_cat=src_cat, tgt_cat=tgt_cat)
        except Exception as e:
            print(f"[Neo4j Sync Warning] {e}")

    def ingest_documents(self, documents: List[str]):
        """Ingests chunks, deduplicates entities, builds graph, and recalculates communities."""
        schema = json.dumps(KnowledgeGraphExtraction.model_json_schema())
        system_prompt = (
            "You are a Knowledge Graph Extraction Engine. Extract all entities and direct relationships. "
            f"Output strictly as a valid JSON object matching this schema:\n{schema}"
        )

        for doc in documents:
            if not doc.strip():
                continue
            self.raw_chunks.append(doc)
            self.chunk_embeddings.append(self.get_embedding(doc))

            try:
                user_prompt = f"Extract the entities and relationships from the following text into a JSON object:\n\n{doc}"
                content = self._call_llm(user_prompt, system_prompt, json_mode=True)
                data = KnowledgeGraphExtraction.model_validate(json.loads(content))

                for entity in data.entities:
                    canonical_name = self.resolve_entity(entity.name)
                    if not self.graph.has_node(canonical_name):
                        self.graph.add_node(canonical_name, category=entity.category)
                        self.entity_embeddings[canonical_name] = self.get_embedding(canonical_name)

                for rel in data.relationships:
                    canonical_src = self.resolve_entity(rel.source)
                    canonical_tgt = self.resolve_entity(rel.target)

                    if not self.graph.has_node(canonical_src):
                        self.graph.add_node(canonical_src, category="Entity")
                        self.entity_embeddings[canonical_src] = self.get_embedding(canonical_src)
                    if not self.graph.has_node(canonical_tgt):
                        self.graph.add_node(canonical_tgt, category="Entity")
                        self.entity_embeddings[canonical_tgt] = self.get_embedding(canonical_tgt)

                    self.graph.add_edge(canonical_src, canonical_tgt, relation=rel.relation)
                    self._persist_to_neo4j(canonical_src, canonical_tgt, rel.relation)
            except Exception as e:
                print(f"[Extraction Error] Chunk failed: {e}")

        # Update global community detection partition after ingestion
        self.build_global_communities()

    def build_global_communities(self):
        """Partitions the knowledge graph via Louvain and summarizes distinct thematic clusters."""
        if self.graph.number_of_nodes() < 2:
            return
        undirected = self.graph.to_undirected()
        partition = community_louvain.best_partition(undirected)
        
        clusters: Dict[int, List[str]] = {}
        for node, comm_id in partition.items():
            clusters.setdefault(comm_id, []).append(node)

        for comm_id, nodes in clusters.items():
            subgraph_edges = []
            for u in nodes:
                for v in self.graph.successors(u):
                    if v in nodes:
                        subgraph_edges.append(f"({u}) -[{self.graph[u][v].get('relation', 'related')}]-> ({v})")
            
            summary_prompt = (
                f"Synthesize an overarching 2-sentence structural theme for this knowledge community:\n"
                f"Entities: {', '.join(nodes)}\n"
                f"Key Relationships:\n" + "\n".join(subgraph_edges[:10])
            )
            try:
                self.community_summaries[comm_id] = self._call_llm(summary_prompt)
            except Exception:
                self.community_summaries[comm_id] = f"Community covering entities: {', '.join(nodes[:5])}"

    def find_seed_entities(self, query: str, top_k: int = 2) -> List[str]:
        q_vec = self.get_embedding(query)
        scores = [(node, self.cosine_sim(q_vec, emb)) for node, emb in self.entity_embeddings.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in scores[:top_k]]

    def traverse_subgraph(self, seeds: List[str], max_hops: int = 3) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        triples_str, structured_edges, visited = [], [], set()

        for seed in seeds:
            if seed not in self.graph:
                continue
            queue = [(seed, 0)]
            visited.add(seed)

            while queue:
                curr, depth = queue.pop(0)
                if depth >= max_hops:
                    continue

                for neighbor in self.graph.successors(curr):
                    rel = self.graph[curr][neighbor].get("relation", "connected")
                    triples_str.append(f"({curr}) -[{rel}]-> ({neighbor})")
                    structured_edges.append((curr, neighbor, rel))
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))

                for predecessor in self.graph.predecessors(curr):
                    rel = self.graph[predecessor][curr].get("relation", "connected")
                    triples_str.append(f"({predecessor}) -[{rel}]-> ({curr})")
                    structured_edges.append((predecessor, curr, rel))
                    if predecessor not in visited:
                        visited.add(predecessor)
                        queue.append((predecessor, depth + 1))

        return list(set(triples_str)), list(set(structured_edges))

    def retrieve_chunks(self, query: str, top_k: int = 2) -> List[str]:
        q_vec = self.get_embedding(query)
        scores = [(self.raw_chunks[i], self.cosine_sim(q_vec, emb)) for i, emb in enumerate(self.chunk_embeddings)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scores[:top_k]]

    def query_local(self, query: str) -> Dict[str, Any]:
        """Local Search: Focuses on specific entity neighborhoods for targeted multi-hop questions."""
        seeds = self.find_seed_entities(query, top_k=2)
        triples_str, structured_edges = self.traverse_subgraph(seeds, max_hops=3)
        chunks = self.retrieve_chunks(query, top_k=2)

        context = "\n".join(triples_str) if triples_str else "No graph paths found."
        prompt = (
            f"Knowledge Graph Traversals:\n{context}\n\n"
            f"Context Passages:\n" + "\n---\n".join(chunks) + f"\n\n"
            f"Question: {query}\n"
            "Synthesize a step-by-step reasoning answer using the traversed relationships:"
        )
        return {
            "mode": "local",
            "query": query,
            "answer": self._call_llm(prompt),
            "seed_nodes": seeds,
            "traversed_triples": triples_str,
            "edges": structured_edges,
            "retrieved_chunks": chunks
        }

    def query_global(self, query: str) -> Dict[str, Any]:
        """Global Search: Sensemaking query synthesizing across Louvain community clusters."""
        if not self.community_summaries:
            return self.query_local(query)

        community_context = "\n\n".join([f"Cluster #{c_id}:\n{summ}" for c_id, summ in self.community_summaries.items()])
        prompt = (
            f"Community Summaries of Knowledge Corpus:\n{community_context}\n\n"
            f"Holistic Question: {query}\n"
            "Synthesize an executive high-level answer integrating the themes above:"
        )
        return {
            "mode": "global",
            "query": query,
            "answer": self._call_llm(prompt),
            "communities_evaluated": len(self.community_summaries)
        }

    def query_naive_rag(self, query: str, top_k: int = 2) -> Dict[str, Any]:
        chunks = self.retrieve_chunks(query, top_k=top_k)
        prompt = (
            f"Context Passages:\n" + "\n---\n".join(chunks) + f"\n\n"
            f"Question: {query}\n"
            "Answer the question strictly from the provided passages:"
        )
        return {"query": query, "answer": self._call_llm(prompt), "retrieved_chunks": chunks}