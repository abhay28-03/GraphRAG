import os
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from pyvis.network import Network
from core.engine import GraphRAGEngine

API_KEY = os.environ.get("GROQ_API_KEY")
engine: Optional[GraphRAGEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = GraphRAGEngine(api_key=API_KEY)
    
    seed_data = [
        "Dr. Evelyn Reed oversees Project Hyperion at the Department of Applied Physics.",
        "Project Hyperion has been deployed at the Atacama Observational Facility in Chile.",
        "The Atacama Observational Facility relies on the Helium-3 Cryogenic Cooling Unit for its sensors.",
        "AlphaCorp acquired security firm CypherTech in late 2024.",
        "CypherTech is the sole creator of the ZeroTrust-Mesh security protocol.",
        "The ZeroTrust-Mesh protocol requires hardware chips manufactured by Valeron Micro.",
        "Valeron Micro operates its primary semiconductor foundry in Dresden, Germany."
    ]
    print("Bootstrap: Ingesting enterprise corpus & resolving entities...")
    engine.ingest_documents(seed_data)
    print(f"Graph loaded: {engine.graph.number_of_nodes()} nodes, {engine.graph.number_of_edges()} edges, {len(engine.community_summaries)} communities.")
    yield
    if engine:
        engine.close()

app = FastAPI(
    title="GraphRAG Production Microservice",
    description="Dual-Mode Knowledge Graph Augmented Generation API with Neo4j and Interactive Visualizer",
    version="2.0.0",
    lifespan=lifespan
)

class IngestRequest(BaseModel):
    documents: List[str]

class QueryRequest(BaseModel):
    query: str

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/graph")

@app.post("/api/v1/ingest")
def ingest(payload: IngestRequest):
    if not payload.documents:
        raise HTTPException(status_code=400, detail="Empty document payload")
    engine.ingest_documents(payload.documents)
    return {
        "status": "success",
        "total_nodes": engine.graph.number_of_nodes(),
        "total_edges": engine.graph.number_of_edges(),
        "communities": len(engine.community_summaries)
    }

@app.post("/api/v1/query/local")
def query_local(payload: QueryRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    return engine.query_local(payload.query)

@app.post("/api/v1/query/global")
def query_global(payload: QueryRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    return engine.query_global(payload.query)

@app.get("/graph", response_class=HTMLResponse)
def render_graph():
    net = Network(height="750px", width="100%", bgcolor="#0f172a", font_color="white", directed=True)
    net.force_atlas_2based()

    for node, attrs in engine.graph.nodes(data=True):
        category = attrs.get("category", "Entity")
        net.add_node(node, label=node, title=f"Type: {category}", color="#38bdf8", size=25)

    for u, v, data in engine.graph.edges(data=True):
        rel = data.get("relation", "connected")
        net.add_edge(u, v, title=rel, label=rel, color="#94a3b8")

    return HTMLResponse(content=net.generate_html())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)