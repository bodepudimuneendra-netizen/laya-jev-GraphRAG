import os

# Force Kuzu as the graph database backend for this test
os.environ["GRAPH_DB_BACKEND"] = "kuzu"

from graphrag.graph.factory import get_graph_client
import logging

logging.basicConfig(level=logging.INFO)

print("🚀 Starting Kùzu Embedded Database Test...")

try:
    # 1. Initialize the client (this automatically spins up the embedded DB)
    client = get_graph_client()
    print("✅ KuzuClient initialized successfully.")
    
    # 2. Create schema
    client.create_schema()
    
    # 3. Add some nodes and an edge
    client.upsert_node(name="AgenticGraphRAG", label="Entity")
    client.upsert_node(name="Developer", label="Entity")
    client.upsert_edge(source="Developer", target="AgenticGraphRAG", rel_type="BUILDS")
    print("✅ Added test nodes and 'BUILDS' relationship.")
    
    # 4. Query the neighbors
    neighbors = client.get_neighbors("Developer")
    print(f"✅ Query results for 'Developer' neighbors: {neighbors}")
    
    print("\n🎉 SUCCESS! Kùzu is working perfectly offline (no Docker needed).")

except Exception as e:
    print(f"\n❌ FAILED: {e}")
