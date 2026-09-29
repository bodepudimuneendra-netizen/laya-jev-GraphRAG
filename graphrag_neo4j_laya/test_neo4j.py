import os
from graphrag.graph.factory import get_graph_client
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)

print("🚀 Starting Neo4j Database Test...")

try:
    # 1. Initialize the client (connects to Docker container)
    client = get_graph_client()
    print("✅ Neo4jClient initialized successfully.")
    
    # 2. Create schema (tests write access and constraints)
    client.create_schema()
    
    # 3. Add some nodes and an edge (tests Bolt driver)
    client.upsert_node(name="AgenticGraphRAG", label="Entity")
    client.upsert_node(name="Developer", label="Entity")
    client.upsert_edge(source="Developer", target="AgenticGraphRAG", rel_type="BUILDS")
    print("✅ Added test nodes and 'BUILDS' relationship.")
    
    # 4. Query the neighbors (tests read access)
    neighbors = client.get_neighbors("Developer")
    print(f"✅ Query results for 'Developer' neighbors: {neighbors}")
    
    # 5. Clean up test data
    client.delete_edge(source="Developer", target="AgenticGraphRAG", rel_type="BUILDS")
    print("✅ Cleaned up test data.")

    print("\n🎉 SUCCESS! Neo4j via Docker is working perfectly.")

except Exception as e:
    print(f"\n❌ FAILED: {e}")
