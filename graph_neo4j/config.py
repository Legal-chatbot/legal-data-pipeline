import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "@Ntth06112004")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
BATCH_SIZE = 2000
EMBEDDING_DIM = 768
