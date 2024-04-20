import os

# Set default values for local development
DEFAULT_NEO4J_URI = "neo4j://localhost:7687"
DEFAULT_NEO4J_USERNAME = "neo4j"
DEFAULT_NEO4J_PASSWORD = "priovar."

# Use os.getenv to get the environment variable, if it exists, or default to the local development value
uri = os.getenv('NEO4J_URI', DEFAULT_NEO4J_URI)
username = os.getenv('NEO4J_USERNAME', DEFAULT_NEO4J_USERNAME)
password = os.getenv('NEO4J_PASSWORD', DEFAULT_NEO4J_PASSWORD)
API_KEY = "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"