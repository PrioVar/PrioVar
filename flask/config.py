import os

# Set default values for local development
DEFAULT_NEO4J_URI = "neo4j://localhost:7687"
DEFAULT_NEO4J_USERNAME = "neo4j"
DEFAULT_NEO4J_PASSWORD = "priovar."

# Use os.getenv to get the environment variable, if it exists, or default to the local development value
uri = os.getenv('NEO4J_URI', DEFAULT_NEO4J_URI)
username = os.getenv('NEO4J_USERNAME', DEFAULT_NEO4J_USERNAME)
password = os.getenv('NEO4J_PASSWORD', DEFAULT_NEO4J_PASSWORD)
API_KEY = "sk-proj-emfzR7GWBMAQY0K6ZFhDT3BlbkFJDeRRSBV83ULYpyUQwxzf"

api_username = "priovar@cs492.com"
api_password = "7G!^fEUwg^v2>b."
api_auth_token = "20ab68f6e357ed6475f5022c0977dbd5eafe0cb2"