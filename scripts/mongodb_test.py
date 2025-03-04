from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get MongoDB URI from environment variable
mongo_uri = os.getenv("MONGO_URI")

# Connect to MongoDB
client = MongoClient(mongo_uri)

# List databases to verify connection
print("Connected to MongoDB successfully!")
print("Available databases:")
for db_name in client.list_database_names():
    print(f"- {db_name}")