import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from moss import MossClient, QueryOptions

# Load environment variables from .env.local
env_path = Path(".env.local")
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


async def main():
    project_id = os.environ.get("MOSS_PROJECT_ID")
    project_key = os.environ.get("MOSS_PROJECT_KEY")

    if not project_id or not project_key:
        raise ValueError(
            "MOSS_PROJECT_ID or MOSS_PROJECT_KEY environment variables are missing."
        )

    client = MossClient(project_id, project_key)

    index_name = "rural-health"
    await client.load_index(index_name)

    query_text = "I have fever and difficulty breathing"
    options = QueryOptions(top_k=4)

    results = await client.query(index_name, query_text, options=options)

    print(f"Query: {query_text}\n")
    print("Results:")
    for i, doc in enumerate(results.docs, 1):
        print(f"{i}. Document ID: {doc.id}")
        print(f"   Score: {doc.score}")
        print(f"   Text: {doc.text[:200]}...\n")

    if hasattr(results, "time_taken_ms") and results.time_taken_ms is not None:
        print(f"Retrieval Time: {results.time_taken_ms} ms")


if __name__ == "__main__":
    asyncio.run(main())
