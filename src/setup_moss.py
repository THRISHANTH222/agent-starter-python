import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from moss import MossClient, DocumentInfo

# Load environment variables from .env.local
env_path = Path(".env.local")
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

async def main():
    print("Loading MOSS credentials...")
    project_id = os.environ.get("MOSS_PROJECT_ID")
    project_key = os.environ.get("MOSS_PROJECT_KEY")

    if not project_id or not project_key:
        raise ValueError("MOSS_PROJECT_ID or MOSS_PROJECT_KEY environment variables are missing.")

    client = MossClient(project_id, project_key)

    knowledge_dir = Path("src/medical_knowledge")
    if not knowledge_dir.exists():
        raise FileNotFoundError(f"Knowledge directory not found: {knowledge_dir}")

    md_files = sorted(list(knowledge_dir.glob("*.md")))
    print(f"Found {len(md_files)} knowledge documents.")

    docs = []
    doc_ids = []
    for filepath in md_files:
        doc_id = filepath.stem  # Filename without extension
        text_content = filepath.read_text(encoding="utf-8")
        docs.append(DocumentInfo(id=doc_id, text=text_content))
        doc_ids.append(doc_id)

    index_name = "rural-health"
    print(f"Creating/updating index: {index_name}")

    existing_indexes = await client.list_indexes()
    existing_names = [idx.name for idx in existing_indexes]
    if index_name in existing_names:
        await client.delete_index(index_name)

    await client.create_index(index_name, docs)

    for doc_id in doc_ids:
        print(f"Indexed: {doc_id}")

    print("MOSS index setup completed.")

if __name__ == "__main__":
    asyncio.run(main())
