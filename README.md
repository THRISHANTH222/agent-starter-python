# Rural Health AI

Rural Health AI is a multilingual voice AI assistant and triage prototype built with LiveKit Agents, Gladia STT, OpenAI, MOSS RAG, and Fish Audio TTS. It is designed to provide accessible health information and triage guidance for rural populations in English, Telugu, and Hindi, with automatic language detection and code-switching.

## Architecture

```
User Voice 
  → LiveKit RTC 
  → Gladia Multilingual STT (en, te, hi + code-switching) 
  → OpenAI LLM 
  → MOSS RAG (rural-health English knowledge retrieval) 
  → Multilingual TTS (Fish Audio s2.1-pro via LiveKit Inference) 
  → User Voice Response
```

## Environment Variables

The following environment variables are required to run the agent:

- `LIVEKIT_URL`: LiveKit Cloud WebSocket URL (e.g. `wss://your-project.livekit.cloud`)
- `LIVEKIT_API_KEY`: LiveKit Cloud API Key
- `LIVEKIT_API_SECRET`: LiveKit Cloud API Secret
- `OPENAI_API_KEY`: OpenAI API Key for LLM inference
- `GLADIA_API_KEY`: Gladia API Key for multilingual STT
- `MOSS_PROJECT_ID`: MOSS Project ID for RAG index
- `MOSS_PROJECT_KEY`: MOSS Project Key for RAG index

See `.env.example` for a template.

## Local Development

### 1. Install Dependencies
```bash
uv sync
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env.local` and fill in your credentials:
```bash
cp .env.example .env.local
```

### 3. Run the Agent Locally
To run the agent in local development mode:
```bash
uv run python src/agent.py dev
```

To run in terminal console mode:
```bash
uv run python src/agent.py console
```

To run unit tests:
```bash
$env:PYTHONPATH="src"; uv run pytest
```

## Render Deployment

Follow these steps to deploy the Python LiveKit voice agent as a service on Render:

1. **Push project to GitHub**: Ensure all code changes, `render.yaml`, `requirements.txt`, and `.env.example` are committed and pushed to your repository.
2. **Create a Render Service**: Log into the [Render Dashboard](https://dashboard.render.com/) and click **New +** -> **Blueprint** (or **Background Worker**).
3. **Connect GitHub Repository**: Select your `rural-health-agent` repository.
4. **Select Service Type**: Select **Background Worker** (or **Web Service** if you prefer an HTTP health endpoint; the code supports both).
5. **Set Build & Start Commands**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python src/agent.py start`
6. **Add Environment Variables**: Under the Environment section, add all 7 required environment variables (`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `OPENAI_API_KEY`, `GLADIA_API_KEY`, `MOSS_PROJECT_ID`, `MOSS_PROJECT_KEY`).
7. **Deploy**: Click **Deploy**.
8. **Check Logs**: Monitor the deployment logs in Render. Look for:
   - `Starting Rural Health AI agent...`
   - `Connecting to LiveKit...`
   - `Agent ready`
9. **Verify in LiveKit Cloud**: Open the LiveKit Cloud Dashboard -> Agents tab to confirm your worker process is registered and ready to receive dispatches.

## Security

- Never commit secrets or API keys to Git.
- Keep `.env` and `.env.local` in `.gitignore`.
- Set all production credentials securely in your Render dashboard environment settings.

## Safety

This system is an educational health information and triage prototype. It does **NOT** diagnose medical conditions, claim to be a doctor, prescribe medications, or provide dosage instructions. For serious or emergency symptoms (such as severe difficulty breathing, loss of consciousness, severe chest pain, seizure, severe confusion, severe dehydration, or sudden severe headache), users must seek urgent professional or emergency medical care immediately.
