# Sales Assistant: a RAG chatbot (LangChain + OpenAI)

A small full-stack app where you chat with an AI assistant that answers questions from your own documents instead of from general knowledge. It loads a PDF and a YouTube video transcript, indexes them as embeddings, and uses the most relevant passages as context for each answer.

## How it works

1. On startup the backend loads a PDF (`back/xbox.pdf`) and the transcript of a YouTube video, and splits them into chunks (LangChain text splitters).
2. The chunks are embedded with OpenAI embeddings and stored in an in-memory vector store.
3. For each question sent to `POST /message`, the server finds the 2 most similar chunks, adds them as context to the prompt, and asks GPT-4 to answer using that context. If the context is not enough, the model asks for more.

## Tech stack

- Backend: Node.js, Express, LangChain, OpenAI API
- Frontend: React + Vite
- Vector search: LangChain MemoryVectorStore (in-memory)

## Run it locally

Backend:

```
cd back
npm install
cp .env.example .env   # then set OPENAI_API_KEY in .env
npm run dev            # http://localhost:3000
```

Frontend:

```
cd front
npm install
npm run dev
```

API example:

```
curl -X POST http://localhost:3000/message -H "Content-Type: application/json" -d '{"message": "What is this document about?"}'
```

## Limitations and next steps

- The vector store is in memory, so documents are re-indexed on every restart. A persistent store (pgvector, Qdrant, Pinecone) would be the next step.
- The conversation history is not kept between requests yet.
- The document sources are hard-coded; an upload endpoint would make them configurable.
- No automated tests yet.
