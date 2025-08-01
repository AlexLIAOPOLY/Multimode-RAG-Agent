import { initApiPassthrough } from "langgraph-nextjs-api-passthrough";

// This file acts as a proxy for requests to your LangGraph server.
// Read the [Going to Production](https://github.com/langchain-ai/agent-chat-ui?tab=readme-ov-file#going-to-production) section for more information.

export const { GET, POST, PUT, PATCH, DELETE, OPTIONS, runtime } =
  initApiPassthrough({
    apiUrl: process.env.LANGGRAPH_API_URL ?? "your-langgraph-api-url-here", // Please set LANGGRAPH_API_URL in your environment
    apiKey: process.env.LANGSMITH_API_KEY ?? "your-langsmith-api-key-here", // Please set LANGSMITH_API_KEY in your environment
    runtime: "edge", // default
  });
