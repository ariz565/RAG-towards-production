import { API_CONFIG } from "./config";

export const BASE_URL = API_CONFIG.baseUrl;

async function fetchApi<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const headers = {
    ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
    ...options.headers,
  };

  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${endpoint}`, {
      credentials: "include", // Send cookies with request
      ...options,
      headers,
    });
  } catch (error) {
    throw new Error(`Network error: Could not reach ${BASE_URL}${endpoint}`);
  }

  if (!response.ok) {
    let errorMessage = "API request failed";
    try {
      const errorData = await response.json();
      errorMessage = errorData.detail || errorMessage;
    } catch {
      // Ignore JSON parse error if response is not JSON
    }
    throw new Error(errorMessage);
  }

  return response.json();
}

// -- 1. Authentication --
export async function login(data: any) {
  return fetchApi<any>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function signup(data: any) {
  return fetchApi<any>("/api/auth/signup", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function getCurrentUser() {
  return fetchApi<any>("/api/auth/me");
}

export async function logout() {
  return fetchApi<any>("/api/auth/logout", {
    method: "POST",
  });
}

// -- 2. Documents --
export async function getDocuments() {
  return fetchApi<any>("/api/documents");
}

export async function deleteAllData() {
  return fetchApi<any>("/api/documents", {
    method: "DELETE",
  });
}

export async function uploadDocument(formData: FormData) {
  return fetchApi<any>("/api/upload", {
    method: "POST",
    body: formData, // fetch will automatically set the correct boundary for multipart/form-data
  });
}

export async function indexDocument(docId: string, strategy: string = "hybrid") {
  const path = strategy === "hybrid" ? "/api/index/hybrid" : (strategy === "all" ? "/api/index/all" : "/api/index");
  return fetchApi<any>(path, {
    method: "POST",
    body: JSON.stringify({ filename: docId }),
  });
}

// -- 3. Ask (Core Q&A) --
export async function askQuestion(data: any) {
  return fetchApi<any>("/api/ask", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function resumeAskQuestion(data: any) {
  return fetchApi<any>("/api/ask/resume", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

// -- 4. Summarize --
export async function summarizeDocument(data: any) {
  return fetchApi<any>("/api/summarize", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function summarizeAsync(data: any) {
  return fetchApi<any>("/api/summarize/async", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function getJob(jobId: string) {
  return fetchApi<any>(`/api/jobs/${jobId}`);
}

// -- 5. Configuration --
export async function getConfig() {
  return fetchApi<any>("/api/config");
}

// -- 6. Inspector --
export async function getHealth() {
  return fetchApi<any>("/api/health");
}

export async function getTree(docId: string) {
  return fetchApi<any>(`/api/tree?doc_id=${docId}`);
}

export async function getPageText(docId: string, pageNum: number) {
  return fetchApi<any>(`/api/page/${pageNum}?doc_id=${docId}`);
}

export async function updateConfig(data: any) {
  return fetchApi<any>("/api/config", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

// -- 7. Second Brain (standalone module's graph, read-only) --
export async function getSecondBrainGraph() {
  return fetchApi<any>("/api/second-brain/graph");
}

// -- Stream helper --
export async function streamAskQuestion(data: any, onEvent: (event: string, data: any) => void) {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };

  const response = await fetch(`${BASE_URL}/api/ask/stream`, {
    method: "POST",
    credentials: "include",
    headers,
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    throw new Error("Stream request failed");
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error("No reader");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    
    // Naive SSE parser
    const lines = buffer.split("\n\n");
    buffer = lines.pop() || ""; // Keep the last incomplete part

    for (const line of lines) {
      if (!line.trim()) continue;
      
      let eventType = "message";
      let eventData = "";

      const parts = line.split("\n");
      for (const part of parts) {
        if (part.startsWith("event:")) {
          eventType = part.slice(6).trim();
        } else if (part.startsWith("data:")) {
          eventData = part.slice(5).trim();
        }
      }

      if (eventData) {
        try {
          onEvent(eventType, JSON.parse(eventData));
        } catch {
          onEvent(eventType, eventData);
        }
      }
    }
  }
}
