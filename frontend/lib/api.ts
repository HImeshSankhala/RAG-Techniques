/**
 * Typed client for the RAG Lab API.
 *
 * These types mirror `backend/api/schemas.py` by hand — that file is the source of
 * truth. If a Pydantic model changes there, change it here in the same commit.
 */

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** Mirrors `api.schemas.Technique`. */
export interface Technique {
  name: string;
  display_name: string;
  tagline: string;
  implemented: boolean;
}

/** Mirrors `api.schemas.ModelInfo`. */
export interface ModelInfo {
  id: string;
  display_name: string;
  backend: string;
  is_paid: boolean;
  is_default: boolean;
  available: boolean;
  note: string;
}

/** Mirrors `api.schemas.Chunk`. */
export interface Chunk {
  text: string;
  source: string;
  score: number;
  chunk_id: string;
}

/** Mirrors `api.schemas.Step`. */
export interface Step {
  name: string;
  detail: string;
  duration_ms: number;
}

/** Mirrors `api.schemas.Metadata`. The compare view (Phase 5) diffs runs on these. */
export interface Metadata {
  model: string;
  backend: string;
  latency_ms: number;
  llm_calls: number;
  retrieval_passes: number;
  tokens_in: number;
  tokens_out: number;
  termination_reason: string;
  groundedness: number;
  cost_estimate_usd: number;
}

/** Mirrors `api.schemas.RunResponse`. */
export interface RunResponse {
  technique: string;
  query: string;
  answer: string;
  retrieved_chunks: Chunk[];
  steps: Step[];
  metadata: Metadata;
}

/** Mirrors `api.schemas.UsageResponse`. */
export interface Usage {
  calls: number;
  input_tokens: number;
  output_tokens: number;
  spend_estimate_usd: number;
  session_calls: number;
  session_call_limit: number;
  note: string;
}

/**
 * An API error carrying the status, so callers can distinguish a setup problem
 * (503, no API key) from a budget stop (429) from a bad request (400). The
 * backend puts an actionable sentence in `detail`; we surface it verbatim rather
 * than inventing our own wording.
 */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      cache: "no-store",
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiError(
      `Could not reach the API at ${API_BASE_URL}. Is the backend running?`,
      0,
    );
  }

  if (!response.ok) {
    // FastAPI puts the message in `detail`; fall back to the status line if the
    // body is not the shape we expect (a proxy error page, say).
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      /* keep the status-line fallback */
    }
    throw new ApiError(detail, response.status);
  }

  return (await response.json()) as T;
}

/** GET /api/techniques — every technique, with `implemented` telling you what can run. */
export function getTechniques(): Promise<Technique[]> {
  return apiFetch<Technique[]>("/api/techniques");
}

/** GET /api/models — selectable models, local and hosted. */
export function getModels(): Promise<ModelInfo[]> {
  return apiFetch<ModelInfo[]>("/api/models");
}

/** GET /api/usage — running estimate of paid spend. */
export function getUsage(): Promise<Usage> {
  return apiFetch<Usage>("/api/usage");
}

/**
 * POST /api/run — execute one technique against one query.
 *
 * Omit `model` to use the configured default (the local one), which is what keeps
 * casual exploration free.
 */
export function runTechnique(
  technique: string,
  query: string,
  model?: string,
): Promise<RunResponse> {
  return apiFetch<RunResponse>("/api/run", {
    method: "POST",
    body: JSON.stringify({ technique, query, model: model ?? null }),
  });
}
