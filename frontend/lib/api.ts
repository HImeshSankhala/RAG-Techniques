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
  /** Pauses mid-run for a person (Interactive RAG): runnable in the playground, not comparable. */
  needs_human: boolean;
  /** Can never run here, however much is built — REALM is a pre-training method. */
  docs_only: boolean;
  /** Editorial range of model calls per query, e.g. "2-5". Prose, not a measurement. */
  llm_calls_range: string;
  /** Editorial range of retrieval passes per query. Same caveat. */
  retrieval_passes_range: string;
  /**
   * Why this technique cannot run against uploaded documents, or "" when it can.
   * The same string the API's 409 uses, so the disabled option and the refusal
   * cannot say different things.
   */
  upload_note: string;
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
  /**
   * Stored votes counted against this run's candidates — Feedback RAG only, 0 for
   * every other technique. Matched rows, NOT rows that moved something: the cap
   * may have discarded some, and a vote can point where the retriever already
   * did. Non-zero means the result depends on accumulated history, which is what
   * the compare row has to say out loud.
   */
  feedback_votes: number;
}

/** Mirrors `api.schemas.RunResponse`. */
export interface RunResponse {
  technique: string;
  query: string;
  answer: string;
  retrieved_chunks: Chunk[];
  steps: Step[];
  metadata: Metadata;
  /** Set when the run paused for human review; pass it to `finalizeDraft`. */
  draft_id: string | null;
  /** Which corpus answered: "demo corpus", or an uploaded corpus's label. */
  corpus: string;
}

/** Mirrors `api.schemas.ComparisonSide`. */
export interface ComparisonSide {
  technique: string;
  model?: string | null;
}

/** Mirrors `api.schemas.ComparisonDiff`. Computed server-side so every client agrees. */
export interface ComparisonDiff {
  chunk_overlap: number;
  chunk_overlap_pct: number;
  shared_chunk_ids: string[];
  only_a_chunk_ids: string[];
  only_b_chunk_ids: string[];
  latency_delta_ms: number;
  llm_calls_delta: number;
  steps_delta: number;
  tokens_in_delta: number;
  tokens_out_delta: number;
  cost_delta_usd: number;
  same_technique: boolean;
  same_model: boolean;
}

/** Mirrors `api.schemas.CompareResponse`. */
export interface CompareResponse {
  query: string;
  a: RunResponse;
  b: RunResponse;
  diff: ComparisonDiff;
}

/** Mirrors `api.schemas.UploadResponse`. */
export interface UploadedCorpus {
  session_id: string;
  label: string;
  documents: number;
  chunks: number;
  expires_in_seconds: number;
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
  sessionId?: string | null,
): Promise<RunResponse> {
  return apiFetch<RunResponse>("/api/run", {
    method: "POST",
    body: JSON.stringify({
      technique,
      query,
      model: model ?? null,
      // Omitted means the demo corpus. There is no third state: the API 404s an
      // id it does not know rather than falling back.
      session_id: sessionId ?? null,
    }),
  });
}

/**
 * POST /api/compare — two (technique × model) sides against one query.
 *
 * Both axes vary independently: same model + different techniques asks "does
 * retrieval differ?", same technique + different models asks "does the model
 * differ?". The diff row is computed server-side.
 */
export function compareTechniques(
  query: string,
  a: ComparisonSide,
  b: ComparisonSide,
  sessionId?: string | null,
): Promise<CompareResponse> {
  return apiFetch<CompareResponse>("/api/compare", {
    method: "POST",
    // One session id for the whole comparison, not one per side — both halves
    // read the same corpus or the comparison is about corpora, not techniques.
    body: JSON.stringify({ query, a, b, session_id: sessionId ?? null }),
  });
}

/**
 * POST /api/run/final — the human's half of an Interactive RAG run.
 *
 * `chunkIds` are the draft passages to keep; the rest are dropped. `hint` steers
 * one extra retrieval and is not shown to the model. The draft's model is reused.
 */
export function finalizeDraft(
  draftId: string,
  chunkIds: string[],
  hint: string,
): Promise<RunResponse> {
  return apiFetch<RunResponse>("/api/run/final", {
    method: "POST",
    body: JSON.stringify({ draft_id: draftId, chunk_ids: chunkIds, hint }),
  });
}

/**
 * POST /api/feedback — one thumbs up or down on the passages a run showed.
 *
 * Only `feedback-rag` accepts votes (409 otherwise). Every call is a vote: there
 * is no dedupe, so clicking twice counts twice, by design.
 */
export function submitFeedback(
  technique: string,
  query: string,
  chunkIds: string[],
  rating: 1 | -1,
): Promise<{ ok: true }> {
  return apiFetch<{ ok: true }>("/api/feedback", {
    method: "POST",
    body: JSON.stringify({ technique, query, chunk_ids: chunkIds, rating }),
  });
}

/**
 * POST /api/documents — index uploaded files into a corpus of their own.
 *
 * Multipart, so this is the one call that does not go through `apiFetch`: setting
 * a JSON content-type on a FormData body stops the browser from adding the
 * multipart boundary, and the request arrives unparseable.
 */
export async function uploadDocuments(files: File[]): Promise<UploadedCorpus> {
  const form = new FormData();
  for (const file of files) form.append("files", file);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/documents`, { method: "POST", body: form });
  } catch {
    throw new ApiError(`Could not reach the API at ${API_BASE_URL}. Is the backend running?`, 0);
  }

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      /* keep the status-line fallback */
    }
    throw new ApiError(detail, response.status);
  }

  return (await response.json()) as UploadedCorpus;
}

/** DELETE /api/documents/:id — drop an uploaded corpus now rather than at its TTL. */
export async function deleteDocuments(sessionId: string): Promise<void> {
  try {
    await fetch(`${API_BASE_URL}/api/documents/${sessionId}`, { method: "DELETE" });
  } catch {
    // Best-effort. The corpus is refused once its TTL passes and dropped by the
    // next sweep either way, so a failed reset costs disk until then — it must
    // not strand the reader on a corpus the UI has already stopped using.
  }
}
