/**
 * Typed client for the RAG Lab API.
 *
 * These types mirror `backend/api/schemas.py` by hand — that file is the source of
 * truth. If a Pydantic model changes there, change it here in the same commit.
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** Mirrors `api.schemas.Technique`. */
export interface Technique {
  name: string;
  display_name: string;
  tagline: string;
  implemented: boolean;
}

/** Thrown when the API is unreachable or answers with a non-2xx status. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      // The catalog is small and changes as techniques land, so never serve a stale
      // copy in dev. Revisit if this page is ever statically generated.
      cache: "no-store",
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch (cause) {
    throw new ApiError(
      `Could not reach the API at ${API_BASE_URL}. Is the backend running?`,
    );
  }

  if (!response.ok) {
    throw new ApiError(
      `GET ${path} failed with ${response.status} ${response.statusText}`,
      response.status,
    );
  }

  return (await response.json()) as T;
}

/** GET /api/techniques — every technique, with `implemented` telling you what can run. */
export function getTechniques(): Promise<Technique[]> {
  return apiFetch<Technique[]>("/api/techniques");
}

export { API_BASE_URL };
