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

/**
 * GET /api/techniques — every technique, with `implemented` telling you what can run.
 *
 * Throws an Error the UI can show verbatim. `cache: "no-store"` is load-bearing: the
 * default ("auto no cache") lets Next statically prerender this route and bake the
 * catalog in at build time, which would pin the page to a stale list the moment a
 * technique becomes runnable. `no-store` forces the route to stay dynamic.
 */
export async function getTechniques(): Promise<Technique[]> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}/api/techniques`, { cache: "no-store" });
  } catch {
    throw new Error(`Could not reach the API at ${API_BASE_URL}. Is the backend running?`);
  }

  if (!response.ok) {
    throw new Error(
      `GET /api/techniques failed with ${response.status} ${response.statusText}`,
    );
  }

  return (await response.json()) as Technique[];
}
