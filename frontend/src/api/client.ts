import type {
  ApiEnvelope,
  ExplainResponse,
  ModelSchema,
  ModelSummary,
  PredictionResponse,
  TargetInfo,
} from "../types/api";

const API_BASE = (import.meta.env.VITE_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export class ApiClientError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code: string | null = null,
    readonly field: string | null = null,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  let body: ApiEnvelope<T> | null = null;
  try {
    body = (await res.json()) as ApiEnvelope<T>;
  } catch {
    throw new ApiClientError(`Respuesta no JSON (status ${res.status})`, res.status);
  }
  if (!res.ok || !body || body.success === false) {
    const first = body?.errors?.[0];
    throw new ApiClientError(
      first?.message ?? `Error HTTP ${res.status}`,
      res.status,
      first?.code ?? null,
      first?.field ?? null,
    );
  }
  if (body.data === null) {
    throw new ApiClientError("Respuesta vacía", res.status);
  }
  return body.data;
}

export type PatientData = Record<string, string | number | null>;

export const api = {
  listTargets: () => request<TargetInfo[]>("/targets"),
  listModels: () => request<ModelSummary[]>("/models"),
  getSchema: (target: string, algorithm: string) =>
    request<ModelSchema>(`/models/${target}/${algorithm}/schema`),
  predict: (target: string, algorithm: string, patient: PatientData) =>
    request<PredictionResponse>(
      `/models/${target}/${algorithm}/predict`,
      { method: "POST", body: JSON.stringify({ patient }) },
    ),
  explain: (
    target: string,
    algorithm: string,
    patient: PatientData,
    topN = 10,
  ) =>
    request<ExplainResponse>(
      `/models/${target}/${algorithm}/explain`,
      { method: "POST", body: JSON.stringify({ patient, top_n: topN }) },
    ),
};

export const apiBaseUrl = API_BASE;
