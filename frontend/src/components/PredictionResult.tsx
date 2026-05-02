import type { PredictionResponse, RiskLevel } from "../types/api";

interface Props {
  result: PredictionResponse;
}

const RISK_COPY: Record<RiskLevel, { label: string; cls: string; description: string }> = {
  low: {
    label: "Bajo",
    cls: "bg-emerald-50 text-emerald-800 border-emerald-200",
    description:
      "La probabilidad estimada se encuentra considerablemente por debajo del umbral del modelo.",
  },
  moderate: {
    label: "Moderado",
    cls: "bg-amber-50 text-amber-800 border-amber-200",
    description:
      "Probabilidad no despreciable. Se sugiere considerar el contexto clínico y monitorear.",
  },
  elevated: {
    label: "Elevado",
    cls: "bg-orange-50 text-orange-800 border-orange-200",
    description:
      "La probabilidad estimada supera el umbral del modelo. Se recomienda evaluación adicional.",
  },
  high: {
    label: "Alto",
    cls: "bg-red-50 text-red-800 border-red-200",
    description:
      "La probabilidad estimada se encuentra significativamente por encima del umbral. Se recomienda priorizar la evaluación.",
  },
};

const NON_INFORMATIVE_WARNINGS = new Set(["generated_by_backfill"]);

function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

export function PredictionResult({ result }: Props) {
  const r = RISK_COPY[result.risk_level];
  const userWarnings = result.warnings.filter((w) => !NON_INFORMATIVE_WARNINGS.has(w));

  return (
    <div className="overflow-hidden rounded-2xl border border-fvl-line bg-white shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-fvl-line bg-fvl-mint/30 px-6 py-4">
        <h3 className="text-base font-bold text-fvl-700">
          Resultado de la predicción
        </h3>
        <span
          className={`rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-wide ${r.cls}`}
        >
          Riesgo {r.label}
        </span>
      </div>

      <div className="space-y-5 px-6 py-6">
        <div className="grid gap-5 sm:grid-cols-3">
          <Metric label="Probabilidad" value={pct(result.probability)} prominent />
          <Metric label="Umbral del modelo" value={pct(result.threshold)} />
          <Metric
            label="Prevalencia (entrenamiento)"
            value={result.prevalence_train != null ? pct(result.prevalence_train) : "—"}
          />
        </div>

        <p className="text-sm leading-relaxed text-slate-700">{r.description}</p>

        <div className="flex flex-wrap items-center gap-2 text-xs">
          {result.calibrated ? (
            <span className="inline-flex items-center gap-1 rounded-full border border-fvl-mint-border bg-fvl-mint px-3 py-1 font-medium text-fvl-700">
              <span aria-hidden>✓</span> Probabilidad calibrada
            </span>
          ) : (
            <span className="rounded-full bg-amber-100 px-3 py-1 font-semibold text-amber-800">
              ⚠ Modelo no calibrado — interpretar la probabilidad con cautela.
            </span>
          )}
          <span className="text-slate-400">·</span>
          <span className="text-slate-600">
            Clase predicha:{" "}
            <code className="rounded bg-fvl-surface px-1.5 py-0.5 font-semibold text-fvl-700">
              {result.predicted_class}
            </code>
          </span>
        </div>

        {userWarnings.length > 0 && (
          <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-800">
            <strong className="mb-1 block">Advertencias del modelo:</strong>
            <ul className="list-disc space-y-0.5 pl-5">
              {userWarnings.map((w) => (
                <li key={w}>
                  <code>{w}</code>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

function Metric({
  label,
  value,
  prominent = false,
}: {
  label: string;
  value: string;
  prominent?: boolean;
}) {
  return (
    <div>
      <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
        {label}
      </p>
      <p
        className={[
          "mt-1 tabular-nums text-fvl-700",
          prominent ? "text-4xl font-bold" : "text-xl font-semibold",
        ].join(" ")}
      >
        {value}
      </p>
    </div>
  );
}
