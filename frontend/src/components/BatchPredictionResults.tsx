import { useMemo } from "react";
import { downloadResults } from "../lib/excel";
import type { ModelSchema, PredictionResponse, RiskLevel } from "../types/api";

interface Props {
  schema: ModelSchema;
  inputs: Record<string, number | null>[];
  predictions: PredictionResponse[];
}

const RISK_PILL: Record<RiskLevel, string> = {
  low: "bg-emerald-50 text-emerald-800 border-emerald-200",
  moderate: "bg-amber-50 text-amber-800 border-amber-200",
  elevated: "bg-orange-50 text-orange-800 border-orange-200",
  high: "bg-red-50 text-red-800 border-red-200",
};

const RISK_LABEL: Record<RiskLevel, string> = {
  low: "Bajo",
  moderate: "Moderado",
  elevated: "Elevado",
  high: "Alto",
};

function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

export function BatchPredictionResults({ schema, inputs, predictions }: Props) {
  const summary = useMemo(() => {
    const counts: Record<RiskLevel, number> = {
      low: 0,
      moderate: 0,
      elevated: 0,
      high: 0,
    };
    let positives = 0;
    for (const p of predictions) {
      counts[p.risk_level] += 1;
      if (p.predicted_class === 1) positives += 1;
    }
    return { counts, positives };
  }, [predictions]);

  return (
    <div className="overflow-hidden rounded-2xl border border-fvl-line bg-white shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-fvl-line bg-fvl-mint/30 px-6 py-4">
        <div>
          <h3 className="text-base font-bold text-fvl-700">
            Resultados de la predicción ({predictions.length})
          </h3>
          <p className="text-xs text-slate-600">
            Casos positivos (clase=1):{" "}
            <strong className="text-fvl-700">{summary.positives}</strong> /{" "}
            {predictions.length}
          </p>
        </div>
        <button
          type="button"
          onClick={() => downloadResults(schema, inputs, predictions)}
          className="rounded-lg border border-fvl-700 bg-white px-4 py-2 text-sm font-semibold text-fvl-700 shadow-sm transition hover:bg-fvl-mint/40 focus:outline-none focus:ring-2 focus:ring-fvl-lime focus:ring-offset-2"
        >
          Exportar a Excel
        </button>
      </div>

      <div className="grid grid-cols-2 gap-3 px-6 py-5 sm:grid-cols-4">
        {(Object.keys(summary.counts) as RiskLevel[]).map((level) => (
          <div
            key={level}
            className={`rounded-xl border px-4 py-3 text-center ${RISK_PILL[level]}`}
          >
            <div className="text-[10px] font-semibold uppercase tracking-wider opacity-80">
              {RISK_LABEL[level]}
            </div>
            <div className="mt-1 text-2xl font-bold tabular-nums">
              {summary.counts[level]}
            </div>
          </div>
        ))}
      </div>

      <div className="overflow-x-auto border-t border-fvl-line">
        <table className="w-full text-sm">
          <thead className="bg-fvl-surface text-xs uppercase tracking-wide text-fvl-700">
            <tr>
              <th className="px-4 py-2.5 text-left font-semibold">#</th>
              <th className="px-4 py-2.5 text-right font-semibold">Probabilidad</th>
              <th className="px-4 py-2.5 text-center font-semibold">Clase</th>
              <th className="px-4 py-2.5 text-center font-semibold">Riesgo</th>
              <th className="px-4 py-2.5 text-right font-semibold">Umbral</th>
              <th className="px-4 py-2.5 text-center font-semibold">Calibrado</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map((p, i) => (
              <tr key={i} className="border-t border-fvl-line">
                <td className="px-4 py-3 tabular-nums text-slate-400">{i + 1}</td>
                <td className="px-4 py-3 text-right font-semibold tabular-nums text-fvl-700">
                  {pct(p.probability)}
                </td>
                <td className="px-4 py-3 text-center tabular-nums">
                  {p.predicted_class}
                </td>
                <td className="px-4 py-3 text-center">
                  <span
                    className={`rounded-full border px-2.5 py-0.5 text-xs font-medium ${RISK_PILL[p.risk_level]}`}
                  >
                    {RISK_LABEL[p.risk_level]}
                  </span>
                </td>
                <td className="px-4 py-3 text-right tabular-nums text-slate-500">
                  {pct(p.threshold)}
                </td>
                <td className="px-4 py-3 text-center">
                  {p.calibrated ? (
                    <span className="text-fvl-700" aria-label="Calibrado">✓</span>
                  ) : (
                    <span className="text-amber-600" title="Modelo no calibrado">⚠</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
