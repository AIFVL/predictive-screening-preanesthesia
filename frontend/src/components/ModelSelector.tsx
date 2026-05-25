import type { ModelSummary } from "../types/api";

interface Props {
  models: ModelSummary[];
  value: string | null;
  onChange: (algorithm: string) => void;
}

const ALGORITHM_LABELS: Record<string, string> = {
  logistic_regression: "Regresión logística",
  random_forest: "Random Forest",
  extra_trees: "Extra Trees",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
  hist_gradient_boosting: "HistGradientBoosting",
  mlp: "Red neuronal (MLP)",
  stacking: "Stacking",
  voting: "Voting",
};

const CALIBRATION_LABELS: Record<string, string> = {
  isotonic: "Isotónica",
  sigmoid: "Platt",
};

function formatPct(v: number | null | undefined): string {
  if (v == null) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function formatBrier(v: number | null | undefined): string {
  if (v == null) return "—";
  return v.toFixed(3);
}

// Indicadores visuales de calidad relativa por métrica.
function rocClass(v: number | null | undefined): string {
  if (v == null) return "text-slate-500";
  if (v >= 0.8) return "text-emerald-700 font-semibold";
  if (v >= 0.7) return "text-emerald-600";
  return "text-slate-700";
}
function f2Class(v: number | null | undefined): string {
  if (v == null) return "text-slate-500";
  if (v >= 0.65) return "text-emerald-700 font-semibold";
  if (v >= 0.55) return "text-emerald-600";
  return "text-slate-700";
}
function brierClass(v: number | null | undefined): string {
  if (v == null) return "text-slate-500";
  if (v <= 0.12) return "text-emerald-700 font-semibold"; // mejor calibración
  if (v <= 0.18) return "text-slate-700";
  return "text-amber-700";
}

export function ModelSelector({ models, value, onChange }: Props) {
  if (models.length === 0) {
    return (
      <p className="text-sm text-slate-500">
        Seleccione primero un tipo de riesgo para visualizar los modelos disponibles.
      </p>
    );
  }

  // Ordenar: recomendados primero, luego por ROC-AUC desc.
  const sorted = [...models].sort((a, b) => {
    if (a.recommended !== b.recommended) return a.recommended ? -1 : 1;
    return (b.performance.roc_auc ?? 0) - (a.performance.roc_auc ?? 0);
  });

  return (
    <div className="overflow-hidden rounded-2xl border border-fvl-line bg-white shadow-sm">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-fvl-mint/40 text-xs uppercase tracking-wide text-fvl-700">
            <tr>
              <th className="px-5 py-3 text-left font-semibold">Algoritmo</th>
              <th className="px-4 py-3 text-right font-semibold">ROC-AUC</th>
              <th className="px-4 py-3 text-right font-semibold">F2</th>
              <th className="px-4 py-3 text-right font-semibold">Brier</th>
              <th className="px-4 py-3 text-center font-semibold">Calibración</th>
              <th className="px-4 py-3 text-center font-semibold">Selección</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((m) => {
              const selected = m.algorithm === value;
              return (
                <tr
                  key={m.model_id}
                  onClick={() => onChange(m.algorithm)}
                  className={[
                    "cursor-pointer border-t border-fvl-line transition",
                    selected
                      ? "bg-fvl-mint"
                      : "hover:bg-fvl-mint/30",
                  ].join(" ")}
                >
                  <td className="relative px-5 py-4">
                    {selected && (
                      <span
                        aria-hidden
                        className="absolute left-0 top-0 h-full w-1 bg-fvl-lime"
                      />
                    )}
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-fvl-700">
                        {ALGORITHM_LABELS[m.algorithm] ?? m.algorithm}
                      </span>
                      {m.recommended && (
                        <span className="rounded-full bg-fvl-lime px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-fvl-700">
                          Recomendado
                        </span>
                      )}
                    </div>
                  </td>
                  <td className={`px-4 py-4 text-right tabular-nums ${rocClass(m.performance.roc_auc)}`}>
                    {formatPct(m.performance.roc_auc)}
                  </td>
                  <td className={`px-4 py-4 text-right tabular-nums ${f2Class(m.performance.f2)}`}>
                    {formatPct(m.performance.f2)}
                  </td>
                  <td className={`px-4 py-4 text-right tabular-nums ${brierClass(m.performance.brier)}`}>
                    {formatBrier(m.performance.brier)}
                  </td>
                  <td className="px-4 py-4 text-center">
                    {m.calibrated ? (
                      <span className="inline-flex items-center gap-1 rounded-full border border-fvl-mint-border bg-fvl-mint px-2.5 py-0.5 text-xs font-medium text-fvl-700">
                        <span aria-hidden>✓</span>
                        {CALIBRATION_LABELS[m.calibration_method ?? ""] ?? m.calibration_method ?? "Sí"}
                      </span>
                    ) : (
                      <span className="rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-medium text-amber-800">
                        No calibrado
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-4 text-center">
                    <span
                      className={[
                        "inline-block h-4 w-4 rounded-full border-2 transition",
                        selected
                          ? "border-fvl-700 bg-fvl-lime"
                          : "border-fvl-line bg-white",
                      ].join(" ")}
                    />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="border-t border-fvl-line bg-fvl-surface px-5 py-2 text-xs text-slate-500">
        Los valores resaltados en verde indican mejor desempeño relativo en cada
        métrica. Brier menor implica mejor calibración.
      </div>
    </div>
  );
}
