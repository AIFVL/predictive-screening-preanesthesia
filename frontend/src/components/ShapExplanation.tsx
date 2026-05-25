import { useEffect, useState } from "react";
import { ApiClientError, api } from "../api/client";
import type { ExplainResponse, ShapContribution } from "../types/api";

interface Props {
  target: string;
  algorithm: string;
  features: Record<string, number | null>;
}

/** Formatea el nombre de la variable para que sea más legible */
function formatFeatureName(name: string): string {
  return name
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/** Formatea el valor del feature para mostrarlo brevemente */
function formatValue(value: number | null): string {
  if (value === null || value === undefined) return "—";
  // Si es binario (0 o 1) mostrarlo como Sí/No
  if (value === 0) return "No";
  if (value === 1) return "Sí";
  // Números con decimales
  if (Number.isInteger(value)) return String(value);
  return value.toFixed(3);
}

interface BarProps {
  contribution: ShapContribution;
  maxAbs: number;
}

function ContributionBar({ contribution, maxAbs }: BarProps) {
  const { feature, value, shap_value } = contribution;
  const isPositive = shap_value >= 0;
  const pct = maxAbs > 0 ? Math.abs(shap_value) / maxAbs : 0;
  const barWidth = `${(pct * 100).toFixed(1)}%`;

  return (
    <div className="grid items-center gap-x-3 gap-y-0.5"
      style={{ gridTemplateColumns: "1fr 52px" }}>
      {/* Nombre de la variable */}
      <div className="min-w-0">
        <p className="truncate text-xs font-medium text-slate-700" title={feature}>
          {formatFeatureName(feature)}
        </p>
        <p className="text-[10px] text-slate-400">valor: {formatValue(value)}</p>
      </div>

      {/* Valor SHAP numérico */}
      <p className={`text-right text-xs font-semibold tabular-nums ${isPositive ? "text-rose-600" : "text-emerald-600"}`}>
        {isPositive ? "+" : ""}{shap_value.toFixed(3)}
      </p>

      {/* Barra de impacto — ocupa las 2 columnas */}
      <div className="col-span-2 h-2 overflow-hidden rounded-full bg-slate-100">
        <div
          className={`h-full rounded-full transition-all duration-500 ${isPositive ? "bg-rose-400" : "bg-emerald-400"}`}
          style={{ width: barWidth }}
        />
      </div>
    </div>
  );
}

export function ShapExplanation({ target, algorithm, features }: Props) {
  const [data, setData] = useState<ExplainResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const [topN, setTopN] = useState(10);

  // Disparar la explicación cuando el usuario abre el panel
  useEffect(() => {
    if (!open) return;
    setLoading(true);
    setError(null);
    api
      .explain(target, algorithm, features, topN)
      .then((res) => setData(res))
      .catch((e: ApiClientError) => setError(e.message))
      .finally(() => setLoading(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, topN, target, algorithm]);

  const maxAbs =
    data
      ? Math.max(...data.contributions.map((c) => Math.abs(c.shap_value)), 1e-9)
      : 0;

  const positives = data?.contributions.filter((c) => c.shap_value > 0) ?? [];
  const negatives = data?.contributions.filter((c) => c.shap_value <= 0) ?? [];

  return (
    <div className="overflow-hidden rounded-2xl border border-fvl-line bg-white shadow-sm">
      {/* Cabecera plegable */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-3 border-b border-fvl-line bg-fvl-mint/30 px-6 py-4 text-left transition hover:bg-fvl-mint/50 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-fvl-lime"
        aria-expanded={open}
      >
        <div className="flex items-center gap-2">
          <span className="text-lg">🔍</span>
          <div>
            <h3 className="text-base font-bold text-fvl-700">
              ¿Por qué esta predicción?
            </h3>
            <p className="text-xs text-slate-500">
              Variables con mayor impacto en la decisión del modelo (SHAP)
            </p>
          </div>
        </div>
        <span
          className={`text-xl text-fvl-700 transition-transform duration-200 ${open ? "rotate-180" : ""}`}
          aria-hidden
        >
          ▾
        </span>
      </button>

      {/* Contenido */}
      {open && (
        <div className="space-y-5 px-6 py-6">
          {/* Selector de top-N */}
          <div className="flex flex-wrap items-center gap-2 text-xs text-slate-600">
            <span className="font-semibold">Mostrar top:</span>
            {[5, 10, 15, 20].map((n) => (
              <button
                key={n}
                type="button"
                onClick={() => setTopN(n)}
                className={`rounded-full border px-2.5 py-0.5 font-semibold transition focus:outline-none focus:ring-1 focus:ring-fvl-lime ${
                  topN === n
                    ? "border-fvl-700 bg-fvl-700 text-white"
                    : "border-fvl-line text-slate-600 hover:border-fvl-700 hover:text-fvl-700"
                }`}
              >
                {n}
              </button>
            ))}
            <span className="text-slate-400">variables</span>
          </div>

          {/* Estado de carga / error */}
          {loading && (
            <div className="flex items-center gap-2 py-8 text-sm text-slate-500">
              <span className="animate-spin text-base">⏳</span>
              Calculando contribuciones SHAP…
            </div>
          )}
          {error && (
            <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-800">
              <strong>No fue posible calcular SHAP:</strong> {error}
            </div>
          )}

          {/* Leyenda */}
          {data && !loading && (
            <>
              <div className="flex flex-wrap gap-4 rounded-lg bg-fvl-surface px-4 py-2.5 text-xs">
                <span className="flex items-center gap-1.5 font-medium text-rose-600">
                  <span className="inline-block h-2.5 w-2.5 rounded-full bg-rose-400" />
                  Aumenta el riesgo
                </span>
                <span className="flex items-center gap-1.5 font-medium text-emerald-600">
                  <span className="inline-block h-2.5 w-2.5 rounded-full bg-emerald-400" />
                  Reduce el riesgo
                </span>
                <span className="ml-auto text-slate-400">
                  El valor SHAP representa el cambio en la probabilidad log-odds.
                </span>
              </div>

              {/* Variables que aumentan el riesgo */}
              {positives.length > 0 && (
                <div>
                  <p className="mb-2 text-[11px] font-bold uppercase tracking-wider text-rose-500">
                    Factores que aumentan el riesgo
                  </p>
                  <div className="space-y-3">
                    {positives.map((c) => (
                      <ContributionBar key={c.feature} contribution={c} maxAbs={maxAbs} />
                    ))}
                  </div>
                </div>
              )}

              {/* Variables que reducen el riesgo */}
              {negatives.length > 0 && (
                <div>
                  <p className="mb-2 text-[11px] font-bold uppercase tracking-wider text-emerald-600">
                    Factores que reducen el riesgo
                  </p>
                  <div className="space-y-3">
                    {negatives.map((c) => (
                      <ContributionBar key={c.feature} contribution={c} maxAbs={maxAbs} />
                    ))}
                  </div>
                </div>
              )}

              {/* Disclaimer clínico */}
              <p className="rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-[10px] leading-relaxed text-slate-500">
                <strong>Nota metodológica:</strong> Los valores SHAP miden la contribución
                marginal de cada variable a la salida del modelo en esta observación concreta.
                No implican causalidad clínica. La interpretación debe realizarse siempre en
                contexto con el juicio del profesional de salud.
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}
