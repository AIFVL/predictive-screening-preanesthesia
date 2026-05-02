import { useEffect, useMemo, useState } from "react";
import type { ModelSchema } from "../types/api";

interface Props {
  schema: ModelSchema;
  loading: boolean;
  onSubmit: (features: Record<string, number | null>) => void;
}

function isIntDtype(dtype: string): boolean {
  const d = dtype.toLowerCase();
  return d.startsWith("int") || d.startsWith("uint") || d.startsWith("bool");
}

function parseValue(raw: string, dtype: string): number | null {
  if (raw === "") return null;
  const n = isIntDtype(dtype) ? parseInt(raw, 10) : parseFloat(raw);
  return Number.isFinite(n) ? n : null;
}

function formatMedianForPlaceholder(median: number, dtype: string): string {
  if (isIntDtype(dtype)) return String(Math.trunc(median));
  // Máximo 4 decimales; sin ceros finales innecesarios.
  return parseFloat(median.toFixed(4)).toString();
}

export function DynamicForm({ schema, loading, onSubmit }: Props) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [filter, setFilter] = useState("");

  useEffect(() => {
    setValues({});
    setFilter("");
  }, [schema.model_id]);

  const filteredFeatures = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return schema.features;
    return schema.features.filter((f) => f.name.toLowerCase().includes(q));
  }, [schema.features, filter]);

  const filledCount = Object.values(values).filter((v) => v !== "").length;
  const total = schema.features.length;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const features: Record<string, number | null> = {};
    for (const f of schema.features) {
      const raw = values[f.name];
      features[f.name] = raw == null || raw === "" ? null : parseValue(raw, f.dtype);
    }
    onSubmit(features);
  };

  const fillAllWithMedians = () => {
    const next: Record<string, string> = {};
    for (const f of schema.features) {
      next[f.name] = f.median != null ? String(f.median) : "";
    }
    setValues(next);
  };

  const clearAll = () => setValues({});

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h3 className="text-base font-semibold text-fvl-700">
            Variables clínicas del paciente
          </h3>
          <p className="mt-1 text-sm text-slate-600">
            Las variables sin valor se imputan automáticamente según la
            estrategia del modelo
            <span className="ml-1 text-slate-500">
              (estrategia: <code className="rounded bg-fvl-surface px-1">{schema.imputation.strategy}</code>
              {schema.imputation.value !== undefined && (
                <>, valor: <code className="rounded bg-fvl-surface px-1">{schema.imputation.value}</code></>
              )}).
            </span>
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className="rounded-full border border-fvl-mint-border bg-fvl-mint px-3 py-1 text-xs font-semibold text-fvl-700 tabular-nums">
            {filledCount} / {total} completadas
          </span>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <div className="relative flex-1 min-w-[240px]">
          <input
            type="search"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Buscar variable…"
            aria-label="Buscar variable"
            className="w-full rounded-lg border border-fvl-line bg-white px-4 py-2.5 pl-10 text-sm shadow-sm focus:border-fvl-lime focus:outline-none focus:ring-2 focus:ring-fvl-lime/30"
          />
          <span
            aria-hidden
            className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
          >
            ⌕
          </span>
        </div>
        <button
          type="button"
          onClick={fillAllWithMedians}
          className="rounded-lg border border-fvl-line bg-white px-4 py-2.5 text-sm font-medium text-fvl-700 transition hover:border-fvl-mint-border hover:bg-fvl-mint/40"
        >
          Cargar valores medianos
        </button>
        <button
          type="button"
          onClick={clearAll}
          className="rounded-lg border border-fvl-line bg-white px-4 py-2.5 text-sm font-medium text-slate-600 transition hover:bg-slate-50"
        >
          Limpiar
        </button>
      </div>

      <div className="grid max-h-[460px] gap-4 overflow-y-auto rounded-xl border border-fvl-line bg-fvl-surface p-5 sm:grid-cols-2 lg:grid-cols-3">
        {filteredFeatures.map((f) => (
          <div key={f.name} className="space-y-1">
            <label
              htmlFor={`f-${f.name}`}
              className="block text-xs font-semibold text-fvl-700"
              title={f.dtype}
            >
              {f.name}
              <span className="ml-1 text-[10px] font-normal text-slate-400">
                ({f.dtype})
              </span>
            </label>
            <input
              id={`f-${f.name}`}
              type="number"
              step={isIntDtype(f.dtype) ? "1" : "any"}
              value={values[f.name] ?? ""}
              placeholder={
                f.median != null
                  ? `mediana: ${formatMedianForPlaceholder(f.median, f.dtype)}`
                  : "—"
              }
              onChange={(e) =>
                setValues((prev) => ({ ...prev, [f.name]: e.target.value }))
              }
              className="w-full rounded-md border border-fvl-line bg-white px-3 py-1.5 text-sm tabular-nums shadow-sm focus:border-fvl-lime focus:outline-none focus:ring-2 focus:ring-fvl-lime/30"
            />
          </div>
        ))}
        {filteredFeatures.length === 0 && (
          <p className="col-span-full text-sm text-slate-500">
            Ninguna variable coincide con la búsqueda.
          </p>
        )}
      </div>

      <div className="flex justify-end pt-2">
        <button
          type="submit"
          disabled={loading}
          className="rounded-full bg-fvl-lime px-7 py-2.5 text-sm font-bold text-fvl-700 shadow-sm transition hover:bg-fvl-lime-hover focus:outline-none focus:ring-2 focus:ring-fvl-lime focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loading ? "Procesando…" : "Generar predicción"}
        </button>
      </div>
    </form>
  );
}
