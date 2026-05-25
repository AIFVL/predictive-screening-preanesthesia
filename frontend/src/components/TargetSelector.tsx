import type { TargetInfo } from "../types/api";

interface Props {
  targets: TargetInfo[];
  value: string | null;
  onChange: (slug: string) => void;
}

export function TargetSelector({ targets, value, onChange }: Props) {
  if (targets.length === 0) {
    return (
      <p className="text-sm text-slate-500">
        No se encontraron tipos de riesgo disponibles.
      </p>
    );
  }

  return (
    <div className="grid gap-4 md:grid-cols-2">
      {targets.map((t) => {
        const selected = t.slug === value;
        return (
          <button
            type="button"
            key={t.slug}
            onClick={() => onChange(t.slug)}
            aria-pressed={selected}
            className={[
              "group relative flex flex-col rounded-2xl border p-5 text-left transition focus:outline-none focus:ring-2 focus:ring-fvl-lime focus:ring-offset-2",
              selected
                ? "border-fvl-700 bg-fvl-mint shadow-sm"
                : "border-fvl-line bg-white hover:border-fvl-mint-border hover:bg-fvl-mint/40",
            ].join(" ")}
          >
            {selected && (
              <span
                aria-hidden
                className="absolute left-0 top-0 h-full w-1 rounded-l-2xl bg-fvl-lime"
              />
            )}
            <div className="mb-2 flex items-center gap-2">
              <h3 className="text-base font-bold text-fvl-700">
                {t.display_name}
              </h3>
              {t.recommended && (
                <span className="rounded-full bg-fvl-lime px-2.5 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-fvl-700">
                  Recomendado
                </span>
              )}
            </div>
            <p className="text-sm leading-relaxed text-slate-700">
              {t.description}
            </p>
            <p className="mt-3 text-xs text-slate-500">
              {t.n_models} modelos disponibles
            </p>
          </button>
        );
      })}
    </div>
  );
}
