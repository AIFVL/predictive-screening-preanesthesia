import { useRef, useState } from "react";
import { downloadTemplate, parseUpload, type ParsedUpload } from "../lib/excel";
import type { ModelSchema } from "../types/api";

interface Props {
  schema: ModelSchema;
  loading: boolean;
  onSubmit: (rows: Record<string, number | null>[]) => void;
}

export function ExcelUploader({ schema, loading, onSubmit }: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [parsed, setParsed] = useState<ParsedUpload | null>(null);
  const [parsing, setParsing] = useState(false);

  const handleFile = async (file: File) => {
    setParsing(true);
    try {
      const result = await parseUpload(file, schema);
      setParsed(result);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setParsed({
        rows: [],
        warnings: [],
        errors: [`No fue posible leer el archivo: ${msg}`],
      });
    } finally {
      setParsing(false);
    }
  };

  const onPick = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  };

  const onDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  };

  const reset = () => {
    setParsed(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const ready = parsed && parsed.errors.length === 0 && parsed.rows.length > 0;
  const isBatch = (parsed?.rows.length ?? 0) > 1;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h3 className="text-base font-semibold text-fvl-700">
            Carga por archivo Excel
          </h3>
          <p className="mt-1 text-sm text-slate-600">
            Descargar la plantilla con las {schema.features.length} variables
            requeridas por el modelo, completar los datos y cargar el archivo.
            Se admite una fila por paciente.
          </p>
        </div>
        <button
          type="button"
          onClick={() => downloadTemplate(schema)}
          className="rounded-lg border border-fvl-700 bg-white px-4 py-2 text-sm font-semibold text-fvl-700 shadow-sm transition hover:bg-fvl-mint/40 focus:outline-none focus:ring-2 focus:ring-fvl-lime focus:ring-offset-2"
        >
          Descargar plantilla
        </button>
      </div>

      <label
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
        className="flex min-h-[160px] cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-fvl-mint-border bg-fvl-mint/30 px-6 py-8 text-center transition hover:border-fvl-700 hover:bg-fvl-mint/60"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".xlsx,.xls,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
          onChange={onPick}
          className="hidden"
        />
        {parsing ? (
          <p className="text-sm text-slate-700">Procesando archivo…</p>
        ) : (
          <>
            <p className="text-sm font-semibold text-fvl-700">
              Arrastrar un archivo .xlsx aquí o hacer clic para seleccionarlo
            </p>
            <p className="mt-1 text-xs text-slate-500">
              Se admiten archivos generados a partir de la plantilla. La fila
              guía de tipos de dato se omite automáticamente.
            </p>
          </>
        )}
      </label>

      {parsed && (
        <div className="space-y-4 rounded-2xl border border-fvl-line bg-white p-5 shadow-sm">
          <div className="flex flex-wrap items-baseline gap-2 text-sm">
            <span className="font-semibold text-fvl-700">
              {parsed.rows.length} fila{parsed.rows.length === 1 ? "" : "s"} detectada
              {parsed.rows.length === 1 ? "" : "s"}
            </span>
            {ready && (
              <span className="rounded-full bg-fvl-mint px-2.5 py-0.5 text-xs font-semibold text-fvl-700">
                {isBatch ? "Predicción por lotes" : "Predicción individual"}
              </span>
            )}
          </div>

          {parsed.errors.length > 0 && (
            <div className="rounded-lg border border-red-300 bg-red-50 p-3 text-xs text-red-800">
              <strong className="block mb-1">Errores que impiden continuar:</strong>
              <ul className="list-disc space-y-0.5 pl-5">
                {parsed.errors.slice(0, 8).map((e, i) => (
                  <li key={i}>{e}</li>
                ))}
                {parsed.errors.length > 8 && (
                  <li>… y {parsed.errors.length - 8} adicionales</li>
                )}
              </ul>
            </div>
          )}

          {parsed.warnings.length > 0 && (
            <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-800">
              <strong className="block mb-1">Advertencias:</strong>
              <ul className="list-disc space-y-0.5 pl-5">
                {parsed.warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              disabled={!ready || loading}
              onClick={() => parsed && onSubmit(parsed.rows)}
              className="rounded-full bg-fvl-lime px-6 py-2 text-sm font-bold text-fvl-700 shadow-sm transition hover:bg-fvl-lime-hover focus:outline-none focus:ring-2 focus:ring-fvl-lime focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading
                ? "Procesando…"
                : ready
                ? `Generar ${parsed.rows.length} predicción${parsed.rows.length === 1 ? "" : "es"}`
                : "Corregir errores antes de continuar"}
            </button>
            <button
              type="button"
              onClick={reset}
              className="rounded-full border border-fvl-line bg-white px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-50"
            >
              Reiniciar
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
