import { useEffect, useMemo, useState } from "react";
import { ApiClientError, api, apiBaseUrl } from "./api/client";
import { BatchPredictionResults } from "./components/BatchPredictionResults";
import { DynamicForm } from "./components/DynamicForm";
import { ExcelUploader } from "./components/ExcelUploader";
import { ModelSelector } from "./components/ModelSelector";
import { PredictionResult } from "./components/PredictionResult";
import { ShapExplanation } from "./components/ShapExplanation";
import { TargetSelector } from "./components/TargetSelector";
import type {
  ModelSchema,
  ModelSummary,
  PredictionResponse,
  TargetInfo,
} from "./types/api";

type InputMode = "manual" | "excel";

export default function App() {
  const [targets, setTargets] = useState<TargetInfo[]>([]);
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [selectedTarget, setSelectedTarget] = useState<string | null>(null);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string | null>(null);
  const [schema, setSchema] = useState<ModelSchema | null>(null);

  const [mode, setMode] = useState<InputMode>("manual");

  const [singlePrediction, setSinglePrediction] = useState<PredictionResponse | null>(null);
  const [singleFeatures, setSingleFeatures] = useState<Record<string, number | null> | null>(null);
  const [batchPredictions, setBatchPredictions] = useState<PredictionResponse[] | null>(null);
  const [batchInputs, setBatchInputs] = useState<Record<string, number | null>[] | null>(null);

  const [bootstrapError, setBootstrapError] = useState<string | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [predictError, setPredictError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.listTargets(), api.listModels()])
      .then(([t, m]) => {
        setTargets(t);
        setModels(m);
        const recommended = t.find((x) => x.recommended) ?? t[0];
        if (recommended) setSelectedTarget(recommended.slug);
      })
      .catch((err: ApiClientError) => {
        setBootstrapError(
          `No fue posible conectarse con el servicio en ${apiBaseUrl}: ${err.message}`,
        );
      });
  }, []);

  const modelsForTarget = useMemo(
    () => models.filter((m) => m.target === selectedTarget),
    [models, selectedTarget],
  );

  useEffect(() => {
    if (selectedTarget == null) return;
    const recommended = modelsForTarget.find((m) => m.recommended);
    setSelectedAlgorithm(recommended?.algorithm ?? modelsForTarget[0]?.algorithm ?? null);
    resetResults();
  }, [selectedTarget, modelsForTarget]);

  useEffect(() => {
    if (!selectedTarget || !selectedAlgorithm) return;
    resetResults();
    api.getSchema(selectedTarget, selectedAlgorithm)
      .then(setSchema)
      .catch((err: ApiClientError) => setPredictError(err.message));
  }, [selectedTarget, selectedAlgorithm]);

  const resetResults = () => {
    setSinglePrediction(null);
    setSingleFeatures(null);
    setBatchPredictions(null);
    setBatchInputs(null);
    setPredictError(null);
  };

  const handleManualSubmit = async (features: Record<string, number | null>) => {
    if (!selectedTarget || !selectedAlgorithm) return;
    setPredicting(true);
    resetResults();
    try {
      const result = await api.predict(selectedTarget, selectedAlgorithm, features);
      setSinglePrediction(result);
      setSingleFeatures(features);
    } catch (err) {
      const e = err as ApiClientError;
      setPredictError(
        e.field ? `Validación fallida en "${e.field}": ${e.message}` : e.message,
      );
    } finally {
      setPredicting(false);
    }
  };

  const handleExcelSubmit = async (rows: Record<string, number | null>[]) => {
    if (!selectedTarget || !selectedAlgorithm) return;
    setPredicting(true);
    resetResults();
    try {
      if (rows.length === 1) {
        const result = await api.predict(selectedTarget, selectedAlgorithm, rows[0]);
        setSinglePrediction(result);
        setSingleFeatures(rows[0]);
        setBatchInputs(rows);
        setBatchPredictions([result]);
      } else {
        const result = await api.predictBatch(selectedTarget, selectedAlgorithm, rows);
        setBatchPredictions(result.predictions);
        setBatchInputs(rows);
      }
    } catch (err) {
      const e = err as ApiClientError;
      setPredictError(
        e.field ? `Validación fallida en "${e.field}": ${e.message}` : e.message,
      );
    } finally {
      setPredicting(false);
    }
  };

  return (
    <div className="min-h-screen bg-fvl-surface">
      {/* Header institucional */}
      <header className="bg-fvl-700 text-white">
        <div className="mx-auto flex max-w-6xl flex-col gap-1 px-6 py-6 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-fvl-lime-soft">
              Fundación Valle del Lili
            </p>
            <h1 className="mt-1 text-2xl font-bold leading-tight md:text-[26px]">
              Sistema de predicción preanestésica
            </h1>
          </div>
          <p className="max-w-md text-sm text-white/75 md:text-right">
            Herramienta clínica de apoyo a la decisión para la valoración
            preoperatoria de pacientes adultos.
          </p>
        </div>
        <div className="h-1 bg-fvl-lime" />
      </header>

      <main className="mx-auto max-w-6xl px-6 py-10">
        {bootstrapError && (
          <div className="mb-8 rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800">
            {bootstrapError}
          </div>
        )}

        <div className="space-y-10">
          <Section
            number="1"
            title="Tipo de riesgo"
            description="Seleccione el dominio clínico sobre el cual se realizará la predicción."
          >
            <TargetSelector
              targets={targets}
              value={selectedTarget}
              onChange={setSelectedTarget}
            />
          </Section>

          {selectedTarget && (
            <Section
              number="2"
              title="Modelo predictivo"
              description="Compare el desempeño de los modelos disponibles para el riesgo seleccionado y elija el más apropiado."
            >
              <ModelSelector
                models={modelsForTarget}
                value={selectedAlgorithm}
                onChange={setSelectedAlgorithm}
              />
            </Section>
          )}

          {schema && (
            <Section
              number="3"
              title="Datos del paciente"
              description="Ingrese los valores clínicos manualmente o cargue un archivo Excel con uno o varios pacientes."
            >
              <div className="space-y-6 rounded-2xl border border-fvl-line bg-white p-6 shadow-sm">
                <ModeToggle mode={mode} onChange={setMode} />
                <div>
                  {mode === "manual" ? (
                    <DynamicForm
                      schema={schema}
                      loading={predicting}
                      onSubmit={handleManualSubmit}
                    />
                  ) : (
                    <ExcelUploader
                      schema={schema}
                      loading={predicting}
                      onSubmit={handleExcelSubmit}
                    />
                  )}
                </div>
              </div>
            </Section>
          )}

          {predictError && (
            <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800">
              {predictError}
            </div>
          )}

          {(singlePrediction && !batchPredictions) || (batchPredictions && schema && batchInputs) ? (
            <Section
              number="4"
              title="Resultado"
              description="Probabilidad calibrada y nivel de riesgo estimado por el modelo seleccionado."
            >
              <div className="space-y-4">
                {singlePrediction && !batchPredictions && (
                  <PredictionResult result={singlePrediction} />
                )}
                {batchPredictions && batchInputs && schema && (
                  <BatchPredictionResults
                    schema={schema}
                    inputs={batchInputs}
                    predictions={batchPredictions}
                  />
                )}
                {/* Explicabilidad SHAP — sólo para predicciones individuales */}
                {singlePrediction && singleFeatures && selectedTarget && selectedAlgorithm && (
                  <ShapExplanation
                    target={selectedTarget}
                    algorithm={selectedAlgorithm}
                    features={singleFeatures}
                  />
                )}
              </div>
            </Section>
          ) : null}
        </div>
      </main>

      <footer className="border-t border-fvl-line bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4 text-xs text-slate-500">
          <span>Fundación Valle del Lili · Uso clínico institucional</span>
          <span>
            API: <code className="rounded bg-fvl-surface px-1.5 py-0.5">{apiBaseUrl}</code>
          </span>
        </div>
      </footer>
    </div>
  );
}

function Section({
  number,
  title,
  description,
  children,
}: {
  number: string;
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <section>
      <div className="mb-4 flex items-baseline gap-3">
        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-fvl-700 text-xs font-bold text-white">
          {number}
        </span>
        <div>
          <h2 className="text-lg font-bold text-fvl-700">{title}</h2>
          <p className="text-sm text-slate-600">{description}</p>
        </div>
      </div>
      {children}
    </section>
  );
}

function ModeToggle({
  mode,
  onChange,
}: {
  mode: InputMode;
  onChange: (m: InputMode) => void;
}) {
  const baseBtn =
    "flex-1 rounded-md px-4 py-2 text-sm font-semibold transition focus:outline-none focus:ring-2 focus:ring-fvl-lime focus:ring-offset-2 sm:flex-initial";
  return (
    <div
      role="tablist"
      aria-label="Modo de carga"
      className="inline-flex w-full gap-1 rounded-lg bg-fvl-mint p-1 sm:w-auto"
    >
      <button
        type="button"
        role="tab"
        aria-selected={mode === "manual"}
        onClick={() => onChange("manual")}
        className={`${baseBtn} ${
          mode === "manual"
            ? "bg-white text-fvl-700 shadow-sm"
            : "text-fvl-700/70 hover:text-fvl-700"
        }`}
      >
        Carga manual
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={mode === "excel"}
        onClick={() => onChange("excel")}
        className={`${baseBtn} ${
          mode === "excel"
            ? "bg-white text-fvl-700 shadow-sm"
            : "text-fvl-700/70 hover:text-fvl-700"
        }`}
      >
        Carga por archivo Excel
      </button>
    </div>
  );
}
