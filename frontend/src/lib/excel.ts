import * as XLSX from "xlsx";
import type {
  ModelSchema,
  PredictionResponse,
} from "../types/api";

const TEMPLATE_SHEET_NAME = "Pacientes";
const HOW_TO_SHEET_NAME = "Instrucciones";

function isIntDtype(dtype: string): boolean {
  const d = dtype.toLowerCase();
  return d.startsWith("int") || d.startsWith("uint") || d.startsWith("bool");
}

function dtypeLabel(dtype: string): string {
  if (isIntDtype(dtype)) return "entero";
  if (dtype.toLowerCase().startsWith("float")) return "decimal";
  return dtype;
}

function templateFilename(schema: ModelSchema): string {
  return `plantilla_${schema.target}_${schema.algorithm}.xlsx`;
}

function resultsFilename(schema: ModelSchema): string {
  return `predicciones_${schema.target}_${schema.algorithm}.xlsx`;
}

/**
 * Genera y descarga un Excel con UNA columna por feature del modelo.
 * Fila 1: nombres exactos (matchean el manifest, NO se deben modificar).
 * Fila 2: tipo de dato sugerido (informativo, no es leído al re-subir).
 * Fila 3: ejemplo con las medianas del training set.
 *
 * El parser luego lee desde la fila 4 en adelante como "filas de datos del paciente".
 */
export function downloadTemplate(schema: ModelSchema): void {
  const featureNames = schema.features.map((f) => f.name);

  // Hoja principal con headers + filas guía.
  const headerRow = featureNames;
  const typeRow = schema.features.map((f) => `(${dtypeLabel(f.dtype)})`);
  const medianRow = schema.features.map((f) =>
    f.median != null ? f.median : "",
  );
  const aoa: (string | number)[][] = [headerRow, typeRow, medianRow];

  const ws = XLSX.utils.aoa_to_sheet(aoa);

  // Anchos de columna razonables.
  ws["!cols"] = featureNames.map((n) => ({
    wch: Math.min(Math.max(n.length + 2, 12), 40),
  }));

  // Hoja de instrucciones — lenguaje neutral institucional.
  const instructions = [
    ["Plantilla para predicción preanestésica"],
    ["Fundación Valle del Lili"],
    [`Modelo: ${schema.algorithm}  ·  Tipo de riesgo: ${schema.target}`],
    [],
    ["Instrucciones de uso:"],
    [`1. Acceder a la hoja "${TEMPLATE_SHEET_NAME}".`],
    ["2. No modificar la fila 1: contiene los nombres exactos de las variables esperadas por el modelo."],
    ["3. La fila 2 indica el tipo de dato esperado (entero o decimal). Es informativa y se omite al procesar."],
    ["4. La fila 3 contiene las medianas del conjunto de entrenamiento. Puede emplearse como referencia o eliminarse."],
    ["5. A partir de la fila siguiente, ingresar un paciente por fila."],
    ["6. Las variables sin valor serán imputadas automáticamente por el modelo según la estrategia indicada."],
    ["7. Cargar el archivo en la aplicación. Una sola fila genera una predicción individual; varias filas generan predicción por lotes."],
    [],
    ["Información del modelo:"],
    [`Umbral óptimo: ${schema.threshold}`],
    [`Calibración: ${schema.calibrated ? "Sí" : "No"}${schema.calibration_method ? ` (${schema.calibration_method})` : ""}`],
    [`Prevalencia (entrenamiento): ${schema.prevalence?.train ?? "—"}`],
    [`Estrategia de imputación: ${schema.imputation.strategy} (valor=${schema.imputation.value ?? "—"})`],
  ];
  const wsInstr = XLSX.utils.aoa_to_sheet(instructions);
  wsInstr["!cols"] = [{ wch: 80 }];

  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, wsInstr, HOW_TO_SHEET_NAME);
  XLSX.utils.book_append_sheet(wb, ws, TEMPLATE_SHEET_NAME);

  XLSX.writeFile(wb, templateFilename(schema));
}

export interface ParsedUpload {
  rows: Record<string, number | null>[];
  warnings: string[];
  errors: string[];
}

/**
 * Lee un Excel subido por el usuario y produce filas de features ya tipadas
 * (números o null). Valida nombres de columnas contra el schema.
 *
 * Política:
 *  - Las columnas del Excel que no existan en el schema → ERROR (la API las rechazaría igual).
 *  - Las columnas del schema que no aparezcan en el Excel → warning (se imputarán).
 *  - Filas 100% vacías → se descartan silenciosamente.
 *  - La fila de "tipo de dato" (todos los valores en formato `(entero)`/`(decimal)`)
 *    se detecta y descarta automáticamente.
 */
export async function parseUpload(
  file: File,
  schema: ModelSchema,
): Promise<ParsedUpload> {
  const buf = await file.arrayBuffer();
  const wb = XLSX.read(buf, { type: "array" });

  // Preferir hoja "Pacientes"; si no existe, tomar la primera.
  const sheetName = wb.SheetNames.includes(TEMPLATE_SHEET_NAME)
    ? TEMPLATE_SHEET_NAME
    : wb.SheetNames[0];
  if (!sheetName) {
    return { rows: [], warnings: [], errors: ["El Excel no tiene hojas."] };
  }
  const ws = wb.Sheets[sheetName];
  const json = XLSX.utils.sheet_to_json<Record<string, unknown>>(ws, {
    raw: true,
    defval: null,
  });

  if (json.length === 0) {
    return { rows: [], warnings: [], errors: ["La hoja del archivo se encuentra vacía."] };
  }

  // Construir maps de validación.
  const schemaNames = new Set(schema.features.map((f) => f.name));
  const dtypeMap = new Map(
    schema.features.map((f) => [f.name, f.dtype] as const),
  );
  const errors: string[] = [];
  const warnings: string[] = [];

  // Detectar columnas extra o faltantes.
  const fileColumns = new Set(Object.keys(json[0]));
  const extraColumns = [...fileColumns].filter((c) => !schemaNames.has(c));
  const missingColumns = schema.features
    .map((f) => f.name)
    .filter((n) => !fileColumns.has(n));

  if (extraColumns.length > 0) {
    errors.push(
      `Columnas no reconocidas por el modelo (deben eliminarse del archivo): ${extraColumns.join(", ")}`,
    );
  }
  if (missingColumns.length > 0) {
    warnings.push(
      `${missingColumns.length} variable${missingColumns.length === 1 ? "" : "s"} del modelo no se encuentran en el archivo. Serán imputadas automáticamente al generar la predicción.`,
    );
  }

  // Convertir filas: castear a int/float según dtype.
  const rows: Record<string, number | null>[] = [];
  json.forEach((row, idx) => {
    // Saltar fila si TODOS los valores son null/vacíos.
    const someValue = Object.values(row).some(
      (v) => v !== null && v !== undefined && String(v).trim() !== "",
    );
    if (!someValue) return;

    // Detectar y saltar la fila de "tipo de dato" auto-generada por la plantilla.
    const looksLikeTypeRow = Object.values(row).every((v) => {
      if (v == null) return true;
      const s = String(v).trim();
      return s === "" || /^\((entero|decimal|.+)\)$/.test(s);
    });
    if (looksLikeTypeRow) return;

    const out: Record<string, number | null> = {};
    for (const name of schemaNames) {
      const raw = row[name];
      if (raw == null || raw === "" || (typeof raw === "string" && raw.trim() === "")) {
        out[name] = null;
        continue;
      }
      const dtype = dtypeMap.get(name) ?? "float64";
      if (typeof raw === "number") {
        out[name] = isIntDtype(dtype) ? Math.trunc(raw) : raw;
      } else {
        const parsed = isIntDtype(dtype)
          ? parseInt(String(raw), 10)
          : parseFloat(String(raw));
        if (Number.isFinite(parsed)) {
          out[name] = parsed;
        } else {
          errors.push(
            `Fila ${idx + 2}, columna "${name}": el valor "${raw}" no es un número válido.`,
          );
          out[name] = null;
        }
      }
    }
    rows.push(out);
  });

  return { rows, warnings, errors };
}

/** Exporta predicciones (single o batch) a Excel — útil para auditoría. */
export function downloadResults(
  schema: ModelSchema,
  inputs: Record<string, number | null>[],
  predictions: PredictionResponse[],
): void {
  if (inputs.length !== predictions.length) {
    throw new Error("inputs.length !== predictions.length");
  }
  const aoa: (string | number | null)[][] = [];

  // Header = predicción + features de input
  const featureNames = schema.features.map((f) => f.name);
  aoa.push([
    "predicted_class",
    "probability",
    "risk_level",
    "threshold",
    "calibrated",
    "warnings",
    ...featureNames,
  ]);

  for (let i = 0; i < predictions.length; i++) {
    const p = predictions[i];
    const inp = inputs[i];
    aoa.push([
      p.predicted_class,
      p.probability,
      p.risk_level,
      p.threshold,
      p.calibrated ? "true" : "false",
      p.warnings.join("|"),
      ...featureNames.map((n) => (inp[n] == null ? "" : (inp[n] as number))),
    ]);
  }

  const ws = XLSX.utils.aoa_to_sheet(aoa);
  ws["!cols"] = aoa[0].map((c) => ({
    wch: Math.min(Math.max(String(c).length + 2, 12), 40),
  }));
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, "Predicciones");
  XLSX.writeFile(wb, resultsFilename(schema));
}
