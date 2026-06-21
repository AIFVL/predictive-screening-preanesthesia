export interface ApiEnvelope<T> {
  success: boolean;
  data: T | null;
  errors: ApiError[] | null;
  meta: { request_id: string; model_id: string | null };
}

export interface ApiError {
  code: string;
  message: string;
  field: string | null;
}

export interface TargetInfo {
  slug: string;
  display_name: string;
  description: string;
  recommended: boolean;
  n_models: number;
}

export interface PerformanceMetrics {
  roc_auc?: number | null;
  pr_auc?: number | null;
  f2?: number | null;
  f1?: number | null;
  recall?: number | null;
  precision?: number | null;
  specificity?: number | null;
  balanced_accuracy?: number | null;
  brier?: number | null;
  accuracy?: number | null;
}

export interface ModelSummary {
  target: string;
  target_display_name: string;
  algorithm: string;
  model_id: string;
  calibrated: boolean;
  calibration_method: string | null;
  threshold: number;
  performance: PerformanceMetrics;
  warnings: string[];
  recommended: boolean;
}

export interface FeatureSpec {
  name: string;
  dtype: string;
  required: boolean;
  description: string | null;
}

export interface ModelSchema {
  model_id: string;
  target: string;
  algorithm: string;
  features: FeatureSpec[];
  threshold: number;
  threshold_metric: string;
  prevalence: { train?: number; n_positive?: number; n_negative?: number; n_total?: number };
  calibrated: boolean;
  calibration_method: string | null;
  warnings: string[];
  input_example: Record<string, string | number | null> | null;
}

export type RiskLevel = "low" | "moderate" | "elevated" | "high";

export interface PredictionResponse {
  predicted_class: 0 | 1;
  probability: number;
  threshold: number;
  risk_level: RiskLevel;
  calibrated: boolean;
  prevalence_train: number | null;
  warnings: string[];
}

export interface ShapContribution {
  feature: string;
  value: number | null;
  shap_value: number;
}

export interface ExplainResponse {
  contributions: ShapContribution[];
  top_n: number;
  algorithm: string;
  model_id: string;
}
