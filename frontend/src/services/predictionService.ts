/**
 * predictionService.ts
 * --------------------
 * Connects the AgroPredict frontend to the FastAPI backend.
 *
 * The form currently collects: temperature, humidity, rainfall,
 * soilType, ph, nitrogen, phosphorus, potassium.
 *
 * The backend expects: state, year, season + optional climate fields.
 *
 * This service maps the form inputs to the API schema and maps the
 * API response back to the shape the existing UI expects
 * (crop, confidence, advice).
 */

// ── config ────────────────────────────────────────────────────────────────────
// Set VITE_API_URL in your .env.local file.
// Example: VITE_API_URL=http://localhost:8000
// Falls back to localhost:8000 in development.
const API_BASE =
  (import.meta as any).env?.VITE_API_URL?.replace(/\/$/, "") ??
  "http://localhost:8000";

// ── types (what the form uses — keep these unchanged) ─────────────────────────
export interface PredictionInputs {
  // Location & time — new required fields added to the form
  state:   string;
  year:    number;
  season:  string;

  // Climate — optional, sent to API when provided
  temperature:  number;   // maps to T2M
  humidity:     number;   // maps to RH2M
  rainfall:     number;   // maps to PRECTOTCORR

  // Kept for form compatibility — not sent to model
  soilType:   string;
  ph:         number;
  nitrogen:   number;
  phosphorus: number;
  potassium:  number;

  // Optional prior yields for lag features
  yieldLag1?: number;
  yieldLag2?: number;
}

export interface ConformalInterval {
  coverage:   number;
  lower:      number;
  upper:      number;
  half_width: number;
}

export interface PredictionResult {
  // Shape the existing UI expects
  crop:       string;   // "X.XX t/ha" — yield formatted as a string
  confidence: number;   // mapped from R² (0.9145 → 0.9145)
  advice:     string;   // human-readable interpretation

  // Extended fields (available for future UI improvements)
  predicted_yield: number;
  unit:            string;
  intervals:       ConformalInterval[];
  latency_ms:      number | null;
}

// ── request builder ───────────────────────────────────────────────────────────
function buildRequest(inputs: PredictionInputs): Record<string, unknown> {
  const req: Record<string, unknown> = {
    state:  inputs.state.trim().toUpperCase(),
    year:   inputs.year,
    season: inputs.season,
  };

  // Climate fields — only include if the user provided them
  if (inputs.temperature !== undefined) req.T2M         = inputs.temperature;
  if (inputs.humidity    !== undefined) req.RH2M        = inputs.humidity;
  if (inputs.rainfall    !== undefined) req.PRECTOTCORR = inputs.rainfall;
  if (inputs.yieldLag1   !== undefined) req.yield_lag1  = inputs.yieldLag1;
  if (inputs.yieldLag2   !== undefined) req.yield_lag2  = inputs.yieldLag2;

  return req;
}

// ── response interpreter ──────────────────────────────────────────────────────
function interpretYield(yieldTha: number): string {
  if (yieldTha < 1.5) return "Low — consider soil improvement";
  if (yieldTha < 3.0) return "Moderate — typical for rain-fed conditions";
  if (yieldTha < 4.5) return "Good — well-managed crop expected";
  return "Excellent — optimal growing conditions";
}

function buildAdvice(
  yieldTha: number,
  state: string,
  season: string,
  interval80: ConformalInterval | undefined
): string {
  const label = interpretYield(yieldTha);
  const range = interval80
    ? ` Expected range: ${interval80.lower.toFixed(2)}–${interval80.upper.toFixed(2)} t/ha (80% confidence).`
    : "";
  return (
    `${label}. Predicted yield for ${state} ${season}: ${yieldTha.toFixed(2)} t/ha.` +
    range +
    ` Model R² = 0.9145 on held-out test set.`
  );
}

// ── main export ───────────────────────────────────────────────────────────────
export async function predictCrop(
  inputs: PredictionInputs
): Promise<PredictionResult> {
  const requestBody = buildRequest(inputs);

  let response: Response;
  try {
    response = await fetch(`${API_BASE}/predict`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(requestBody),
    });
  } catch (networkError) {
    throw new Error(
      `Cannot reach the prediction API at ${API_BASE}. ` +
      "Make sure the backend is running: uvicorn api:app --port 8000"
    );
  }

  if (!response.ok) {
    let detail = `API error ${response.status}`;
    try {
      const err = await response.json();
      detail = err?.detail ?? detail;
    } catch {}
    throw new Error(detail);
  }

  const data = await response.json();

  // Map API response → shape the existing UI expects
  const yieldTha: number  = data.predicted_yield;
  const intervals: ConformalInterval[] = data.intervals ?? [];
  const interval80 = intervals.find((i) => i.coverage === 0.8);

  return {
    // Fields the existing ResultDisplay component uses
    crop:       `${yieldTha.toFixed(2)} t/ha`,
    confidence: 0.9145,                          // model-level R²
    advice:     buildAdvice(yieldTha, inputs.state, inputs.season, interval80),

    // Extended fields
    predicted_yield: yieldTha,
    unit:            data.unit ?? "t/ha",
    intervals,
    latency_ms:      data.latency_ms ?? null,
  };
}
