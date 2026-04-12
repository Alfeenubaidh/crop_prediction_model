/**
 * predictionService.ts
 */

const API_BASE =
  (import.meta as any).env?.VITE_API_URL?.replace(/\/$/, "") ??
  "http://localhost:8000";

export interface PredictionInputs {
  state:        string;
  year:         number;
  season:       string;
  temperature:  number;
  humidity:     number;
  rainfall:     number;
  soilType:     string;
  ph:           number;
  nitrogen:     number;
  phosphorus:   number;
  potassium:    number;
  yieldLag1?:   number;
  yieldLag2?:   number;
  tempMax:      number;  // T2M_MAX
  tempMin:      number;  // T2M_MIN
  solarRad:     number;  // ALLSKY_SFC_SW_DWN
  windSpeed:    number;  // WS2M
}

export interface ConformalInterval {
  coverage:   number;
  lower:      number;
  upper:      number;
  half_width: number;
}

export interface PredictionResult {
  crop:            string;
  confidence:      number;
  advice:          string;
  predicted_yield: number;
  unit:            string;
  intervals:       ConformalInterval[];
  latency_ms:      number | null;
}

function buildRequest(inputs: PredictionInputs): Record<string, unknown> {
  const req: Record<string, unknown> = {
    state:  inputs.state.trim().toUpperCase(),
    year:   inputs.year,
    season: inputs.season,
  };
  if (inputs.temperature !== undefined) req.T2M         = inputs.temperature;
  if (inputs.humidity    !== undefined) req.RH2M        = inputs.humidity;
  if (inputs.rainfall    !== undefined) req.PRECTOTCORR = inputs.rainfall;
  if (inputs.yieldLag1   !== undefined) req.yield_lag1        = inputs.yieldLag1;
  if (inputs.yieldLag2   !== undefined) req.yield_lag2        = inputs.yieldLag2;
  if (inputs.tempMax     !== undefined) req.T2M_MAX           = inputs.tempMax;
  if (inputs.tempMin     !== undefined) req.T2M_MIN           = inputs.tempMin;
  if (inputs.solarRad    !== undefined) req.ALLSKY_SFC_SW_DWN = inputs.solarRad;
  if (inputs.windSpeed   !== undefined) req.WS2M              = inputs.windSpeed;
  return req;
}

function buildAdvice(
  yieldTha: number,
  state: string,
  season: string,
  interval80: ConformalInterval | undefined
): string {
  let label = "Low — consider soil improvement";
  if (yieldTha >= 1.5) label = "Moderate — typical for rain-fed conditions";
  if (yieldTha >= 3.0) label = "Good — well-managed crop expected";
  if (yieldTha >= 4.5) label = "Excellent — optimal growing conditions";
  const range = interval80
    ? ` Expected range: ${interval80.lower.toFixed(2)}–${interval80.upper.toFixed(2)} t/ha (80% confidence).`
    : "";
  return `${label}. Predicted yield for ${state} ${season}: ${yieldTha.toFixed(2)} t/ha.${range} Model R² = 0.9145.`;
}

export async function predictCrop(
  inputs: PredictionInputs
): Promise<PredictionResult> {
  const body = buildRequest(inputs);

  let response: Response;
  try {
    response = await fetch(`${API_BASE}/predict`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(body),
    });
  } catch {
    throw new Error(
      `Cannot reach API at ${API_BASE}. Make sure uvicorn is running.`
    );
  }

  if (!response.ok) {
    let detail = `API error ${response.status}`;
    try { const err = await response.json(); detail = err?.detail ?? detail; } catch {}
    throw new Error(detail);
  }

  const data = await response.json();
  const yieldTha: number = data.predicted_yield;
  const intervals: ConformalInterval[] = data.intervals ?? [];
  const interval80 = intervals.find((i) => i.coverage === 0.8);

  return {
    crop:            `${yieldTha.toFixed(2)} t/ha`,
    confidence:      0.9145,
    advice:          buildAdvice(yieldTha, inputs.state, inputs.season, interval80),
    predicted_yield: yieldTha,
    unit:            data.unit ?? "t/ha",
    intervals,
    latency_ms:      data.latency_ms ?? null,
  };
}
