/**
 * API utility functions for iQueue frontend
 * Fetches real data from the backend API instead of using hardcoded data
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

/** Base URL for non-/api routes (e.g. POST /predict) */
function apiRootUrl(): string {
  const base = API_BASE_URL.replace(/\/$/, '');
  return base.endsWith('/api') ? base.slice(0, -4) : base.replace(/\/api$/, '');
}

export async function fetchModelPerformance() {
  try {
    const response = await fetch(`${API_BASE_URL}/model-performance`);
    if (!response.ok) throw new Error('Failed to fetch model performance');
    return await response.json();
  } catch (error) {
    console.error('Error fetching model performance:', error);
    throw error;
  }
}

export async function fetchFeatureImportance() {
  try {
    const response = await fetch(`${API_BASE_URL}/feature-importance`);
    if (!response.ok) throw new Error('Failed to fetch feature importance');
    return await response.json();
  } catch (error) {
    console.error('Error fetching feature importance:', error);
    throw error;
  }
}

export async function fetchHistoricalAnalytics() {
  try {
    const response = await fetch(`${API_BASE_URL}/historical-analytics`);
    if (!response.ok) throw new Error('Failed to fetch historical analytics');
    return await response.json();
  } catch (error) {
    console.error('Error fetching historical analytics:', error);
    throw error;
  }
}

export async function fetchPredictiveAnalytics() {
  try {
    const response = await fetch(`${API_BASE_URL}/predictive-analytics`);
    if (!response.ok) throw new Error('Failed to fetch predictive analytics');
    return await response.json();
  } catch (error) {
    console.error('Error fetching predictive analytics:', error);
    throw error;
  }
}

export async function fetchDatasetSummary() {
  try {
    const response = await fetch(`${API_BASE_URL}/dataset-summary`);
    if (!response.ok) throw new Error('Failed to fetch dataset summary');
    return await response.json();
  } catch (error) {
    console.error('Error fetching dataset summary:', error);
    throw error;
  }
}

/** Returns the real calendar date for the given day name within the current week (Mon-Sat). */
function getDateForDayThisWeek(dayName: string): string {
  const ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  const today = new Date();
  // Snap to Monday of this week (local time)
  const dow = today.getDay(); // 0=Sun, 1=Mon … 6=Sat
  const toMonday = dow === 0 ? -6 : 1 - dow;
  const monday = new Date(today);
  monday.setDate(today.getDate() + toMonday);
  monday.setHours(0, 0, 0, 0);

  const idx = ORDER.indexOf(dayName);
  const target = new Date(monday);
  target.setDate(monday.getDate() + (idx >= 0 ? idx : 0));

  // Format as YYYY-MM-DD using LOCAL date components (not UTC)
  const y = target.getFullYear();
  const m = String(target.getMonth() + 1).padStart(2, '0');
  const d = String(target.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

const HOLIDAY_DEMO_DATE = '2026-01-01';

export type LivePredictInput = {
  day: string;
  time: string;
  queueLength?: number;
  isHoliday?: boolean;
};

export async function fetchLivePrediction(
  input: LivePredictInput,
  options?: { signal?: AbortSignal }
) {
  const hour = parseInt(input.time.split(':')[0], 10);
  const date = input.isHoliday ? HOLIDAY_DEMO_DATE : getDateForDayThisWeek(input.day);
  const body = {
    date,
    hour,
    day_of_week: input.day,
    queue_length_at_arrival: input.queueLength ?? 12,
  };
  const root = apiRootUrl();
  const response = await fetch(`${root}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: options?.signal,
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error((err as { error?: string }).error || 'Prediction request failed');
  }
  return response.json() as Promise<{
    success: boolean;
    prediction: number;
    confidence?: number | null;
    range?: { p10: number; p50?: number; p90: number } | null;
    congestion?: string;
    recommendation?: string;
    unit: string;
    method?: string;
    timestamp: string;
  }>;
}

export async function fetchWeeklyForecast(date: string) {
  const root = apiRootUrl();
  const response = await fetch(`${root}/api/weekly-forecast?date=${date}`);
  if (!response.ok) throw new Error('Failed to fetch weekly forecast');
  return response.json() as Promise<{
    weekLabel: string;
    weekOf: string;
    days: Array<{
      date: string;
      dayName: string;
      shortDate: string;
      isHoliday: boolean;
      overall: number | null;
      congestion: string;
      bestTime: string | null;
      bestWait: number | null;
      bestP10: number | null;
      bestP90: number | null;
      worstTime: string | null;
      worstWait: number | null;
      hourly: Array<{ hour: string; wait: number; p10: number; p90: number }>;
    }>;
  }>;
}
