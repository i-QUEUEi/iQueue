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

// ---------------------------------------------------------------------------
// Simple in-memory cache
// ---------------------------------------------------------------------------
// - Static endpoints (model perf, historical, etc.) → TTL_SESSION (no expiry)
// - Weekly forecast                                 → TTL_30MIN per week key
// - Live Simulation                                 → never cached (interactive)
// ---------------------------------------------------------------------------

const TTL_SESSION = Infinity;   // never expires within a page load

interface CacheEntry<T> { data: T; timestamp: number; ttl: number }
const _cache = new Map<string, CacheEntry<unknown>>();

function cacheGet<T>(key: string): T | undefined {
  const entry = _cache.get(key) as CacheEntry<T> | undefined;
  if (!entry) return undefined;
  if (entry.ttl !== Infinity && Date.now() - entry.timestamp > entry.ttl) {
    _cache.delete(key);
    return undefined;
  }
  return entry.data;
}

function cacheSet<T>(key: string, data: T, ttl: number) {
  _cache.set(key, { data, timestamp: Date.now(), ttl });
}

/** Fetch with cache. If the result is already cached and still fresh, return it immediately. */
async function cachedFetch<T>(
  key: string,
  fetcher: () => Promise<T>,
  ttl: number
): Promise<T> {
  const hit = cacheGet<T>(key);
  if (hit !== undefined) return hit;
  const data = await fetcher();
  cacheSet(key, data, ttl);
  return data;
}

// ---------------------------------------------------------------------------
// Static endpoints — cached for entire session
// ---------------------------------------------------------------------------

export async function fetchModelPerformance() {
  return cachedFetch('model-performance', async () => {
    const response = await fetch(`${API_BASE_URL}/model-performance`);
    if (!response.ok) throw new Error('Failed to fetch model performance');
    return response.json();
  }, TTL_SESSION);
}

export async function fetchFeatureImportance() {
  return cachedFetch('feature-importance', async () => {
    const response = await fetch(`${API_BASE_URL}/feature-importance`);
    if (!response.ok) throw new Error('Failed to fetch feature importance');
    return response.json();
  }, TTL_SESSION);
}

export async function fetchHistoricalAnalytics() {
  return cachedFetch('historical-analytics', async () => {
    const response = await fetch(`${API_BASE_URL}/historical-analytics`);
    if (!response.ok) throw new Error('Failed to fetch historical analytics');
    return response.json();
  }, TTL_SESSION);
}

export async function fetchPredictiveAnalytics() {
  return cachedFetch('predictive-analytics', async () => {
    const response = await fetch(`${API_BASE_URL}/predictive-analytics`);
    if (!response.ok) throw new Error('Failed to fetch predictive analytics');
    return response.json();
  }, TTL_SESSION);
}

export async function fetchDatasetSummary() {
  return cachedFetch('dataset-summary', async () => {
    const response = await fetch(`${API_BASE_URL}/dataset-summary`);
    if (!response.ok) throw new Error('Failed to fetch dataset summary');
    return response.json();
  }, TTL_SESSION);
}

// ---------------------------------------------------------------------------
// Weekly forecast — cached 30 min per week key
// ---------------------------------------------------------------------------

export async function fetchWeeklyForecast(date: string) {
  return cachedFetch(`weekly-forecast:${date}`, async () => {
    const root = apiRootUrl();
    const response = await fetch(`${root}/api/weekly-forecast?date=${date}`);
    if (!response.ok) throw new Error('Failed to fetch weekly forecast');
    return response.json();
  }, TTL_SESSION) as Promise<{
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

// ---------------------------------------------------------------------------
// Live Simulation — NOT cached (interactive, user expects fresh result)
// ---------------------------------------------------------------------------

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
