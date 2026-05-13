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

/** Demo dates aligned with training data (2026) so day_of_week matches `date`. */
const DEMO_DATE_BY_DAY: Record<string, string> = {
  Monday: '2026-05-11',
  Tuesday: '2026-05-12',
  Wednesday: '2026-05-13',
  Thursday: '2026-05-14',
  Friday: '2026-05-15',
  Saturday: '2026-05-16',
};

const HOLIDAY_DEMO_DATE = '2026-01-01';

export type LivePredictInput = {
  day: string;
  time: string;
  queueLength?: number;
  isHoliday?: boolean;
};

export async function fetchLivePrediction(input: LivePredictInput) {
  const hour = parseInt(input.time.split(':')[0], 10);
  const date = input.isHoliday ? HOLIDAY_DEMO_DATE : DEMO_DATE_BY_DAY[input.day] || '2026-05-13';
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
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error((err as { error?: string }).error || 'Prediction request failed');
  }
  return response.json() as Promise<{
    success: boolean;
    prediction: number;
    unit: string;
    timestamp: string;
  }>;
}
