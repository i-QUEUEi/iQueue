/**
 * API utility functions for iQueue frontend
 * Fetches real data from the backend API instead of using hardcoded data
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

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
