import { useState, useEffect, Suspense, lazy } from 'react';
import HeroSection from '../components/landing/HeroSection';
import ProblemDatasetSection from '../components/landing/ProblemDatasetSection';
import Navigation from '../components/landing/Navigation';
import ThemedSuspenseLoader from '../components/landing/ThemedSuspenseLoader';

// Lazy-loaded heavy sections to reduce main bundle size
const MLModelsSection = lazy(() => import('../components/landing/MLModelsSection'));
const ModelPerformanceSection = lazy(() => import('../components/landing/ModelPerformanceSection'));
const FeatureImportanceSection = lazy(() => import('../components/landing/FeatureImportanceSection'));
const PredictiveAnalyticsSection = lazy(() => import('../components/landing/PredictiveAnalyticsSection'));
const HistoricalAnalyticsSection = lazy(() => import('../components/landing/HistoricalAnalyticsSection'));
const SystemReliabilitySection = lazy(() => import('../components/landing/SystemReliabilitySection'));
const LiveSimulationSection = lazy(() => import('../components/landing/LiveSimulationSection'));
const ImpactSection = lazy(() => import('../components/landing/ImpactSection'));
const WeeklyForecastSection = lazy(() => import('../components/landing/WeeklyForecastSection'));

type ThemeMode = 'dark' | 'light';

export default function LandingPage() {
  const [scrollProgress, setScrollProgress] = useState(0);
  const [theme, setTheme] = useState<ThemeMode>(() => {
    if (typeof window === 'undefined') {
      return 'dark';
    }

    const storedTheme = window.localStorage.getItem('iqueue-theme');
    return storedTheme === 'light' ? 'light' : 'dark';
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.classList.toggle('dark', theme === 'dark');
    window.localStorage.setItem('iqueue-theme', theme);
  }, [theme]);

  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.scrollY;
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;
      const scrolled = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
      setScrollProgress(scrolled);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="landing-shell min-h-screen overflow-hidden transition-colors duration-300">
      <Navigation
        scrollProgress={scrollProgress}
        theme={theme}
        onThemeToggle={() => setTheme((currentTheme) => (currentTheme === 'dark' ? 'light' : 'dark'))}
      />
      <HeroSection />
      <ProblemDatasetSection />
      <Suspense fallback={<ThemedSuspenseLoader label="Loading ML modules…" />}>
        <MLModelsSection />
      </Suspense>

      <Suspense fallback={<ThemedSuspenseLoader label="Loading performance charts…" />}>
        <ModelPerformanceSection />
      </Suspense>

      <Suspense fallback={<ThemedSuspenseLoader label="Loading feature insights…" />}>
        <FeatureImportanceSection />
      </Suspense>

      <Suspense fallback={<ThemedSuspenseLoader label="Loading predictive analytics…" />}>
        <PredictiveAnalyticsSection />
      </Suspense>

      <Suspense fallback={<ThemedSuspenseLoader label="Loading historical analytics…" />}>
        <HistoricalAnalyticsSection />
      </Suspense>

      <Suspense fallback={<ThemedSuspenseLoader label="Loading reliability tools…" />}>
        <SystemReliabilitySection />
      </Suspense>

      <Suspense fallback={<ThemedSuspenseLoader label="Loading simulation demo…" />}>
        <LiveSimulationSection />
      </Suspense>

      <Suspense fallback={<ThemedSuspenseLoader label="Loading weekly forecast…" />}>
        <WeeklyForecastSection />
      </Suspense>

      <Suspense fallback={<ThemedSuspenseLoader label="Finishing up…" />}>
        <ImpactSection />
      </Suspense>
    </div>
  );
}
