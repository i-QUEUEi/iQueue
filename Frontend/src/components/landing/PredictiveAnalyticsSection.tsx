import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import MaterialSymbol from './MaterialSymbol';
import { fetchPredictiveAnalytics } from '../../lib/api';

export default function PredictiveAnalyticsSection() {
  const [selectedTime, setSelectedTime] = useState('morning');
  const [predictions, setPredictions] = useState<any>({});
  const [timeSlots, setTimeSlots] = useState<any[]>([]);
  const [systemReliability, setSystemReliability] = useState<any>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      try {
        const data = await fetchPredictiveAnalytics();
        setPredictions(data.predictions);
        setTimeSlots(data.timeSlots);
        setSystemReliability(data.systemReliability);
      } catch (error) {
        console.error('Error loading predictive analytics:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  const current = predictions[selectedTime as keyof typeof predictions];

  if (loading || !current) {
    return (
      <section id="predictions" className="relative py-10 overflow-hidden">
        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <p className="text-(--text-secondary) text-center">Loading predictions...</p>
        </div>
      </section>
    );
  }

  const GaugeIndicator = ({ value, label }: { value: number; label: string }) => (
    <div className="flex flex-col items-center">
      <div className="relative w-32 h-32 mb-4">
        <svg className="w-full h-full" viewBox="0 0 120 120">
          <circle cx="60" cy="60" r="50" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="8" />
          <circle
            cx="60"
            cy="60"
            r="50"
            fill="none"
            stroke="url(#gauge-gradient)"
            strokeWidth="8"
            strokeDasharray={`${(value / 100) * 314} 314`}
            strokeLinecap="round"
            transform="rotate(-90 60 60)"
          />
          <defs>
            <linearGradient id="gauge-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#EF4444" />
              <stop offset="100%" stopColor="#F97316" />
            </linearGradient>
          </defs>
          <text x="60" y="70" textAnchor="middle" className="text-2xl font-bold fill-[var(--text-primary)]">
            {value}%
          </text>
        </svg>
      </div>
      <p className="text-sm text-(--text-secondary) text-center">{label}</p>
    </div>
  );

  return (
    <section id="predictions" className="relative py-10 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 landing-section-shell">
        <motion.div
          className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-teal-600/10 rounded-full blur-3xl"
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: 8, repeat: Infinity }}
        />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          className="space-y-3 mb-10 text-center"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h2 className="text-5xl sm:text-6xl font-bold">
            Predictive <span className="text-red-500">Analytics</span>
          </h2>
          <p className="text-lg text-(--text-secondary) max-w-2xl mx-auto">
            Real-time AI forecasting for optimal visit planning
          </p>
        </motion.div>

        {/* Time Slot Selector */}
        <motion.div
          className="flex flex-col sm:flex-row gap-3 justify-center mb-10"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          {timeSlots.map((slot) => (
            <motion.button
              key={slot.id}
              onClick={() => setSelectedTime(slot.id)}
              className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                selectedTime === slot.id
                  ? 'bg-gradient-to-r from-red-600 to-orange-600 text-white brand-contrast'
                  : 'border border-white/20 text-(--text-primary) hover:border-white/40'
              }`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <div className="text-lg font-semibold">{slot.label}</div>
              <div className="text-m text-(--text-primary)">{slot.time}</div>
            </motion.button>
          ))}
        </motion.div>

        {/* Prediction Cards */}
        <motion.div
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5 mb-10"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          {/* Waiting Time */}
          <motion.div
            className={`p-5 rounded-lg border border-white/10 bg-gradient-to-br ${current.color}/10 backdrop-blur-sm`}
            key={`${selectedTime}-wait`}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4 }}
          >
            <p className="text-m text-(--text-secondary) mb-2">Expected Wait Time</p>
            <p className="text-4xl font-bold text-white mb-2">{current.waitTime}</p>
            <p className="text-s text-(--text-secondary)">minutes</p>
          </motion.div>

          {/* Congestion Level */}
          <motion.div
            className={`p-5 rounded-lg border border-white/10 bg-gradient-to-br ${current.color}/10 backdrop-blur-sm`}
            key={`${selectedTime}-congestion`}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4, delay: 0.05 }}
          >
            <p className="text-m text-(--text-secondary) mb-2">Congestion Level</p>
            <div className="flex items-center gap-2 mb-2">
              <div
                className={`w-3 h-3 rounded-full ${
                  current.congestion === 'Low'
                    ? 'bg-green-500'
                    : current.congestion === 'High'
                    ? 'bg-red-500'
                    : 'bg-yellow-500'
                }`}
              />
              <p className="text-4xl font-bold text-white">{current.congestion}</p>
            </div>
          </motion.div>

          {/* Confidence Score */}
          <motion.div
            className={`p-5 rounded-lg border border-white/10 bg-gradient-to-br ${current.color}/10 backdrop-blur-sm`}
            key={`${selectedTime}-confidence`}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4, delay: 0.1 }}
          >
            <p className="text-m text-(--text-secondary) mb-4">Prediction Confidence</p>
            <div className="relative h-8 rounded-full bg-white/10 border border-white/20 overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-red-600 to-orange-600"
                initial={{ width: 0 }}
                animate={{ width: `${current.confidence}%` }}
                transition={{ duration: 0.8, delay: 0.2 }}
              />
            </div>
            <p className="text-lg font-bold text-white mt-2">{current.confidence}%</p>
          </motion.div>

          {/* Status */}
          <motion.div
            className={`p-5 rounded-lg border border-white/10 bg-gradient-to-br ${current.color}/10 backdrop-blur-sm`}
            key={`${selectedTime}-status`}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4, delay: 0.15 }}
          >
            <p className="text-m text-(--text-secondary) mb-2">Status</p>
            <div className="flex items-center gap-2">
              <MaterialSymbol icon="check_circle" className="text-[20px] text-green-500" />
              <p className="font-semibold text-white text-lg">Ready</p>
            </div>
            <p className="text-s text-(--text-secondary) mt-3">AI inference active</p>
          </motion.div>
        </motion.div>

        {/* AI Recommendation */}
        <motion.div
          className="p-5 rounded-lg bg-gradient-to-r from-red-600/20 to-orange-600/20 border border-red-600/30 flex items-start gap-4 mb-10"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.8 }}
          viewport={{ once: true }}
        >
          <MaterialSymbol icon="speed" className="text-[30px] text-red-500 flex-shrink-0 mt-1" />
          <div className="space-y-2">
            <h3 className="font-semibold text-red-400 text-xl">AI Recommendation Engine</h3>
            <p className="text-(--text-secondary) text-lg">
              {current.recommendation}
            </p>
          </div>
        </motion.div>

        {/* Prediction Breakdown */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <motion.div
            className="p-5 rounded-lg border border-white/10 bg-gradient-to-br from-white/5 to-white/10 backdrop-blur-sm"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1, duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h4 className="font-semibold text-white mb-4 text-lg">Operational</h4>
            <GaugeIndicator value={systemReliability.operational || 74} label="Probability" />
          </motion.div>

          <motion.div
            className="p-5 rounded-lg border border-white/10 bg-gradient-to-br from-white/5 to-white/10 backdrop-blur-sm"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h4 className="font-semibold text-white mb-4 text-lg">Slow</h4>
            <GaugeIndicator value={systemReliability.slow || 21} label="Probability" />
          </motion.div>

          <motion.div
            className="p-5 rounded-lg border border-white/10 bg-gradient-to-br from-white/5 to-white/10 backdrop-blur-sm"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h4 className="font-semibold text-white mb-4 text-lg">Down</h4>
            <GaugeIndicator value={systemReliability.down || 5} label="Probability" />
          </motion.div>
        </div>
      </div>
    </section>
  );
}
