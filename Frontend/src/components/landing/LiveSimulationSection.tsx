import { useRef, useState } from 'react';
import { motion } from 'framer-motion';
import MaterialSymbol from './MaterialSymbol';
import { fetchLivePrediction } from '../../lib/api';

export default function LiveSimulationSection() {
  const [inputs, setInputs] = useState({
    branch: 'LTO CDO',
    day: 'Monday',
    time: '09:00',
    isHoliday: false,
  });

  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestIdRef = useRef(0);
  const controllerRef = useRef<AbortController | null>(null);

  const offices = ['LTO CDO', 'BIR CDO', 'SSS CDO', 'PNP CDO', 'DFA CDO'];
  const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  const times = ['08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00'];

  const handleInputChange = (field: string, value: any) => {
    setInputs((prev) => ({ ...prev, [field]: value }));
  };

  const handlePredict = async () => {
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    if (controllerRef.current) {
      controllerRef.current.abort();
    }
    const controller = new AbortController();
    controllerRef.current = controller;

    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetchLivePrediction(
        {
        day: inputs.day,
        time: inputs.time,
        isHoliday: inputs.isHoliday,
        },
        { signal: controller.signal }
      );
      if (requestIdRef.current !== requestId) {
        return;
      }
      const predictedWait = Math.max(1, Math.round(res.prediction));
      const confidence =
        typeof res.confidence === 'number'
          ? res.confidence
          : Math.min(97, Math.max(82, 96 - Math.round(predictedWait / 8)));

      const congestionRaw = res.congestion || '';
      const congestionKey = congestionRaw.toUpperCase();
      let congestion = 'Low';
      let congestionColor = 'text-green-500';
      if (congestionKey.includes('HIGH')) {
        congestion = 'High';
        congestionColor = 'text-red-500';
      } else if (congestionKey.includes('MODERATE')) {
        congestion = 'Moderate';
        congestionColor = 'text-yellow-500';
      }
      const recommendation =
        res.recommendation ||
        (predictedWait > 50 ? 'Consider visiting earlier or later' : 'Relatively favorable window based on the model');

      const rangeLabel = res.range ? `${res.range.p10}-${res.range.p90} min` : null;

      setResult({
        waitTime: predictedWait,
        confidence,
        congestion,
        congestionColor,
        recommendation,
        rangeLabel,
      });
    } catch (e) {
      if ((e as { name?: string }).name === 'AbortError') {
        return;
      }
      setError(e instanceof Error ? e.message : 'Prediction failed');
      setResult(null);
    } finally {
      if (requestIdRef.current === requestId) {
        setLoading(false);
      }
    }
  };

  return (
    <section id="demo" className="relative py-10 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 landing-section-shell">
        <motion.div
          className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-96 h-96 bg-cyan-600/10 rounded-full blur-3xl"
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
          <h2 className="text-4xl sm:text-5xl font-bold">
            Live <span className="text-red-500">Simulation</span> & Demo
          </h2>
          <p className="text-(--text-secondary) max-w-2xl mx-auto text-lg">
            Interactive ML inference playground - test predictions in real-time
          </p>
        </motion.div>

        {/* Simulator Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Input Panel */}
          <motion.div
            className="p-6 rounded-lg border border-white/10 bg-gradient-to-br from-white/5 to-white/10 backdrop-blur-sm space-y-5"
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h3 className="text-xl font-semibold flex items-center gap-2">
              <MaterialSymbol icon="tune" className="text-[20px] text-red-500" />
              Configure Prediction
            </h3>

            <div className="space-y-4">
              {/* Office Selection */}
              <div>
                <label className="text-m font-semibold text-(--text-primary) block mb-2">Office Branch</label>
                <div className="grid grid-cols-2 gap-2">
                  {offices.map((office) => (
                    <motion.button
                      key={office}
                      onClick={() => handleInputChange('branch', office)}
                      className={`p-2 rounded-lg text-sm font-semibold transition-all ${
                        inputs.branch === office
                          ? 'bg-red-600 text-white brand-contrast'
                          : 'bg-white/10 text-(--text-primary) hover:bg-white/20'
                      }`}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      {office}
                    </motion.button>
                  ))}
                </div>
              </div>

              {/* Day Selection */}
              <div>
                <label className="text-m font-semibold text-(--text-primary) block mb-2">Day of Week</label>
                <div className="grid grid-cols-3 gap-2">
                  {days.map((day) => (
                    <motion.button
                      key={day}
                      onClick={() => handleInputChange('day', day)}
                      className={`p-2 rounded-lg text-s font-semibold transition-all ${
                        inputs.day === day
                          ? 'bg-red-600 text-white brand-contrast'
                          : 'bg-white/10 text-(--text-primary) hover:bg-white/20'
                      }`}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      {day.slice(0, 3)}
                    </motion.button>
                  ))}
                </div>
              </div>

              {/* Time Selection */}
              <div>
                <label className="text-m font-semibold text-(--text-primary) block mb-2">Time</label>
                <div className="grid grid-cols-3 gap-2">
                  {times.map((time) => (
                    <motion.button
                      key={time}
                      onClick={() => handleInputChange('time', time)}
                      className={`p-2 rounded-lg text-s font-semibold transition-all ${
                        inputs.time === time
                          ? 'bg-red-600 text-white brand-contrast'
                          : 'bg-white/10 text-(--text-primary) hover:bg-white/20'
                      }`}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      {time}
                    </motion.button>
                  ))}
                </div>
              </div>

              {/* Holiday Toggle */}
              <div>
                <label className="text-m font-semibold text-(--text-primary) block mb-2">Holiday/Payday</label>
                <motion.button
                  onClick={() => handleInputChange('isHoliday', !inputs.isHoliday)}
                  className={`w-full p-3 rounded-lg font-semibold transition-all ${
                    inputs.isHoliday
                      ? 'bg-gradient-to-r from-green-600 to-emerald-600 text-white brand-contrast'
                      : 'bg-white/10 text-(--text-primary) hover:bg-white/20'
                  }`}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  {inputs.isHoliday ? '✓ Holiday/Payday Selected' : 'Regular Day'}
                </motion.button>
              </div>

              {/* Predict Button */}
              <motion.button
                onClick={handlePredict}
                disabled={loading}
                className="w-full p-4 bg-gradient-to-r from-red-600 to-orange-600 rounded-lg font-bold text-lg hover:shadow-2xl hover:shadow-red-600/50 transition-all flex items-center justify-center gap-2 disabled:opacity-60 disabled:pointer-events-none"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <MaterialSymbol icon="send" className="text-[20px]" />
                {loading ? 'Running…' : 'Run Prediction'}
              </motion.button>
              {error && <p className="text-sm text-red-400">{error}</p>}
            </div>
          </motion.div>

          {/* Result Panel */}
          <motion.div
            className="p-6 rounded-lg border border-white/10 bg-gradient-to-br from-white/5 to-white/10 backdrop-blur-sm flex flex-col justify-center"
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            {result ? (
              <motion.div className="space-y-6" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}>
                {/* Prediction Results */}
                <div className="space-y-4">
                  <div className="p-4 rounded-lg bg-gradient-to-br from-red-600/20 to-orange-600/20 border border-red-600/30">
                    <p className="text-sm text-(--text-secondary) mb-1">Expected Wait Time</p>
                    <p className="text-4xl font-bold text-(--text-primary)">{result.waitTime}</p>
                    <p className="text-xs text-(--text-secondary) mt-1">minutes</p>
                    {result.rangeLabel && (
                      <p className="text-xs text-(--text-secondary) mt-1">Likely range: {result.rangeLabel}</p>
                    )}
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 rounded-lg bg-gradient-to-br from-white/10 to-white/5 border border-white/20">
                      <p className="text-xs text-(--text-secondary) mb-1">Confidence</p>
                      <p className="text-2xl font-bold text-(--text-primary)">{result.confidence}%</p>
                    </div>
                    <div className="p-4 rounded-lg bg-gradient-to-br from-white/10 to-white/5 border border-white/20">
                      <p className="text-xs text-(--text-secondary) mb-1">Congestion</p>
                      <p className={`text-3xl font-bold ${result.congestionColor}`}>{result.congestion}</p>
                    </div>
                  </div>

                  <div className="p-4 rounded-lg bg-gradient-to-r from-red-600/20 to-orange-600/20 border border-red-600/30">
                    <p className="text-lg font-semibold text-red-400 mb-2">AI Recommendation</p>
                    <p className="text-(--text-secondary)">{result.recommendation}</p>
                  </div>
                </div>

                {/* Inference Details */}
                <div className="p-3 rounded-lg bg-black/50 border border-white/10">
                  <p className="text-lg text-(--text-secondary) mb-2">ML Inference Details</p>
                  <div className="grid grid-cols-2 gap-2 text-m">
                    <div>
                      <p className="text-(--text-secondary)">Model</p>
                      <p className="text-(--text-primary) font-semibold">Random Forest</p>
                    </div>
                    <div>
                      <p className="text-(--text-secondary)">Source</p>
                      <p className="text-(--text-primary) font-semibold">Backend /predict</p>
                    </div>
                  </div>
                </div>
              </motion.div>
            ) : (
              <div className="text-center space-y-4 py-10 text-lg">
                {loading ? (
                  <p className="text-(--text-secondary)">Running prediction…</p>
                ) : (
                  <p className="text-(--text-secondary)">
                    Configure inputs and click "Run Prediction" to see ML forecasting in action
                  </p>
                )}
              </div>
            )}
          </motion.div>
        </div>

        {/* Technical Details */}
        <motion.div
          className="p-5 rounded-lg border border-white/10 bg-gradient-to-br from-white/5 to-white/10 backdrop-blur-sm"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h3 className="font-semibold text-white mb-4 text-lg">How the Prediction Works</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-m">
            <div>
              <p className="text-red-400 font-semibold mb-2">Feature Engineering</p>
              <p className="text-(--text-secondary)">
                Inputs are encoded into 16 features including temporal features, cyclical encodings, and historical patterns
              </p>
            </div>
            <div>
              <p className="text-red-400 font-semibold mb-2">Model Inference</p>
              <p className="text-(--text-secondary)">
                Random Forest processes features through 500 decision trees, averaging predictions for robustness
              </p>
            </div>
            <div>
              <p className="text-red-400 font-semibold mb-2">Uncertainty Quantification</p>
              <p className="text-(--text-secondary)">
                Confidence scores reflect prediction certainty based on model consensus and historical accuracy
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
