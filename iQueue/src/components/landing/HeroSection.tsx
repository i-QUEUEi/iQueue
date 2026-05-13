import { motion } from 'framer-motion';
import MaterialSymbol from './MaterialSymbol';
import iQueueWordmarkRed from '../../assets/iQueueWordmarkRed.png';

export default function HeroSection() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.8 },
    },
  };

  const stats = [
    { label: 'Prediction Accuracy', value: '87.4%' },
    { label: 'Avg. Error', value: '±8 mins' },
    { label: 'Crowdsourced Reports', value: '12,483' },
    { label: 'Offices Simulated', value: '15' },
  ];

  return (
    <section id="hero" className="relative min-h-screen pt-20 overflow-hidden">
      {/* Background Grid */}
      <div className="absolute inset-0 landing-section-shell">
        <div className="absolute inset-0 bg-grid-pattern opacity-[0.03]" />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-12">
        <motion.div
          className="space-y-6"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          {/* Two-column headline row */}
          <motion.div variants={itemVariants} className="flex flex-col lg:flex-row lg:items-start lg:gap-6">

            {/* Left — wordmark + description + features + CTAs */}
            <div className="flex-1 space-y-6">
              <div>
                <img
                  src={iQueueWordmarkRed}
                  alt="iQueue"
                  className="h-26 sm:h-26 w-auto mb-6"
                />
                <p className="text-xl sm:text-2xl text-(--text-secondary) max-w-2xl">
                  Predictive Queue Intelligence for{' '}
                  <span className="text-red-500 font-semibold">Philippine Government Offices</span>
                </p>
              </div>

              {/* Feature bullets */}
              <div className="space-y-3">
                <div className="flex items-center gap-3 text-(--text-primary)">
                  <div className="w-1 h-1 bg-red-500 rounded-full shrink-0" />
                  <span>Waiting time prediction powered by machine learning</span>
                </div>
                <div className="flex items-center gap-3 text-(--text-primary)">
                  <div className="w-1 h-1 bg-orange-500 rounded-full shrink-0" />
                  <span>Congestion forecasting and system reliability analysis</span>
                </div>
                <div className="flex items-center gap-3 text-(--text-primary)">
                  <div className="w-1 h-1 bg-teal-500 rounded-full shrink-0" />
                  <span>Real-time analytics dashboard for government offices</span>
                </div>
              </div>

              {/* CTA Buttons */}
              <div className="flex flex-col sm:flex-row gap-3">
                <motion.button
                  className="brand-contrast px-8 py-4 bg-gradient-to-r from-red-600 to-red-700 rounded-xl font-semibold text-lg flex items-center justify-center gap-2 hover:shadow-2xl hover:shadow-red-600/50 transition-all interactive-btn"
                  whileHover={{ scale: 1.05, boxShadow: '0 0 40px rgba(239, 68, 68, 0.6)' }}
                  whileTap={{ scale: 0.95 }}
                >
                  Explore ML Predictions
                  <MaterialSymbol icon="arrow_forward" className="text-[20px]" />
                </motion.button>
                <motion.button
                  className="px-8 py-4 border border-white/20 rounded-xl font-semibold text-lg hover:bg-white/5 transition-all interactive-btn"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  View Analytics
                </motion.button>
              </div>
            </div>

            {/* Right  */}
            <motion.div
              variants={itemVariants}
              className="lg:w-1/2 xl:w-1/2 h-56 text-center shrink-0 mt-6 lg:mt-12 grid grid-cols-2 gap-3"
            >
              {stats.map((stat, idx) => (
                <motion.div
                  key={idx}
                  className="space-y-1 p-4 rounded-xl bg-white/5 border border-white/10 backdrop-blur-sm transition-all interactive-card"
                  whileHover={{ scale: 1.04, backgroundColor: 'rgba(255,255,255,0.08)' }}
                >
                  <p className="text-2xl font-bold text-red-500">{stat.value}</p>
                  <p className="text-s text-(--text-secondary) leading-snug">{stat.label}</p>
                </motion.div>
              ))}
            </motion.div>
          </motion.div>
        </motion.div>

        {/* Hero Visualization */}
        <motion.div
          className="mt-12 relative"
          variants={itemVariants}
          initial="hidden"
          animate="visible"
        >
          <motion.div
            className="relative rounded-2xl border border-white/10 bg-gradient-to-b from-white/5 to-transparent p-6 overflow-hidden"
            animate={{ boxShadow: ['0 0 20px rgba(239, 68, 68, 0.1)', '0 0 40px rgba(239, 68, 68, 0.2)', '0 0 20px rgba(239, 68, 68, 0.1)'] }}
            transition={{ duration: 3, repeat: Infinity }}
          >
            {/* Dashboard Preview Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-3">
              <motion.div
                className="h-32 rounded-lg bg-gradient-to-br from-red-600/20 to-red-900/20 border border-red-600/30 p-4 flex flex-col justify-between"
                whileHover={{ scale: 1.05 }}
              >
                <p className="text-md font-semibold text-(--text-primary)">Queue Length</p>
                <div className="flex items-end gap-1 h-12">
                  {[40, 60, 45, 70, 50].map((h, i) => (
                    <motion.div
                      key={i}
                      className="flex-1 bg-gradient-to-t from-red-700 to-red-500 rounded-t opacity-70"
                      style={{ height: `${h}%` }}
                      animate={{ height: [`${h}%`, `${h + 10}%`, `${h}%`] }}
                      transition={{ duration: 2, delay: i * 0.2, repeat: Infinity }}
                    />
                  ))}
                </div>
              </motion.div>

              <motion.div
                className="h-32 rounded-lg bg-gradient-to-br from-orange-500/20 to-orange-900/20 border border-orange-600/30 p-4 flex flex-col justify-between"
                whileHover={{ scale: 1.05 }}
              >
                <p className="text-md font-semibold text-(--text-primary)">Wait Time (mins)</p>
                <p className="text-3xl font-bold text-orange-400">45</p>
                <p className="text-s text-(--text-secondary)">±8 min confidence</p>
              </motion.div>

              <motion.div
                className="h-32 rounded-lg bg-gradient-to-br from-teal-600/20 to-teal-900/20 border border-teal-600/30 p-4 flex flex-col justify-between"
                whileHover={{ scale: 1.05 }}
              >
                <p className="text-md font-semibold text-(--text-primary)">Prediction</p>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />
                  <span className="text-xl font-semibold text-teal-400">Operational</span>
                </div>
                <p className="text-s text-(--text-secondary)">87.4% accuracy</p>
              </motion.div>
            </div>

            {/* Animated Chart Preview */}
            <div className="h-48 rounded-lg bg-black/50 border border-white/10 p-4 flex items-end gap-1">
              {Array.from({ length: 15 }).map((_, i) => (
                <motion.div
                  key={i}
                  className="flex-1 bg-gradient-to-t from-red-600 via-orange-500 to-yellow-400 rounded-t opacity-80"
                  style={{ height: `${30 + Math.sin(i * 0.5) * 20}%` }}
                  animate={{ height: [`${30 + Math.sin(i * 0.5) * 20}%`, `${40 + Math.cos(i * 0.5) * 25}%`, `${30 + Math.sin(i * 0.5) * 20}%`] }}
                  transition={{ duration: 3, delay: i * 0.05, repeat: Infinity }}
                />
              ))}
            </div>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}