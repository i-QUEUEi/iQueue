import { motion } from 'framer-motion';
import MaterialSymbol from './MaterialSymbol';

export default function SystemReliabilitySection() {
  const probabilities = [
    { label: 'Operational', value: 74, icon: 'check_circle', color: 'from-green-600', trend: '+2%' },
    { label: 'Slow', value: 21, icon: 'warning', color: 'from-yellow-600', trend: '-1%' },
    { label: 'Down', value: 5, icon: 'cancel', color: 'from-red-600', trend: '-1%' },
  ];

  const metricsBreakdown = [
    { title: 'System Uptime', value: '99.2%', desc: 'Last 90 days' },
    { title: 'Average Response', value: '1.2s', desc: 'API latency' },
    { title: 'Peak Capacity', value: '2.5K/min', desc: 'Transactions' },
    { title: 'Data Freshness', value: '< 5 min', desc: 'Queue updates' },
  ];

  return (
    <section className="relative py-12 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 landing-section-shell">
        <motion.div
          className="absolute top-40 -right-40 w-96 h-96 bg-green-600/10 rounded-full blur-3xl"
          animate={{ scale: [1, 1.1, 1], opacity: [0.5, 0.8, 0.5] }}
          transition={{ duration: 6, repeat: Infinity }}
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
            System <span className="text-red-500">Reliability</span> Analytics
          </h2>
          <p className="text-(--text-secondary) max-w-2xl mx-auto text-lg">
            Predictive system status classification for operational intelligence
          </p>
        </motion.div>

        {/* Probability Indicators */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
          {probabilities.map((prob, idx) => {
            return (
              <motion.div
                key={idx}
                className={`relative p-6 rounded-lg border border-white/10 bg-linear-to-br ${prob.color}/10 backdrop-blur-sm`}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
                viewport={{ once: true }}
                whileHover={{ scale: 1.05 }}
              >
                {/* Side icon */}
                <div className="absolute top-4 right-4 flex items-center justify-center w-10 h-10 rounded-full bg-white/10 border border-white/10">
                  <MaterialSymbol icon={prob.icon} className="text-[20px] text-(--text-primary)" />
                </div>

                {/* Circular Progress */}
                <div className="relative w-32 h-32 mx-auto mb-4">
                  <svg className="w-full h-full" viewBox="0 0 120 120">
                    {/* Background circle */}
                    <circle cx="60" cy="60" r="54" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="8" />

                    {/* Progress circle */}
                    <motion.circle
                      cx="60"
                      cy="60"
                      r="54"
                      fill="none"
                      stroke="url(#gradient)"
                      strokeWidth="8"
                      strokeDasharray={`${(prob.value / 100) * 339} 339`}
                      strokeLinecap="round"
                      transform="rotate(-90 60 60)"
                      initial={{ strokeDasharray: '0 339' }}
                      whileInView={{ strokeDasharray: `${(prob.value / 100) * 339} 339` }}
                      transition={{ duration: 1, delay: idx * 0.1 }}
                      viewport={{ once: true }}
                    />

                    <defs>
                      <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                        {prob.color === 'from-green-600' && (
                          <>
                            <stop offset="0%" stopColor="#10B981" />
                            <stop offset="100%" stopColor="#059669" />
                          </>
                        )}
                        {prob.color === 'from-yellow-600' && (
                          <>
                            <stop offset="0%" stopColor="#FBBF24" />
                            <stop offset="100%" stopColor="#F59E0B" />
                          </>
                        )}
                        {prob.color === 'from-red-600' && (
                          <>
                            <stop offset="0%" stopColor="#EF4444" />
                            <stop offset="100%" stopColor="#DC2626" />
                          </>
                        )}
                      </linearGradient>
                    </defs>

                    {/* Center text */}
                    <text x="60" y="75" textAnchor="middle" className="text-3xl font-bold fill-(--text-primary)">
                      {prob.value}%
                    </text>
                  </svg>
                </div>

                <div className="text-center space-y-2">
                  <h3 className="text-lg font-semibold text-(--text-primary)">{prob.label}</h3>
                  <motion.p
                    className="text-sm font-semibold text-(--text-secondary)"
                    initial={{ opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    transition={{ delay: idx * 0.1 + 0.5 }}
                    viewport={{ once: true }}
                  >
                    {prob.trend}
                  </motion.p>
                </div>
              </motion.div>
            );
          })}
        </div>

        {/* Metrics Grid */}
        <motion.div
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-10"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.8 }}
          viewport={{ once: true }}
        >
          {metricsBreakdown.map((metric, idx) => (
            <motion.div
              key={idx}
              className="p-4 rounded-lg bg-linear-to-br from-white/10 to-white/5 border border-white/10 hover:border-red-600/30 transition-all"
              whileHover={{ scale: 1.05 }}
            >
              <p className="text-xs text-(--text-secondary) mb-2">{metric.title}</p>
              <p className="text-2xl font-bold text-(--text-primary) mb-1">{metric.value}</p>
              <p className="text-xs text-(--text-secondary)">{metric.desc}</p>
            </motion.div>
          ))}
        </motion.div>

        {/* Reliability Note */}
        <motion.div
          className="p-5 rounded-lg bg-linear-to-r from-green-600/20 to-emerald-600/20 border border-green-600/30"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4, duration: 0.8 }}
          viewport={{ once: true }}
        >
          <div className="flex items-start gap-4">
            <MaterialSymbol icon="check_circle" className="text-[24px] text-green-500 shrink-0 mt-1" />
            <div className="space-y-2">
              <h4 className="font-semibold text-(--text-primary) text-lg">Reliability Classification</h4>
              <p className="text-(--text-secondary)">
                The system uses Random Forest classification to predict office status as Operational, Slow, or Down.
                This meta-model helps stakeholders understand not just queue length, but overall system health.
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
