import { motion } from 'framer-motion';
import MaterialSymbol from './MaterialSymbol';

export default function ImpactSection() {
  const impacts = [
    {
      icon: 'groups',
      title: 'Reduce Wasted Travel Time',
      desc: 'Citizens can plan visits strategically using predictive intelligence',
      metric: '40-60%',
      metricLabel: 'time savings',
    },
    {
      icon: 'trending_up',
      title: 'Improve Service Transparency',
      desc: 'Data-driven insights reveal operational patterns and capacity issues',
      metric: '100%',
      metricLabel: 'visibility',
    },
    {
      icon: 'bar_chart',
      title: 'Support Data-Driven Governance',
      desc: 'LGUs use queue analytics for resource allocation and planning',
      metric: 'Real-time',
      metricLabel: 'analytics',
    },
    {
      icon: 'public',
      title: 'Reduce Physical Congestion',
      desc: 'Predictive scheduling distributes demand across business hours',
      metric: '25-35%',
      metricLabel: 'peak reduction',
    },
  ];

  const opportunities = [
    { title: 'LGU Integration', desc: 'Deploy dashboards in local government units' },
    { title: 'Performance Reports', desc: 'Generate benchmarks for office productivity' },
    { title: 'Analytics Subscriptions', desc: 'Premium insights for government stakeholders' },
    { title: 'API Access', desc: 'Third-party integrations for queue systems' },
  ];

  return (
    <section className="relative py-8 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 landing-section-shell">
        <motion.div
          className="absolute inset-0 bg-linear-to-t from-red-600/5 via-transparent to-transparent"
          animate={{ opacity: [0.3, 0.6, 0.3] }}
          transition={{ duration: 6, repeat: Infinity }}
        />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-white">
        {/* Header */}
        <motion.div
          className="space-y-3 mb-12 text-center"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h2 className="text-5xl sm:text-6xl font-bold text-white">
            Social <span className="text-red-500">Impact</span> & Startup Potential
          </h2>
          <p className="text-white max-w-2xl mx-auto text-lg">
            iQueue transforms Philippine government service delivery through AI-powered intelligence
          </p>
        </motion.div>

        {/* Impact Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
          {impacts.map((impact, idx) => {
            return (
              <motion.div
                key={idx}
                className="p-6 rounded-lg border border-white/10 bg-linear-to-br from-white/5 to-white/10 backdrop-blur-sm hover:border-red-600/30 transition-all group"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
                viewport={{ once: true }}
                whileHover={{ scale: 1.02 }}
              >
                <div className="flex items-start gap-4 mb-4">
                  <div className="p-3 rounded-lg bg-linear-to-br from-red-600/20 to-orange-600/20 group-hover:from-red-600/40 group-hover:to-orange-600/40 transition-all">
                    <MaterialSymbol icon={impact.icon} className="text-[24px] text-red-500" />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-white">{impact.title}</h3>
                    <p className="text-lg text-white mt-1">{impact.desc}</p>
                  </div>
                </div>
                <div className="pl-16 space-y-1">
                  <p className="text-3xl font-bold text-white">{impact.metric}</p>
                  <p className="text-m text-white">{impact.metricLabel}</p>
                </div>
              </motion.div>
            );
          })}
        </div>

        {/* Business Opportunities */}
        <motion.div
          className="mb-12"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4, duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h3 className="text-2xl font-semibold text-white mb-6 text-center">Revenue Opportunities</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {opportunities.map((opp, idx) => (
              <motion.div
                key={idx}
                className="p-5 rounded-lg border border-white/10 bg-linear-to-br from-red-700 to-red-500 backdrop-blur-sm hover:border-red-600/30 transition-all text-white"
                whileHover={{ scale: 1.05 }}
                initial={{ opacity: 0, y: 10 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
                viewport={{ once: true }}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-semibold text-white mb-1 text-lg">{opp.title}</h4>
                    <p className="text-m text-white/85">{opp.desc}</p>
                  </div>
                  <MaterialSymbol icon="arrow_forward" className="text-[20px] text-white shrink-0" />
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Vision Statement */}
        <motion.div
          className="p-6 rounded-lg border border-red-600/30 bg-linear-to-r from-red-700 via-red-600/30 to-red-500 relative overflow-hidden"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6, duration: 0.8 }}
          viewport={{ once: true }}
        >
          <div className="absolute inset-0 bg-linear-to-r from-transparent via-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

          <div className="relative space-y-4">
            <h3 className="text-2xl font-bold text-white">
              Building the Future of Philippine Government Services
            </h3>
            <p className="text-white text-lg leading-relaxed">
              iQueue represents a new paradigm for government service delivery: <span className="font-semibold text-white">predictive, transparent, and citizen-centric</span>.
              By applying machine learning to queue dynamics, we're not just improving wait times—we're democratizing access to
              public services through data-driven intelligence. This is modern governance powered by AI.
            </p>

            <div className="pt-4 flex items-center gap-2 text-white font-semibold">
              <span>Ready to transform queue intelligence?</span>
              <MaterialSymbol icon="arrow_forward" className="text-[20px]" />
            </div>
          </div>
        </motion.div>

        {/* Footer CTA */}
        <motion.div
          className="mt-12 text-center space-y-4"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8, duration: 0.8 }}
          viewport={{ once: true }}
        >
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <motion.button
              className="px-8 py-4 bg-linear-to-r from-red-600 to-orange-600 rounded-xl font-semibold text-lg text-white hover:shadow-2xl hover:shadow-red-600/50 transition-all"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Get Started
            </motion.button>
            <motion.button
              className="px-8 py-4 border border-white/20 rounded-lg font-semibold text-lg text-white hover:bg-white/5 transition-all"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Documentation
            </motion.button>
          </div>

          <div className="pt-8 border-t border-white/10">
            <p className="text-white text-sm">
              © 2026 iQueue. Predictive Queue Intelligence for Modern Governance.
            </p>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
