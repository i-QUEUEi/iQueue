import { Fragment } from 'react';
import { motion } from 'framer-motion';
import MaterialSymbol from './MaterialSymbol';

export default function ProblemDatasetSection() {
  const pipelineSteps = [
    { label: 'Raw Data', icon: 'database' },
    { label: 'Missing Values', icon: 'filter_alt' },
    { label: 'Outliers', icon: 'trending_up' },
    { label: 'Encoding', icon: 'bar_chart' },
    { label: 'Normalization', icon: 'filter_alt' },
    { label: 'Ready', icon: 'database' },
  ];

  const datasetFeatures = [
    { name: 'Office Branch', color: 'red' },
    { name: 'Visit Timestamp', color: 'orange' },
    { name: 'Waiting Time', color: 'yellow' },
    { name: 'Queue Length', color: 'green' },
    { name: 'System Status', color: 'blue' },
    { name: 'Holiday Indicator', color: 'purple' },
  ];

  const colorStyles: Record<string, string> = {
    red: 'bg-gradient-to-br from-red-600/15 to-red-500/6 border-red-600/30 shadow-[0_8px_30px_rgba(239,68,68,0.10)]',
    orange: 'bg-gradient-to-br from-orange-600/12 to-orange-500/6 border-orange-600/30 shadow-[0_8px_30px_rgba(249,115,22,0.10)]',
    yellow: 'bg-gradient-to-br from-yellow-500/12 to-yellow-400/6 border-yellow-500/30 shadow-[0_8px_30px_rgba(234,179,8,0.10)]',
    green: 'bg-gradient-to-br from-green-600/12 to-green-500/6 border-green-600/30 shadow-[0_8px_30px_rgba(16,185,129,0.10)]',
    blue: 'bg-gradient-to-br from-blue-600/12 to-blue-500/6 border-blue-600/30 shadow-[0_8px_30px_rgba(59,130,246,0.10)]',
    purple: 'bg-gradient-to-br from-purple-600/12 to-purple-500/6 border-purple-600/30 shadow-[0_8px_30px_rgba(139,92,246,0.10)]',
  };

  const accentClasses: Record<string, string> = {
    red: 'bg-red-400/40',
    orange: 'bg-orange-400/36',
    yellow: 'bg-yellow-300/36',
    green: 'bg-green-400/36',
    blue: 'bg-blue-400/36',
    purple: 'bg-purple-400/36',
  };

  const stats = [
    { value: '25,000', label: 'Total Records' },
    { value: '12', label: 'Features' },
    { value: '1,320', label: 'Cleaned Values' },
    { value: '46 mins', label: 'Avg Queue Time' },
  ];

  return (
    <section id="dataset" className="relative py-12 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 landing-section-shell">
        <motion.div
          className="absolute left-1/2 transform -translate-x-1/2 w-96 h-96 bg-red-600/10 rounded-full blur-3xl"
          animate={{ scale: [1, 1.2, 1], opacity: [0.5, 0.8, 0.5] }}
          transition={{ duration: 6, repeat: Infinity }}
        />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-2">
        {/* Header */}
        <motion.div
          className="space-y-3 mb-10 text-center"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h2 className="text-5xl sm:text-6xl font-bold">
            Problem & <span className="text-red-500">Dataset</span>
          </h2>
          <p className="text-(--text-secondary) max-w-3xl mx-auto text-lg">
            Philippine government offices face critical queue management challenges. Our ML system learns from comprehensive queue data to predict waiting times and optimize service delivery.
          </p>
        </motion.div>

        {/* Problem Statement */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-10 mb-10">
          <motion.div
            className="space-y-6"
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h3 className="text-3xl font-bold">The Challenge</h3>
            <div className="space-y-4">
              <p className="text-(--text-secondary) text-lg">
                Queue congestion in Philippine government offices impacts millions of citizens annually. Without predictive intelligence:
              </p>
              <ul className="space-y-3">
                {['Unpredictable waiting times', 'Inefficient resource allocation', 'Poor citizen experience', 'No data-driven planning'].map((item, i) => (
                  <motion.li
                    key={i}
                    className="flex items-center gap-3 text-(--text-primary) text-lg"
                    initial={{ opacity: 0, x: -10 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.1 }}
                    viewport={{ once: true }}
                  >
                    <div className="w-2 h-2 bg-red-500 rounded-full" />
                    {item}
                  </motion.li>
                ))}
              </ul>
            </div>
          </motion.div>

          <motion.div
            className="grid grid-cols-2 gap-4"
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            {stats.map((stat, idx) => (
              <motion.div
                key={idx}
                className="p-6 rounded-lg bg-gradient-to-br from-white/10 to-white/5 border border-white/10 backdrop-blur-sm"
                whileHover={{ scale: 1.05, borderColor: 'rgba(239, 68, 68, 0.5)' }}
              >
                <p className="text-4xl font-bold text-red-500 mb-2">{stat.value}</p>
                <p className="text-m text-(--text-secondary)">{stat.label}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>

        {/* Data Features */}
          <motion.div
            className="mb-10"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h3 className="text-4xl font-semibold mb-6">Dataset Overview</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {datasetFeatures.map((feature, idx) => (
              <motion.div
                key={idx}
                className={`relative p-5 rounded-lg border backdrop-blur-sm ${colorStyles[feature.color]} transition-transform duration-300 ease-out hover:scale-105 hover:-translate-y-1`}
                initial={{ opacity: 0, y: 10 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.05 }}
                viewport={{ once: true }}
              >
                <span className={`absolute -top-3 right-6 w-20 h-12 rounded-full pointer-events-none blur-2xl opacity-60 ${accentClasses[feature.color]}`} />
                <div className="relative z-10 flex flex-col gap-1">
                  <p className="text-m font-semibold text-white leading-none drop-shadow-sm">{feature.name}</p>
                </div>
              </motion.div>
            ))}
          </div>
          {/* Section divider */}
          <div className="mt-8 flex justify-center">
            <div className="w-full max-w-6xl h-[1px] rounded-full bg-gradient-to-r from-transparent via-white/10 to-transparent opacity-60 blur-sm" />
          </div>
        </motion.div>

        {/* Data Pipeline */}
          <motion.div
            className="space-y-4"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h3 className="text-3xl font-semibold text-center">Data Preprocessing Pipeline</h3>
          
          {/* Desktop Pipeline */}
          <div className="hidden md:block overflow-x-auto">
            <div className="flex items-center justify-center gap-3 min-w-max pb-3 mx-auto">
              {pipelineSteps.map((step, idx) => {
                return (
                  <Fragment key={idx}>
                    <motion.div
                      className="flex flex-col items-center gap-2"
                      whileHover={{ scale: 1.1 }}
                      initial={{ opacity: 0, y: 20 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      transition={{ delay: idx * 0.1 }}
                      viewport={{ once: true }}
                    >
                      <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-red-600 to-orange-600 flex items-center justify-center shadow-lg shadow-red-600/20">
                        <MaterialSymbol icon={step.icon} className="text-[24px] brand-contrast" />
                      </div>
                      <p className="text-sm font-semibold whitespace-nowrap">{step.label}</p>
                    </motion.div>
                    {idx < pipelineSteps.length - 1 && (
                      <motion.div
                        className="w-12 h-1 bg-gradient-to-r from-red-600 to-orange-600 rounded-full"
                        animate={{ scaleX: [0, 1, 0] }}
                        transition={{ duration: 2, delay: idx * 0.2, repeat: Infinity }}
                      />
                    )}
                  </Fragment>
                );
              })}
            </div>
          </div>

          {/* Mobile Pipeline */}
          <div className="md:hidden space-y-2.5">
            {pipelineSteps.map((step, idx) => {
              return (
                <motion.div
                  key={idx}
                  className="flex items-center gap-4"
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  viewport={{ once: true }}
                >
                  <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-red-600 to-orange-600 flex items-center justify-center flex-shrink-0 shadow-lg shadow-red-600/20">
                    <MaterialSymbol icon={step.icon} className="text-[20px] brand-contrast" />
                  </div>
                  <p className="font-semibold">{step.label}</p>
                  {idx < pipelineSteps.length - 1 && (
                    <div className="flex-1 h-px bg-gradient-to-r from-red-600 to-transparent" />
                  )}
                </motion.div>
              );
            })}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
