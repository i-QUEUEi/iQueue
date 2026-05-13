import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { fetchModelPerformance } from '../../lib/api';

export default function ModelPerformanceSection() {
  const [comparisonData, setComparisonData] = useState<any[]>([]);
  const [performanceMetrics, setPerformanceMetrics] = useState<any[]>([]);
  const [chartData, setChartData] = useState<any[]>([]);

  useEffect(() => {
    const loadData = async () => {
      try {
        const data = await fetchModelPerformance();
        setComparisonData(data.comparisonData);
        setPerformanceMetrics(data.performanceMetrics);
        setChartData(data.chartData);
      } catch (error) {
        console.error('Error loading model performance:', error);
      }
    };

    loadData();
  }, []);

  const chartTextColor = 'var(--text-secondary)';
  const chartBorderColor = 'var(--border-subtle)';
  const chartTooltipBg = 'var(--popover)';
  const chartTooltipText = 'var(--popover-foreground)';
  const colors = ['#EF4444', '#F97316', '#3B82F6'];

  return (
    <section id="performance" className="relative py-12 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 landing-section-shell">
        <motion.div
          className="absolute -bottom-40 right-0 w-96 h-96 bg-orange-600/10 rounded-full blur-3xl"
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
            Model <span className="text-red-500">Performance</span> Comparison
          </h2>
          <p className="text-(--text-secondary) max-w-2xl mx-auto text-lg">
            Rigorous evaluation across random split, chronological split, and cross-validation ensures robust production performance
          </p>
        </motion.div>

        {/* Performance Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-10">
          {performanceMetrics.map((metric, idx) => (
            <motion.div
              key={idx}
              className={`relative rounded-lg border border-white/10 bg-linear-to-br ${metric.color}/10 p-6 overflow-hidden group hover:border-white/20 transition-all`}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              viewport={{ once: true }}
              whileHover={{ scale: 1.02 }}
            >
              <div className="absolute inset-0 bg-linear-to-r from-white/0 via-white/5 to-white/0 opacity-0 group-hover:opacity-100 transition-opacity" />

              <div className="relative space-y-4">
                <div>
                  <p className="text-lg text-(--text-primary)">{metric.model}</p>
                  <p className="text-m text-(--text-primary) mt-1">{metric.metricType}</p>
                </div>

                <div className="space-y-2">
                  <div className="flex items-baseline gap-2">
                    <span className="text-4xl font-bold text-(--text-primary)">{metric.mae}</span>
                    <span className="text-(--text-primary)">{metric.maeUnit} MAE</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-lg text-(--text-primary)">Accuracy</span>
                    <span className="font-semibold text-primary">{metric.accuracy}</span>
                  </div>
                </div>

                <div className="pt-2 border-t border-white/10">
                  <p className="text-m text-(--text-primary) italic">{metric.status}</p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-10">
          {/* MAE Comparison */}
          <motion.div
            className="p-5 rounded-lg border border-white/10 bg-linear-to-br from-white/5 to-white/10 backdrop-blur-sm"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h3 className="text-lg font-semibold mb-4 text-(--text-primary)">Mean Absolute Error (MAE)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={comparisonData}>
                <CartesianGrid strokeDasharray="3 3" stroke={chartBorderColor} />
                <XAxis dataKey="model" stroke={chartTextColor} tick={{ fill: chartTextColor }} />
                <YAxis stroke={chartTextColor} tick={{ fill: chartTextColor }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: chartTooltipBg,
                    color: chartTooltipText,
                    border: `1px solid ${chartBorderColor}`,
                  }}
                />
                <Bar dataKey="mae" radius={[8, 8, 0, 0]}>
                  {comparisonData.map((_entry, index) => (
                    <Cell key={`cell-${index}`} fill={colors[index]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="text-s text-(--text-tertiary) mt-4">Lower is better. Random Forest achieves 8 mins average error.</p>
          </motion.div>

          {/* R² Score Comparison */}
          <motion.div
            className="p-5 rounded-lg border border-white/10 bg-linear-to-br from-white/5 to-white/10 backdrop-blur-sm"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1, duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h3 className="text-lg font-semibold mb-4 text-(--text-primary)">R² Score (Model Quality)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={comparisonData}>
                <CartesianGrid strokeDasharray="3 3" stroke={chartBorderColor} />
                <XAxis dataKey="model" stroke={chartTextColor} tick={{ fill: chartTextColor }} />
                <YAxis stroke={chartTextColor} tick={{ fill: chartTextColor }} domain={[0, 1]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: chartTooltipBg,
                    color: chartTooltipText,
                    border: `1px solid ${chartBorderColor}`,
                  }}
                />
                <Bar dataKey="r2" radius={[8, 8, 0, 0]}>
                  {comparisonData.map((_entry, index) => (
                    <Cell key={`cell-${index}`} fill={colors[index]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="text-s text-(--text-tertiary) mt-4">Higher is better. Random Forest explains 89% of variance.</p>
          </motion.div>
        </div>

        {/* Prediction Accuracy Line Chart */}
        <motion.div
          className="p-5 rounded-lg border border-white/10 bg-linear-to-br from-white/5 to-white/10 backdrop-blur-sm"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h3 className="text-lg font-semibold mb-4 text-(--text-primary)">Waiting Time Prediction Accuracy</h3>
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke={chartBorderColor} />
              <XAxis dataKey="hour" stroke={chartTextColor} tick={{ fill: chartTextColor }} />
              <YAxis stroke={chartTextColor} tick={{ fill: chartTextColor }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: chartTooltipBg,
                  color: chartTooltipText,
                  border: `1px solid ${chartBorderColor}`,
                }}
              />
              <Legend wrapperStyle={{ color: chartTextColor }} />
              <Line
                type="monotone"
                dataKey="actual"
                stroke="#FFFFFF"
                strokeWidth={2}
                dot={{ fill: '#FFFFFF', r: 4 }}
                name="Actual Wait Time"
              />
              <Line
                type="monotone"
                dataKey="predicted"
                stroke="#EF4444"
                strokeWidth={2}
                dot={{ fill: '#EF4444', r: 3 }}
                name="Random Forest Prediction"
              />
              <Line
                type="monotone"
                dataKey="gb"
                stroke="#F97316"
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={{ fill: '#F97316', r: 3 }}
                name="Gradient Boosting"
              />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-s text-(--text-tertiary) mt-4">Random Forest predictions closely track actual waiting times throughout business hours.</p>
        </motion.div>

        {/* Key Insights */}
        <motion.div
          className="mt-12 grid grid-cols-1 md:grid-cols-2 gap-6"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.8 }}
          viewport={{ once: true }}
        >
          {[
            {
              title: 'Best Performer',
              content: 'Random Forest achieves 8 mins average error with 89% R² score on unseen test data',
            },
            {
              title: 'Robust Evaluation',
              content: 'Models evaluated across 3 methods: random split, chronological split, and 5-fold cross-validation',
            },
          ].map((insight, idx) => (
            <motion.div
              key={idx}
              className="p-4 rounded-lg bg-linear-to-r from-red-600/10 to-orange-600/10 border border-red-600/20"
              whileHover={{ scale: 1.02 }}
            >
              <h4 className="font-semibold text-(--text-primary) mb-2">{insight.title}</h4>
              <p className="text-(--text-secondary) text-m">{insight.content}</p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
