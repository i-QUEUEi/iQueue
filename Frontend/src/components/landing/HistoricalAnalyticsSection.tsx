import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { fetchHistoricalAnalytics } from '../../lib/api';

export default function HistoricalAnalyticsSection() {
  const [dailyData, setDailyData] = useState<any[]>([]);
  const [hourlyData, setHourlyData] = useState<any[]>([]);
  const [heatmapData, setHeatmapData] = useState<any[]>([]);
  const [insights, setInsights] = useState<any[]>([]);

  useEffect(() => {
    const loadData = async () => {
      try {
        const data = await fetchHistoricalAnalytics();
        setDailyData(data.dailyData);
        setHourlyData(data.hourlyData);
        setHeatmapData(data.heatmapData);
        setInsights(data.insights);
      } catch (error) {
        console.error('Error loading historical analytics:', error);
      }
    };

    loadData();
  }, []);

  const chartTextColor = 'var(--text-primary)';
  const chartBorderColor = 'var(--border-subtle)';
  const chartTooltipBg = 'var(--popover)';
  const chartTooltipText = 'var(--popover-foreground)';

  return (
    <section className="relative py-10 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 landing-section-shell">
        <motion.div
          className="absolute -top-40 right-1/4 w-96 h-96 bg-purple-600/10 rounded-full blur-3xl"
          animate={{ scale: [1, 1.15, 1] }}
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
            Historical Queue <span className="text-red-500">Analytics</span>
          </h2>
          <p className="text-(--text-secondary) max-w-2xl mx-auto text-lg">
            Temporal patterns learned from 90 days of queue simulation data
          </p>
        </motion.div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-10">
          {/* Daily Average */}
          <motion.div
            className="p-5 rounded-lg border border-white/10 bg-linear-to-br from-white/5 to-white/10 backdrop-blur-sm"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h3 className="text-lg font-semibold mb-4 text-(--text-primary)">Average Wait by Day</h3>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={dailyData}>
                <CartesianGrid strokeDasharray="3 3" stroke={chartBorderColor} />
                <XAxis dataKey="day" stroke={chartTextColor} tick={{ fill: chartTextColor }} />
                <YAxis stroke={chartTextColor} tick={{ fill: chartTextColor }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: chartTooltipBg,
                    color: chartTooltipText,
                    border: `1px solid ${chartBorderColor}`,
                  }}
                />
                <Bar dataKey="avgWait" fill="#EF4444" radius={[8, 8, 0, 0]} />
              </ComposedChart>
            </ResponsiveContainer>
            <div className="mt-4 space-y-2">
              {dailyData.map((d, i) => (
                <div key={i} className="flex justify-between items-center text-sm">
                  <span className="text-(--text-secondary)">{d.day}</span>
                  <span className="font-semibold text-(--text-primary)">{d.avgWait} mins</span>
                  {d.busiest && <span className="text-red-500">●</span>}
                </div>
              ))}
            </div>
          </motion.div>

          {/* Hourly Pattern */}
          <motion.div
            className="p-5 rounded-lg border border-white/10 bg-linear-to-br from-white/5 to-white/10 backdrop-blur-sm"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1, duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h3 className="text-lg font-semibold mb-4 text-(--text-primary)">Hourly Queue Patterns</h3>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={hourlyData}>
                <CartesianGrid strokeDasharray="3 3" stroke={chartBorderColor} />
                <XAxis dataKey="hour" stroke={chartTextColor} tick={{ fill: chartTextColor }} label={{ value: 'Hour of Day', position: 'bottom' }} />
                <YAxis stroke={chartTextColor} tick={{ fill: chartTextColor }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: chartTooltipBg,
                    color: chartTooltipText,
                    border: `1px solid ${chartBorderColor}`,
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="wait"
                  stroke="#F97316"
                  strokeWidth={3}
                  dot={{ fill: '#F97316', r: 4 }}
                  name="Avg Wait (mins)"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </motion.div>
        </div>

        {/* Heatmap */}
        <motion.div
          className="p-5 rounded-lg border border-white/10 bg-linear-to-br from-white/5 to-white/10 backdrop-blur-sm mb-10"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h3 className="text-lg font-semibold mb-4 text-(--text-primary)">Waiting Time Heatmap (Day × Period)</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {heatmapData.map((row, i) => (
              <div key={i} className="space-y-2">
                <p className="font-semibold text-(--text-primary)">{row.day}</p>
                <div className="space-y-2">
                  {[
                    { label: 'Morning', value: row.morning },
                    { label: 'Afternoon', value: row.afternoon },
                    { label: 'Evening', value: row.evening },
                  ].map((period, j) => (
                    <motion.div
                      key={j}
                      className="relative h-10 rounded-lg overflow-hidden bg-white/5 border border-white/10"
                      whileHover={{ scale: 1.02 }}
                    >
                      <motion.div
                        className="h-full bg-linear-to-r from-red-600 via-orange-500 to-yellow-400"
                        style={{ width: `${(period.value / 70) * 100}%` }}
                        initial={{ width: 0 }}
                        whileInView={{ width: `${(period.value / 70) * 100}%` }}
                        transition={{ duration: 0.8 }}
                        viewport={{ once: true }}
                      />
                      <div className="absolute inset-0 flex items-center justify-between px-2 text-s">
                        <span className="text-(--text-primary) font-semibold">{period.label}</span>
                        <span className="text-(--text-primary) font-semibold">{period.value}m</span>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Key Insights */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          {insights.map((insight, idx) => (
            <motion.div
              key={idx}
              className="p-4 rounded-lg bg-linear-to-br from-white/10 to-white/5 border border-white/10 hover:border-white/20 transition-all"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              viewport={{ once: true }}
              whileHover={{ scale: 1.05 }}
            >
              <h4 className="font-semibold text-xl mb-2">{insight.title}</h4>
              <p className="text-(--text-secondary) text-m mb-2">{insight.desc}</p>
              <p className="text-lg font-bold text-(--text-primary)">{insight.value}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
