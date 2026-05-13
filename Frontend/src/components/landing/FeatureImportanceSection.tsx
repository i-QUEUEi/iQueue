import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import MaterialSymbol from './MaterialSymbol';
import { fetchFeatureImportance } from '../../lib/api';

export default function FeatureImportanceSection() {
  const [importanceData, setImportanceData] = useState<any[]>([]);
  const [topInsights, setTopInsights] = useState<any[]>([]);

  useEffect(() => {
    const loadData = async () => {
      try {
        const data = await fetchFeatureImportance();
        setImportanceData(data.importanceData);
        setTopInsights(data.topInsights);
      } catch (error) {
        console.error('Error loading feature importance:', error);
      }
    };

    loadData();
  }, []);

  return (
    <section id="analytics" className="relative py-10 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 landing-section-shell">
        <motion.div
          className="absolute top-1/4 -left-40 w-96 h-96 bg-yellow-600/10 rounded-full blur-3xl"
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
            Feature <span className="text-red-500">Importance</span> Analytics
          </h2>
          <p className="text-lg max-w-2xl mx-auto">
            Understanding what factors influence queue waiting times using SHAP analysis and permutation importance
          </p>
        </motion.div>

        {/* Feature Importance Chart */}
        <motion.div
            className="p-5 rounded-lg border border-white/10 bg-linear-to-br from-white/5 to-white/10 backdrop-blur-sm mb-10"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-(--text-primary)">
            <MaterialSymbol icon="trending_up" className="text-[20px] text-red-500" />
            Feature Importance Ranking
          </h3>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart
              data={importanceData}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 200, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
              <XAxis type="number" stroke="var(--text-primary)" tick={{ fill: 'var(--text-primary)' }} domain={[0, 0.3]} />
              <YAxis dataKey="feature" type="category" stroke="var(--text-primary)" tick={{ fill: 'var(--text-primary)' }} width={180} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'var(--popover)',
                  color: 'var(--popover-foreground)',
                  border: '1px solid var(--border-subtle)',
                }}
                formatter={(value: unknown) => `${((value as number) * 100).toFixed(1)}%`}
              />
              <Bar dataKey="importance" radius={[0, 8, 8, 0]} fill="#EF4444" />
            </BarChart>
          </ResponsiveContainer>
          <p className="text-s text-(--text-tertiary) mt-4">
            Time of day is the strongest predictor, accounting for 28.5% of model importance.
          </p>
        </motion.div>

        {/* Insights Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-10">
          {topInsights.map((insight, idx) => (
            <motion.div
              key={idx}
              className="p-5 rounded-lg border border-white/10 bg-linear-to-br from-white/5 to-white/10 backdrop-blur-sm hover:border-white/20 transition-all group"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              viewport={{ once: true }}
              whileHover={{ scale: 1.02 }}
            >
              <div className="flex items-start gap-4">
                  <MaterialSymbol icon={insight.icon} className="text-[30px] text-red-500 shrink-0" />
                <div>
                  <h4 className="font-semibold text-lg mb-2">{insight.title}</h4>
                  <p className="text-(--text-secondary) text-m">{insight.description}</p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Key Finding */}
        <motion.div
            className="p-5 rounded-lg bg-linear-to-r from-red-600/20 to-orange-600/20 border border-red-600/30 flex items-start gap-4"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4, duration: 0.8 }}
          viewport={{ once: true }}
        >
          <MaterialSymbol icon="warning" className="text-[24px] text-red-500 shrink-0 mt-1" />
          <div className="space-y-2">
            <h4 className="font-semibold text-lg">Key Finding</h4>
            <p className="text-m">
              Peak congestion is most strongly influenced by <span className="font-semibold text-(--text-primary)">time of day</span> and <span className="font-semibold text-(--text-primary)">payday schedules</span>. 
              Citizens can expect significantly shorter waits during off-peak hours (early morning or late afternoon) and on non-payday weekdays.
            </p>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
