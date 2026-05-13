import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import MaterialSymbol from './MaterialSymbol';

export default function MLModelsSection() {
  const [expandedModel, setExpandedModel] = useState<string | null>('random-forest');

  const models = [
    {
      id: 'linear-regression',
      name: 'Linear Regression',
      category: 'Baseline Model',
      purpose: 'Waiting-time prediction baseline',
      color: 'from-blue-600 to-blue-900',
      highlights: [
        'Simple and interpretable',
        'Identifies general trends',
        'Fast computation',
      ],
      bestFor: 'Establishing baseline performance',
      icon: 'trending_up',
    },
    {
      id: 'random-forest',
      name: 'Random Forest Regressor',
      category: 'Primary Model ⭐',
      purpose: 'Main waiting-time prediction',
      color: 'from-red-600 to-red-900',
      highlights: [
        'Handles nonlinear queue behavior',
        'Resistant to overfitting',
        'Excellent with tabular data',
        'Feature importance analysis',
      ],
      bestFor: 'Production waiting-time forecasting',
      inputs: ['Day', 'Hour', 'Branch', 'Congestion Level', 'Historical Averages'],
      icon: 'forest',
      featured: true,
    },
    {
      id: 'gradient-boosting',
      name: 'Gradient Boosting Regressor',
      category: 'Ensemble Model',
      purpose: 'Advanced waiting-time prediction',
      color: 'from-orange-600 to-orange-900',
      highlights: [
        'Captures residual structure',
        'Different inductive bias than bagging',
        'High predictive power',
      ],
      bestFor: 'Capturing complex queue patterns',
      icon: 'rocket_launch',
    },
  ];

  return (
    <section id="models" className="relative py-6 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 landing-section-shell">
        <motion.div
          className="absolute left-1/2 transform -translate-x-1/2 w-96 h-96 bg-red-600/5 rounded-full blur-3xl"
          animate={{ scale: [1, 1.3, 1] }}
          transition={{ duration: 8, repeat: Infinity }}
        />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          className="space-y-3 mb-12 text-center"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h2 className="text-5xl sm:text-6xl font-bold">
            Machine Learning <span className="text-red-500">Models</span>
          </h2>
          <p className="text-(--text-secondary) max-w-3xl text-xl mx-auto">
            Multi-model architecture combining interpretability and predictive power for queue forecasting
          </p>
        </motion.div>

        {/* Models Grid */}
        <div className="space-y-3">
          {models.map((model, idx) => (
            <motion.div
              key={model.id}
              className={`relative rounded-lg border border-white/10 bg-white/5 backdrop-blur-sm cursor-pointer overflow-hidden transition-all hover:bg-white/10 hover:border-white/20 ${model.featured ? 'ring-1 ring-red-600/30' : ''}`}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              viewport={{ once: true }}
              onClick={() => setExpandedModel(expandedModel === model.id ? null : model.id)}
            >
              {/* Featured Badge */}
              {model.featured && (
                <motion.div
                  className="absolute top-0 right-0 px-3 py-1 bg-red-600 text-white text-s font-semibold rounded-bl-lg brand-contrast"
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  ⭐ BEST PERFORMER
                </motion.div>
              )}

              {/* Header */}
              <div className="p-5 bg-white/5">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <MaterialSymbol icon={model.icon} className="text-[30px] text-red-500" />
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="text-2xl font-bold">{model.name}</h3>
                          {model.featured && <MaterialSymbol icon="bolt" className="text-[16px] text-red-500" />}
                        </div>
                        <p className="text-m font-semibold text-(--text-secondary)">{model.category}</p>
                      </div>
                    </div>
                    <p className="text-(--text-secondary) text-m mt-2">{model.purpose}</p>
                  </div>
                  <motion.div
                    animate={{ rotate: expandedModel === model.id ? 180 : 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <MaterialSymbol icon="expand_more" className="text-[24px] text-(--text-secondary)" />
                  </motion.div>
                </div>
              </div>

              {/* Expanded Content */}
              <AnimatePresence>
                {expandedModel === model.id && (
                  <motion.div
                    className={`px-5 pb-5 pt-4 border-t border-white/10 space-y-3 bg-gradient-to-r ${model.color}/12`}
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    {/* Why Used */}
                    <div>
                      <h4 className="text-xl font-bold text-white mb-2">Why Used</h4>
                      <ul className="space-y-2">
                        {model.highlights.map((highlight, i) => (
                          <li key={i} className="flex items-center gap-2 text-white/80 text-lg">
                            <div className="w-2 h-2 bg-white/80 rounded-full" />
                            {highlight}
                          </li>
                        ))}
                      </ul>
                    </div>

                    {/* Inputs */}
                    {model.inputs && (
                      <div>
                        <h4 className="text-lg font-semibold text-white mb-2">Model Inputs</h4>
                        <div className="flex flex-wrap gap-2">
                          {model.inputs.map((input, i) => (
                            <span key={i} className="px-2 py-1 bg-white/10 border border-white/15 rounded text-m text-white/85">
                              {input}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Best For */}
                    <div className="p-3 rounded-lg bg-white/10 border border-white/15">
                      <p className="text-lg text-white/100">
                        <span className="font-semibold text-white/100">Best For: </span>
                        {model.bestFor}
                      </p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          ))}
        </div>

        {/* Architecture Note */}
        <motion.div
          className="mt-8 p-5 rounded-lg bg-gradient-to-r from-white/5 to-white/10 border border-white/10 backdrop-blur-sm"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <div className="flex items-start gap-4">
            <MaterialSymbol icon="psychology" className="text-[24px] text-red-500 flex-shrink-0 mt-1" />
            <div className="space-y-2">
              <h4 className="font-semibold text-white text-2xl">ML Architecture Strategy</h4>
              <p className="text-(--text-secondary) text-lg">
                We evaluate all three models using robust evaluation: random split, chronological split, and k-fold cross-validation. The model with the lowest robust MAE (Mean Absolute Error) is selected for production deployment. This ensures reliability across different temporal patterns and data distributions.
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
