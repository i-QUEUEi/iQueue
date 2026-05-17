import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import MaterialSymbol from './MaterialSymbol';
import { fetchWeeklyForecast } from '../../lib/api';

type DayForecast = {
  date: string;
  dayName: string;
  shortDate: string;
  isHoliday: boolean;
  overall: number | null;
  congestion: string;
  bestTime: string | null;
  bestWait: number | null;
  bestP10: number | null;
  bestP90: number | null;
  worstTime: string | null;
  worstWait: number | null;
  hourly: Array<{ hour: string; wait: number; p10: number; p90: number }>;
};

type WeekForecast = {
  weekLabel: string;
  weekOf: string;
  days: DayForecast[];
};

const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const MONTHS = [
  'January','February','March','April','May','June',
  'July','August','September','October','November','December',
];

function getMondayOfWeek(date: Date): Date {
  const d = new Date(date);
  const day = d.getDay(); // 0=Sun
  const diff = day === 0 ? -6 : 1 - day;
  d.setDate(d.getDate() + diff);
  d.setHours(0, 0, 0, 0);
  return d;
}

function toYMD(d: Date): string {
  // Use local date components — NOT toISOString() which converts to UTC
  // and would shift e.g. May 11 midnight local (+08:00) → May 10 UTC
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

function congestionColor(c: string) {
  if (c === 'HIGH') return { bg: 'from-red-600/30 to-red-900/20', badge: 'bg-red-600', text: 'text-red-400', dot: 'bg-red-500' };
  if (c === 'MODERATE') return { bg: 'from-yellow-600/30 to-yellow-900/20', badge: 'bg-yellow-600', text: 'text-yellow-400', dot: 'bg-yellow-500' };
  if (c === 'CLOSED') return { bg: 'from-gray-700/30 to-gray-900/20', badge: 'bg-gray-600', text: 'text-gray-400', dot: 'bg-gray-500' };
  return { bg: 'from-green-600/30 to-green-900/20', badge: 'bg-green-600', text: 'text-green-400', dot: 'bg-green-500' };
}

function congestionLabel(c: string) {
  if (c === 'HIGH') return 'HIGH';
  if (c === 'MODERATE') return 'MODERATE';
  if (c === 'CLOSED') return 'CLOSED';
  return 'LOW';
}

export default function WeeklyForecastSection() {
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  const [calMonth, setCalMonth] = useState(today.getMonth());
  const [calYear, setCalYear] = useState(today.getFullYear());
  const [selectedMonday, setSelectedMonday] = useState<Date>(getMondayOfWeek(today));
  const [forecast, setForecast] = useState<WeekForecast | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadForecast = useCallback(async (monday: Date) => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchWeeklyForecast(toYMD(monday));
      setForecast(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load forecast');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadForecast(selectedMonday);
  }, [selectedMonday, loadForecast]);

  // Build calendar grid
  const firstDay = new Date(calYear, calMonth, 1);
  const lastDay = new Date(calYear, calMonth + 1, 0);
  const startOffset = firstDay.getDay(); // 0=Sun
  const totalCells = Math.ceil((startOffset + lastDay.getDate()) / 7) * 7;
  const cells: (Date | null)[] = [];
  for (let i = 0; i < totalCells; i++) {
    const dayNum = i - startOffset + 1;
    if (dayNum < 1 || dayNum > lastDay.getDate()) {
      cells.push(null);
    } else {
      cells.push(new Date(calYear, calMonth, dayNum));
    }
  }

  const isSameWeek = (d: Date | null) => {
    if (!d) return false;
    const m = getMondayOfWeek(d);
    return toYMD(m) === toYMD(selectedMonday);
  };

  const handleDayClick = (d: Date | null) => {
    if (!d) return;
    const mon = getMondayOfWeek(d);
    setSelectedMonday(mon);
    // Ensure calendar shows this week's month
    if (mon.getMonth() !== calMonth || mon.getFullYear() !== calYear) {
      setCalMonth(mon.getMonth());
      setCalYear(mon.getFullYear());
    }
  };

  const prevMonth = () => {
    if (calMonth === 0) { setCalMonth(11); setCalYear(y => y - 1); }
    else setCalMonth(m => m - 1);
  };
  const nextMonth = () => {
    if (calMonth === 11) { setCalMonth(0); setCalYear(y => y + 1); }
    else setCalMonth(m => m + 1);
  };
  const goToday = () => {
    const mon = getMondayOfWeek(today);
    setSelectedMonday(mon);
    setCalMonth(today.getMonth());
    setCalYear(today.getFullYear());
  };

  return (
    <section id="weekly-forecast" className="relative py-10 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 landing-section-shell">
        <motion.div
          className="absolute top-0 right-1/4 w-96 h-96 bg-indigo-600/10 rounded-full blur-3xl"
          animate={{ scale: [1, 1.15, 1] }}
          transition={{ duration: 9, repeat: Infinity }}
        />
        <motion.div
          className="absolute bottom-0 left-1/4 w-96 h-96 bg-purple-600/10 rounded-full blur-3xl"
          animate={{ scale: [1, 1.2, 1] }}
          transition={{ duration: 11, repeat: Infinity }}
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
            Weekly <span className="text-red-500">Forecast</span>
          </h2>
          <p className="text-(--text-secondary) max-w-2xl mx-auto text-lg">
            Pick any week on the calendar — the ML model forecasts Monday through Saturday
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Calendar */}
          <motion.div
            className="p-6 rounded-xl border border-white/10 bg-gradient-to-br from-white/5 to-white/10 backdrop-blur-sm"
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            {/* Month Nav */}
            <div className="flex items-center justify-between mb-5">
              <motion.button
                onClick={prevMonth}
                className="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-all"
                whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}
              >
                <MaterialSymbol icon="chevron_left" className="text-[20px] text-(--text-primary)" />
              </motion.button>
              <div className="text-center">
                <p className="font-bold text-white text-lg">{MONTHS[calMonth]} {calYear}</p>
                <button
                  onClick={goToday}
                  className="text-xs text-red-400 hover:text-red-300 transition-colors mt-0.5"
                >
                  Today
                </button>
              </div>
              <motion.button
                onClick={nextMonth}
                className="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-all"
                whileHover={{ scale: 1.1 }} whileTap={{ scale: 0.9 }}
              >
                <MaterialSymbol icon="chevron_right" className="text-[20px] text-(--text-primary)" />
              </motion.button>
            </div>

            {/* Day headers */}
            <div className="grid grid-cols-7 mb-2">
              {DAYS.map(d => (
                <div key={d} className="text-center text-xs font-semibold text-(--text-secondary) py-1">
                  {d}
                </div>
              ))}
            </div>

            {/* Calendar cells */}
            <div className="grid grid-cols-7 gap-y-1">
              {cells.map((cell, idx) => {
                const inSelectedWeek = isSameWeek(cell);
                const isToday = cell && toYMD(cell) === toYMD(today);
                return (
                  <motion.button
                    key={idx}
                    onClick={() => handleDayClick(cell)}
                    disabled={!cell}
                    className={`relative text-center py-1.5 rounded-lg text-sm font-medium transition-all ${
                      !cell
                        ? 'invisible'
                        : inSelectedWeek
                        ? 'bg-red-600/40 text-white border border-red-500/50'
                        : 'text-(--text-secondary) hover:bg-white/10 hover:text-(--text-primary)'
                    }`}
                    whileHover={cell ? { scale: 1.1 } : {}}
                    whileTap={cell ? { scale: 0.9 } : {}}
                  >
                    {cell?.getDate()}
                    {isToday && (
                      <span className="absolute bottom-0.5 left-1/2 -translate-x-1/2 w-1 h-1 rounded-full bg-red-500" />
                    )}
                  </motion.button>
                );
              })}
            </div>

            {/* Selected week label */}
            <div className="mt-4 p-3 rounded-lg bg-red-600/10 border border-red-600/20 text-center">
              <p className="text-xs text-(--text-secondary)">Selected week</p>
              <p className="font-semibold text-red-400 text-sm mt-0.5">
                {forecast?.weekLabel ?? 'Loading…'}
              </p>
            </div>
          </motion.div>

          {/* Quick stats panel */}
          <motion.div
            className="p-6 rounded-xl border border-white/10 bg-gradient-to-br from-white/5 to-white/10 backdrop-blur-sm flex flex-col justify-center"
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            {loading ? (
              <div className="text-center py-10">
                <motion.div
                  className="w-10 h-10 border-2 border-red-500 border-t-transparent rounded-full mx-auto mb-4"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                />
                <p className="text-(--text-secondary)">Running ML forecast…</p>
              </div>
            ) : error ? (
              <p className="text-red-400 text-center">{error}</p>
            ) : forecast ? (
              <div className="space-y-3">
                <p className="text-sm font-semibold text-(--text-secondary) mb-3">Week at a Glance</p>
                {forecast.days.map((day) => {
                  const colors = congestionColor(day.congestion);
                  return (
                    <div key={day.date} className="flex items-center gap-3">
                      <div className="w-8 text-xs font-bold text-(--text-secondary)">{day.dayName.slice(0,3)}</div>
                      <div className={`w-2 h-2 rounded-full flex-shrink-0 ${colors.dot}`} />
                      <div className="flex-1 text-sm">
                        {day.isHoliday ? (
                          <span className="text-gray-500">⛔ Holiday — Closed</span>
                        ) : day.overall !== null ? (
                          <span className="text-(--text-primary)">
                            <span className="font-semibold">{day.overall} min</span>
                            <span className={`ml-2 text-xs ${colors.text}`}>{congestionLabel(day.congestion)}</span>
                          </span>
                        ) : (
                          <span className="text-(--text-secondary)">No data</span>
                        )}
                      </div>
                      {day.bestTime && (
                        <div className="text-xs text-green-400 font-medium">
                          Best: {day.bestTime}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            ) : null}
          </motion.div>
        </div>

        {/* Day cards */}
        <AnimatePresence mode="wait">
          {forecast && !loading && (
            <motion.div
              key={forecast.weekOf}
              className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.5 }}
            >
              {forecast.days.map((day, idx) => {
                const colors = congestionColor(day.congestion);
                return (
                  <motion.div
                    key={day.date}
                    className={`p-4 rounded-xl border border-white/10 bg-gradient-to-br ${colors.bg} backdrop-blur-sm`}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.05, duration: 0.4 }}
                    whileHover={{ scale: 1.03, y: -3 }}
                  >
                    {/* Day header */}
                    <div className="mb-3">
                      <p className="font-bold text-white text-sm">{day.dayName.slice(0, 3)}</p>
                      <p className="text-xs text-(--text-secondary)">{day.shortDate}</p>
                    </div>

                    {day.isHoliday ? (
                      <div className="space-y-1">
                        <span className="text-lg">⛔</span>
                        <p className="text-xs text-gray-400 font-semibold">Holiday</p>
                        <p className="text-xs text-gray-500">Closed</p>
                      </div>
                    ) : day.overall !== null ? (
                      <div className="space-y-2">
                        {/* Overall */}
                        <div>
                          <p className="text-2xl font-bold text-white">{Math.round(day.overall)}</p>
                          <p className="text-xs text-(--text-secondary)">min avg</p>
                        </div>

                        {/* Congestion badge */}
                        <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-bold text-white ${colors.badge}`}>
                          {congestionLabel(day.congestion)}
                        </span>

                        {/* Best time */}
                        {day.bestTime && (
                          <div className="pt-1 border-t border-white/10">
                            <p className="text-xs text-green-400 font-semibold">
                              Best {day.bestTime}
                            </p>
                            <p className="text-xs text-(--text-secondary)">{day.bestWait} min</p>
                          </div>
                        )}

                        {/* Worst time */}
                        {day.worstTime && (
                          <div>
                            <p className="text-xs text-red-400 font-semibold">
                              Avoid {day.worstTime}
                            </p>
                            <p className="text-xs text-(--text-secondary)">{day.worstWait} min</p>
                          </div>
                        )}
                      </div>
                    ) : (
                      <p className="text-xs text-(--text-secondary)">No prediction</p>
                    )}
                  </motion.div>
                );
              })}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Footer note */}
        <motion.p
          className="text-center text-xs text-(--text-secondary) mt-6"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          viewport={{ once: true }}
        >
          Forecasts generated by the Random Forest model using 500 Monte Carlo simulations per hour
        </motion.p>
      </div>
    </section>
  );
}
