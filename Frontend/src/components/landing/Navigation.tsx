import { motion, AnimatePresence } from 'framer-motion';
import { useState } from 'react';
import MaterialSymbol from './MaterialSymbol';
import logo from '../../assets/iQueueLogoRed.png';

interface NavigationProps {
  scrollProgress: number;
  theme: 'dark' | 'light';
  onThemeToggle: () => void;
}

export default function Navigation({ scrollProgress, theme, onThemeToggle }: NavigationProps) {
  const [isOpen, setIsOpen] = useState(false);

  const navItems = [
    { label: 'Overview', href: '#hero' },
    { label: 'Dataset', href: '#dataset' },
    { label: 'Models', href: '#models' },
    { label: 'Performance', href: '#performance' },
    { label: 'Analytics', href: '#analytics' },
    { label: 'Demo', href: '#demo' },
  ];

  return (
    <>
      {/* Scroll Progress Bar */}
      <div
        className="fixed top-0 left-0 right-0 h-[2px] z-50 transition-all duration-150"
        style={{
          width: `${scrollProgress}%`,
          background: 'linear-gradient(90deg, #dc2626, #f97316)',
        }}
      />

      {/* Navigation Bar */}
      <motion.nav
        className="fixed top-0 left-0 right-0 z-40 backdrop-blur-md bg-[color:var(--surface)] border-b border-[color:var(--border-subtle)]"
        style={{ boxShadow: '0 2px 16px 0 rgba(0,0,0,0.10)' }}
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-3 items-center h-20">

            {/* Left — Logo */}
            <motion.a
              href="#hero"
              className="flex items-center gap-2.5 justify-self-start"
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.97 }}
            >
              <img src={logo} alt="iQueue" className="w-9 h-9" />
            </motion.a>

            {/* Center — Nav Links */}
            <div className="hidden md:flex items-center justify-center gap-9">
              {navItems.map((item) => (
                <a
                  key={item.label}
                  href={item.href}
                  className="relative text-m text-[color:var(--text-secondary)] hover:text-[color:var(--text-primary)] transition-colors duration-200 group py-2"
                >
                  {item.label}
                  {/* Subtle underline on hover */}
                  <span className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-red-600 to-orange-500 scale-x-0 group-hover:scale-x-100 transition-transform duration-250 origin-left rounded-full" />
                </a>
              ))}
            </div>

            {/* Right — Theme + CTA */}
            <div className="hidden md:flex items-center gap-3 justify-self-end">
              {/* Theme Toggle */}
              <motion.button
                className="p-2 rounded-xl border border-[color:var(--border-subtle)] bg-[color:var(--surface)] text-[color:var(--text-primary)] transition-all duration-200 hover:border-red-500/40 hover:shadow-[0_0_12px_rgba(220,38,38,0.15)]"
                aria-label="Toggle light and dark mode"
                onClick={onThemeToggle}
                whileHover={{ scale: 1.06 }}
                whileTap={{ scale: 0.93 }}
              >
                <AnimatePresence mode="wait">
                  <motion.span
                    key={theme}
                    initial={{ rotate: -60, opacity: 0 }}
                    animate={{ rotate: 0, opacity: 1 }}
                    exit={{ rotate: 60, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="block"
                  >
                    <MaterialSymbol icon={theme === 'dark' ? 'light_mode' : 'dark_mode'} className="text-[20px]" />
                  </motion.span>
                </AnimatePresence>
              </motion.button>

              {/* Try Now */}
              <motion.button
                className="brand-contrast px-5 py-2.5 bg-gradient-to-r from-red-700 to-red-500 rounded-xl text-sm font-semibold transition-all duration-200 hover:shadow-[0_4px_20px_rgba(220,38,38,0.35)] hover:brightness-110"
                whileHover={{ scale: 1.04 }}
                whileTap={{ scale: 0.95 }}
              >
                Try Now
              </motion.button>
            </div>

            {/* Mobile Menu Button */}
            <div className="md:hidden justify-self-end">
              <motion.button
                className="p-2.5 rounded-xl border border-[color:var(--border-subtle)] bg-[color:var(--surface)]"
                onClick={() => setIsOpen(!isOpen)}
                whileTap={{ scale: 0.93 }}
              >
                <AnimatePresence mode="wait">
                  <motion.span
                    key={isOpen ? 'close' : 'menu'}
                    initial={{ rotate: -60, opacity: 0 }}
                    animate={{ rotate: 0, opacity: 1 }}
                    exit={{ rotate: 60, opacity: 0 }}
                    transition={{ duration: 0.18 }}
                    className="block text-[color:var(--text-primary)]"
                  >
                    <MaterialSymbol icon={isOpen ? 'close' : 'menu'} className="text-[24px]" />
                  </motion.span>
                </AnimatePresence>
              </motion.button>
            </div>
          </div>

          {/* Mobile Menu */}
          <AnimatePresence>
            {isOpen && (
              <motion.div
                className="md:hidden pb-4 space-y-1"
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.2 }}
              >
                {navItems.map((item) => (
                  <a
                    key={item.label}
                    href={item.href}
                    className="block px-4 py-3 rounded-xl text-sm text-[color:var(--text-secondary)] hover:text-[color:var(--text-primary)] hover:bg-red-500/5 transition-all duration-150"
                    onClick={() => setIsOpen(false)}
                  >
                    {item.label}
                  </a>
                ))}
                <div className="h-px bg-[color:var(--border-subtle)] my-2" />
                <button
                  className="w-full px-4 py-3 rounded-xl text-sm text-[color:var(--text-secondary)] hover:text-[color:var(--text-primary)] hover:bg-red-500/5 transition-all duration-150 text-left"
                  onClick={onThemeToggle}
                >
                  {theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
                </button>
                <button className="w-full px-4 py-3 rounded-xl text-sm font-semibold text-white bg-gradient-to-r from-red-600 to-orange-500">
                  Try Now
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.nav>
    </>
  );
}