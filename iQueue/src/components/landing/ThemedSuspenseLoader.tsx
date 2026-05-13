import { motion } from 'framer-motion';
import logo from '../../assets/iQueueLogoWordmark.png';

type Props = {
  label?: string;
};

export default function ThemedSuspenseLoader({ label = 'Loading…' }: Props) {
  return (
    <div
      role="status"
      aria-busy="true"
      className="w-full flex items-center justify-center py-16"
    >
      <div className="flex items-center gap-4">
        <motion.img
          src={logo}
          alt="iQueue"
          className="h-8 opacity-90"
          animate={{ y: [0, -6, 0] }}
          transition={{ duration: 1.4, repeat: Infinity }}
        />

        <div className="flex items-center gap-2">
          <motion.span
            className="w-2 h-2 rounded-full bg-current/60"
            style={{ color: 'var(--text-secondary)' }}
            animate={{ y: [0, -6, 0] }}
            transition={{ duration: 0.9, repeat: Infinity, delay: 0 }}
          />
          <motion.span
            className="w-2 h-2 rounded-full bg-current/60"
            style={{ color: 'var(--text-secondary)' }}
            animate={{ y: [0, -6, 0] }}
            transition={{ duration: 0.9, repeat: Infinity, delay: 0.15 }}
          />
          <motion.span
            className="w-2 h-2 rounded-full bg-current/60"
            style={{ color: 'var(--text-secondary)' }}
            animate={{ y: [0, -6, 0] }}
            transition={{ duration: 0.9, repeat: Infinity, delay: 0.3 }}
          />
        </div>

        <span className="ml-3 text-sm landing-text-secondary">{label}</span>
      </div>
    </div>
  );
}
