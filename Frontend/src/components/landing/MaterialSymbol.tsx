interface MaterialSymbolProps {
  icon: string;
  className?: string;
}

export default function MaterialSymbol({ icon, className = '' }: MaterialSymbolProps) {
  return (
    <span aria-hidden="true" className={`material-symbols-rounded ${className}`.trim()}>
      {icon}
    </span>
  );
}