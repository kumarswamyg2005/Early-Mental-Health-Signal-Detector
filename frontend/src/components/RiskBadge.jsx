import './RiskBadge.css';

const CONFIG = {
  HIGH:     { label: 'High',     cls: 'risk-high',     dot: '#8C2E42' },
  MODERATE: { label: 'Moderate', cls: 'risk-moderate', dot: '#A85428' },
  LOW:      { label: 'Low',      cls: 'risk-low',      dot: '#65520E' },
  MINIMAL:  { label: 'Minimal',  cls: 'risk-minimal',  dot: '#3E6C52' },
};

export default function RiskBadge({ level, size = 'md' }) {
  const cfg = CONFIG[level] ?? CONFIG.MINIMAL;
  return (
    <span className={`risk-badge risk-badge--${size} ${cfg.cls}`}>
      <span className="risk-badge-dot" style={{ background: cfg.dot }} aria-hidden="true" />
      <span className="risk-badge-label">{cfg.label} risk</span>
    </span>
  );
}
