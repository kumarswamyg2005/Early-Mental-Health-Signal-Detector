import './ScoreBar.css';

const LABEL_CONFIG = {
  depression: { color: '#6B5C20', bg: 'rgba(107,92,32,0.12)', label: 'Depression' },
  anxiety:    { color: '#B56030', bg: 'rgba(181,96,48,0.12)', label: 'Anxiety' },
  crisis:     { color: '#9B3347', bg: 'rgba(155,51,71,0.12)', label: 'Crisis' },
  neutral:    { color: '#3E7055', bg: 'rgba(62,112,85,0.12)', label: 'Neutral' },
};

export default function ScoreBars({ scores }) {
  const ordered = ['crisis', 'depression', 'anxiety', 'neutral'];
  return (
    <div className="score-bars">
      {ordered.map((key) => {
        const cfg = LABEL_CONFIG[key];
        const pct = Math.round((scores[key] ?? 0) * 100);
        return (
          <div key={key} className="score-row">
            <div className="score-meta">
              <span className="score-label">{cfg.label}</span>
              <span className="score-value" style={{ color: cfg.color }}>{pct}%</span>
            </div>
            <div className="score-track" style={{ background: cfg.bg }}>
              <div
                className="score-fill animate-bar"
                style={{
                  width: `${pct}%`,
                  background: `linear-gradient(90deg, ${cfg.color}cc, ${cfg.color})`,
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
