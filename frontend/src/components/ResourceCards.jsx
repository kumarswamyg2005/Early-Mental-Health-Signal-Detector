import './ResourceCards.css';

export default function ResourceCards({ resources, riskLevel }) {
  if (!resources?.length) return null;

  return (
    <div className="resources-wrap">
      <div className="resources-header">
        <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" width="14" height="14" aria-hidden="true">
          <path d="M9 2C6.2 2 4 4.2 4 7c0 2 1 3.8 2.6 4.8L7 14h4l.4-2.2C13 10.8 14 9 14 7c0-2.8-2.2-5-5-5z"/>
          <path d="M7 14h4M8 17h2"/>
        </svg>
        <h4>Recommended resources</h4>
      </div>
      <div className="resources-grid">
        {resources.map((r, i) => (
          <div key={i} className={`resource-card resource-card--${riskLevel?.toLowerCase()}`}>
            <div className="resource-name">{r.name}</div>
            <div className="resource-detail">{r.detail}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
