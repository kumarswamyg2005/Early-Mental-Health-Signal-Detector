import RiskBadge from './RiskBadge';
import ScoreBars from './ScoreBar';
import WordHighlight from './WordHighlight';
import ResourceCards from './ResourceCards';
import './ResultPanel.css';

export default function ResultPanel({ result, inputText }) {
  if (!result) return null;

  return (
    <div className="result-panel animate-fade-up">

      {/* Risk banner */}
      <div className={`result-banner result-banner--${result.risk_level.toLowerCase()}`}>
        <div className="result-banner-left">
          <RiskBadge level={result.risk_level} size="lg" />
          <span className="result-primary-label">
            Primary signal: <strong>{result.primary_label}</strong>
          </span>
        </div>
        <span className="label-caps result-banner-tag">Assessment result</span>
      </div>

      {/* Scores + Features side by side */}
      <div className="result-grid">

        <section className="result-section card">
          <h3 className="result-section-title">
            <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" width="14" height="14" aria-hidden="true">
              <rect x="2" y="10" width="3" height="6" rx="1"/>
              <rect x="7.5" y="5.5" width="3" height="10.5" rx="1"/>
              <rect x="13" y="2" width="3" height="14" rx="1"/>
            </svg>
            Confidence Scores
          </h3>
          <ScoreBars scores={result.scores} />
        </section>

        <section className="result-section card">
          <h3 className="result-section-title">
            <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" width="14" height="14" aria-hidden="true">
              <path d="M3 5h12M3 9h8M3 13h5"/>
            </svg>
            Language Indicators
          </h3>
          <div className="features-list">
            {result.top_features?.length > 0 ? (
              result.top_features.map((f, i) => (
                <div key={i} className="feature-item">
                  <span className="feature-dot" />
                  <span className="feature-text">{f.description}</span>
                </div>
              ))
            ) : (
              <span className="no-features">No notable indicators detected</span>
            )}
          </div>
          {result.feature_values && Object.keys(result.feature_values).length > 0 && (
            <details className="feature-details">
              <summary className="feature-details-toggle">View all indicators</summary>
              <div className="feature-table">
                {Object.entries(result.feature_values).map(([name, val], i) => (
                  <div key={i} className="feature-table-row">
                    <span className="fv-name">{name.replace(/_/g, ' ')}</span>
                    <span className="fv-value">
                      {typeof val === 'number' ? (val * 100).toFixed(1) + '%' : val}
                    </span>
                  </div>
                ))}
              </div>
            </details>
          )}
        </section>
      </div>

      {/* Word highlights */}
      {inputText && (
        <section className="result-section card">
          <h3 className="result-section-title">
            <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" width="14" height="14" aria-hidden="true">
              <path d="M2 14l4-10 4 10M4.5 9h5M14 4v10M14 4l2 2M14 4l-2 2"/>
            </svg>
            Highlighted Phrases
          </h3>
          <WordHighlight text={inputText} wordWeights={result.word_weights} />
        </section>
      )}

      {/* Resources */}
      <section className="result-section card">
        <ResourceCards resources={result.resources} riskLevel={result.risk_level} />
      </section>

      {/* Clinical reminder */}
      <div className="result-clinical-note">
        <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" width="13" height="13" aria-hidden="true">
          <circle cx="8" cy="8" r="6.5"/>
          <path d="M8 5.5v3M8 11h.01"/>
        </svg>
        <span>
          This assessment is a <em>clinical support aid only</em> — not a diagnostic instrument.
          Results must be interpreted by a qualified professional in context of the full clinical picture.
          No text has been stored or transmitted.
        </span>
      </div>
    </div>
  );
}
