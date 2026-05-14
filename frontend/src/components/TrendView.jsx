import { useState } from 'react';
import RiskBadge from './RiskBadge';
import './TrendView.css';

const API = 'http://127.0.0.1:8000';

const RISK_ORDER = { HIGH: 4, MODERATE: 3, LOW: 2, MINIMAL: 1 };
const RISK_COLOR = {
  HIGH:     '#8C2E42',
  MODERATE: '#A85428',
  LOW:      '#65520E',
  MINIMAL:  '#3E6C52',
};

export default function TrendView() {
  const [inputs, setInputs] = useState([
    { id: 1, label: 'Session 1', text: '' },
    { id: 2, label: 'Session 2', text: '' },
    { id: 3, label: 'Session 3', text: '' },
  ]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  function addEntry() {
    const nextId = Math.max(...inputs.map((i) => i.id)) + 1;
    setInputs([...inputs, { id: nextId, label: `Session ${nextId}`, text: '' }]);
  }
  function removeEntry(id) {
    if (inputs.length <= 2) return;
    setInputs(inputs.filter((i) => i.id !== id));
  }
  function updateText(id, text) {
    setInputs(inputs.map((i) => (i.id === id ? { ...i, text } : i)));
  }

  async function handleAnalyze(e) {
    e.preventDefault();
    const filled = inputs.filter((i) => i.text.trim());
    if (filled.length < 2) {
      setError('Please fill in at least 2 sessions for trend analysis.');
      return;
    }
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const res = await fetch(`${API}/analyze/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts: filled.map((i) => i.text.trim()) }),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail ?? `Server error ${res.status}`);
      }
      const data = await res.json();
      setResults({ entries: filled, results: data.results });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const highestRisk = results
    ? results.results.reduce((best, r) =>
        (RISK_ORDER[r.risk_level] ?? 0) > (RISK_ORDER[best.risk_level] ?? 0) ? r : best,
        results.results[0]
      )
    : null;

  return (
    <div className="trend-view">

      <div className="tv-header">
        <h2>Trend Analysis</h2>
        <p>Compare de-identified text entries across sessions to identify shifts in wellbeing signals over time.</p>
      </div>

      <div className="tv-disclaimer">
        <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" width="13" height="13" aria-hidden="true">
          <path d="M8 2l6 11H2L8 2z"/>
          <path d="M8 7v3M8 12h.01"/>
        </svg>
        <span>Remove all personal identifiers from text before uploading. Each session's text is processed locally and immediately discarded.</span>
      </div>

      <div className="tv-input-card card">
        <form onSubmit={handleAnalyze}>
          <div className="tv-entries">
            {inputs.map((entry) => (
              <div key={entry.id} className="tv-entry animate-fade-in">
                <div className="tv-entry-header">
                  <span className="tv-entry-label label-caps">{entry.label}</span>
                  {inputs.length > 2 && (
                    <button
                      type="button"
                      className="tv-remove-btn"
                      onClick={() => removeEntry(entry.id)}
                      title={`Remove ${entry.label}`}
                      aria-label={`Remove ${entry.label}`}
                    >
                      ×
                    </button>
                  )}
                </div>
                <textarea
                  className="tv-textarea"
                  value={entry.text}
                  onChange={(e) => updateText(entry.id, e.target.value)}
                  placeholder={`Paste anonymised text for ${entry.label}…`}
                  rows={3}
                  maxLength={5000}
                />
              </div>
            ))}
          </div>

          <div className="tv-form-footer">
            <button type="button" className="tv-add-btn" onClick={addEntry} disabled={inputs.length >= 10}>
              + Add session
            </button>
            <button type="submit" className="tv-analyze-btn" disabled={loading}>
              {loading ? (
                <><span className="av-spinner" aria-hidden="true" /> Assessing…</>
              ) : (
                'Analyse Trends'
              )}
            </button>
          </div>
        </form>
      </div>

      {error && (
        <div className="av-error animate-fade-in" role="alert">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" width="14" height="14" aria-hidden="true">
            <circle cx="8" cy="8" r="6"/><path d="M8 5v3M8 11h.01"/>
          </svg>
          {error}
        </div>
      )}

      {results && (
        <div className="tv-results animate-fade-up">

          {/* Summary */}
          <div className="tv-summary card">
            <div className="tv-summary-inner">
              <span className="label-caps">Highest level detected</span>
              {highestRisk && <RiskBadge level={highestRisk.risk_level} size="lg" />}
            </div>
            <span className="tv-summary-note">
              Across {results.entries.length} session{results.entries.length !== 1 ? 's' : ''} analysed
            </span>
          </div>

          {/* Bar chart */}
          <div className="tv-timeline card">
            <h3 className="tv-section-title">
              <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" width="14" height="14" aria-hidden="true">
                <path d="M3 15l4-6 4 3 4-7"/><path d="M3 17h14"/>
              </svg>
              Risk Level Over Sessions
            </h3>
            <div className="tv-chart" role="img" aria-label="Risk level bar chart across sessions">
              {results.results.map((r, i) => {
                const entry = results.entries[i];
                const score = RISK_ORDER[r.risk_level] ?? 1;
                const heightPct = (score / 4) * 100;
                return (
                  <div key={i} className="tv-bar-col">
                    <div className="tv-bar-wrap">
                      <div
                        className="tv-bar animate-bar"
                        style={{ height: `${heightPct}%`, background: RISK_COLOR[r.risk_level] }}
                        title={`${entry.label}: ${r.risk_level}`}
                      />
                    </div>
                    <span className="tv-bar-label">{entry.label}</span>
                    <RiskBadge level={r.risk_level} size="sm" />
                  </div>
                );
              })}
            </div>
          </div>

          {/* Score table */}
          <div className="tv-table-card card">
            <h3 className="tv-section-title">
              <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" width="14" height="14" aria-hidden="true">
                <rect x="2" y="2" width="14" height="14" rx="2"/>
                <path d="M6 9h6M6 12h4M6 6h6"/>
              </svg>
              Score Breakdown
            </h3>
            <div className="tv-table-wrap">
              <table className="tv-table">
                <thead>
                  <tr>
                    <th>Session</th>
                    <th>Depression</th>
                    <th>Anxiety</th>
                    <th>Crisis</th>
                    <th>Neutral</th>
                    <th>Level</th>
                  </tr>
                </thead>
                <tbody>
                  {results.results.map((r, i) => (
                    <tr key={i}>
                      <td className="tv-td-entry">{results.entries[i].label}</td>
                      <td>{Math.round((r.scores.depression ?? 0) * 100)}%</td>
                      <td>{Math.round((r.scores.anxiety    ?? 0) * 100)}%</td>
                      <td>{Math.round((r.scores.crisis     ?? 0) * 100)}%</td>
                      <td>{Math.round((r.scores.neutral    ?? 0) * 100)}%</td>
                      <td><RiskBadge level={r.risk_level} size="sm" /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Clinical note */}
          <div className="tv-clinical-note">
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" width="13" height="13" aria-hidden="true">
              <circle cx="8" cy="8" r="6.5"/><path d="M8 5.5v3M8 11h.01"/>
            </svg>
            <span>Trend data is for clinical decision support only. All session texts were processed locally and are not retained.</span>
          </div>
        </div>
      )}
    </div>
  );
}
