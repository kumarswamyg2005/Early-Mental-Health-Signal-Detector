import { useState } from 'react';
import ResultPanel from './ResultPanel';
import './AnalyzeView.css';

const API = 'http://127.0.0.1:8000';

const EXAMPLES = [
  "I've been feeling really overwhelmed lately and can't seem to find joy in things I used to love.",
  "Everything feels hopeless and I don't see the point of anything anymore.",
  "I feel anxious all the time, my heart races and I can't stop worrying about everything.",
  "Today was a great day! I went for a walk and felt really present and grateful.",
];

export default function AnalyzeView() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [submitted, setSubmitted] = useState('');

  async function handleAnalyze(e) {
    e.preventDefault();
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setSubmitted(text.trim());
    try {
      const res = await fetch(`${API}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim(), num_lime_samples: 50 }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail ?? `Server error ${res.status}`);
      }
      setResult(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function handleExample(ex) {
    setText(ex);
    setResult(null);
    setError(null);
  }

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const overLimit = text.length > 5000;
  const tooShort = wordCount > 0 && wordCount < 10;

  return (
    <div className="analyze-view">

      {/* Professional disclaimer */}
      <div className="av-disclaimer">
        <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" width="14" height="14" aria-hidden="true">
          <path d="M8 2l6 11H2L8 2z"/>
          <path d="M8 7v3M8 12h.01"/>
        </svg>
        <span>
          <strong>Professionals only.</strong> This tool assists licensed mental health practitioners.
          It is not a diagnostic instrument — results must be interpreted by a qualified clinician.
          Ensure informed consent before analyzing any individual's text.
        </span>
      </div>

      {/* Input card */}
      <div className="av-input-card card">
        <div className="av-card-header">
          <div>
            <h2 className="av-card-title">Text Assessment</h2>
            <p className="av-card-sub">
              Paste de-identified text below. Remove all names, dates, and personal identifiers before analysis.
            </p>
          </div>
        </div>

        {/* Quick examples */}
        <div className="av-examples">
          <span className="label-caps">Quick examples</span>
          <div className="av-chips">
            {EXAMPLES.map((ex, i) => (
              <button key={i} className="av-chip" onClick={() => handleExample(ex)} type="button">
                {ex.slice(0, 48)}…
              </button>
            ))}
          </div>
        </div>

        <form onSubmit={handleAnalyze} className="av-form">
          <div className="av-textarea-wrap">
            <textarea
              className="av-textarea"
              value={text}
              onChange={(e) => { setText(e.target.value); setResult(null); }}
              placeholder="Paste anonymised text here — journal entry, chat message, or written reflection…"
              rows={7}
              maxLength={5100}
              aria-label="Text to analyze"
            />
            <div className="av-textarea-footer">
              <span className={`av-word-count${tooShort ? ' av-word-warn' : wordCount >= 50 ? ' av-word-ok' : ''}`}>
                {wordCount > 0 ? `${wordCount} word${wordCount !== 1 ? 's' : ''}` : ''}
                {tooShort && ' — at least 10 words needed'}
                {wordCount >= 50 && wordCount <= 2000 && ' · good length'}
              </span>
              <span className={`av-char-count${overLimit ? ' av-char-over' : ''}`}>
                {text.length.toLocaleString()} / 5,000
              </span>
            </div>
          </div>

          <div className="av-form-footer">
            <button
              type="button"
              className="av-btn-secondary"
              onClick={() => { setText(''); setResult(null); setError(null); }}
              disabled={!text && !result}
            >
              Clear
            </button>
            <button
              type="submit"
              className="av-btn-primary"
              disabled={loading || !text.trim() || overLimit || tooShort}
            >
              {loading ? (
                <>
                  <span className="av-spinner" aria-hidden="true" />
                  Assessing…
                </>
              ) : (
                <>
                  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="14" height="14" aria-hidden="true">
                    <circle cx="7" cy="7" r="5"/>
                    <path d="M11 11l3 3"/>
                  </svg>
                  Run Assessment
                </>
              )}
            </button>
          </div>
        </form>
      </div>

      {/* Error state */}
      {error && (
        <div className="av-error animate-fade-in" role="alert">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" width="14" height="14" aria-hidden="true">
            <circle cx="8" cy="8" r="6"/><path d="M8 5v3M8 11h.01"/>
          </svg>
          <span>{error}</span>
        </div>
      )}

      {/* Results */}
      {result && <ResultPanel result={result} inputText={submitted} />}
    </div>
  );
}
