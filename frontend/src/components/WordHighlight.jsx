import './WordHighlight.css';

export default function WordHighlight({ text, wordWeights }) {
  if (!text) return null;

  const tokens = text.split(/(\s+)/);
  const hasTags = Object.keys(wordWeights ?? {}).length > 0;

  return (
    <div className="word-highlight-wrap">
      <div className="word-highlight-legend">
        <span className="wh-legend-item">
          <span className="wh-swatch wh-risk" />
          Risk signal
        </span>
        <span className="wh-legend-item">
          <span className="wh-swatch wh-protect" />
          Protective signal
        </span>
      </div>
      <div className="word-highlight-text">
        {hasTags ? (
          tokens.map((token, i) => {
            const clean = token.replace(/[^\w']/g, '').toLowerCase();
            const weight = (wordWeights ?? {})[clean] ?? 0;
            if (weight > 0.03) {
              return (
                <mark key={i} className="wh-mark wh-mark--risk" title={`Risk signal: ${weight > 0 ? '+' : ''}${weight.toFixed(3)}`}>
                  {token}
                </mark>
              );
            }
            if (weight < -0.03) {
              return (
                <mark key={i} className="wh-mark wh-mark--protect" title={`Protective signal: ${weight.toFixed(3)}`}>
                  {token}
                </mark>
              );
            }
            return <span key={i}>{token}</span>;
          })
        ) : (
          <span>{text}</span>
        )}
      </div>
    </div>
  );
}
