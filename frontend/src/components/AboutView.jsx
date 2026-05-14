import './AboutView.css';

const ETHICS = [
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="11" width="18" height="11" rx="2"/>
        <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
      </svg>
    ),
    title: 'Privacy First',
    body: 'All processing runs entirely on your device. No text is ever stored, logged, or transmitted to any external server.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 2C8 2 5 5 5 9c0 3 1.5 5.5 4 7v2h6v-2c2.5-1.5 4-4 4-7 0-4-3-7-7-7z"/>
        <path d="M9 20h6"/>
        <path d="M12 14v-4"/>
      </svg>
    ),
    title: 'Support Tool Only',
    body: 'This tool assists — it does not replace — qualified mental health professionals. Every result must be interpreted by a trained clinician within the full clinical context.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10"/>
        <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
      </svg>
    ),
    title: 'Cultural Awareness',
    body: 'This tool was developed using English-language data and may not generalise well across all cultural and linguistic contexts. Apply professional judgement accordingly.',
  },
  {
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
        <circle cx="9" cy="7" r="4"/>
        <path d="M23 21v-2a4 4 0 0 0-3-3.87"/>
        <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
      </svg>
    ),
    title: 'Consent Required',
    body: "Obtain informed consent before analysing any person's text. This tool must not be used for covert monitoring or without the individual's knowledge.",
  },
];

const TOOL_DETAILS = [
  { label: 'Intended users',   value: 'Licensed mental health professionals' },
  { label: 'Categories',       value: 'Depression · Anxiety · Crisis · Neutral' },
  { label: 'Phrase analysis',  value: 'Importance-weighted word highlights' },
  { label: 'Inference',        value: '100% local — no internet required' },
];

export default function AboutView() {
  return (
    <div className="about-view">

      {/* Hero */}
      <div className="about-hero card">
        <div className="about-hero-mark">
          <svg viewBox="0 0 40 40" fill="none" aria-hidden="true">
            <circle cx="20" cy="20" r="18" fill="var(--sage-subtle)" stroke="var(--sage)" strokeWidth="1.2"/>
            <path
              d="M20 10 C20 10, 14 15, 14 20.5 C14 25 16.5 28 20 29 C23.5 28 26 25 26 20.5 C26 15 20 10 20 10Z"
              fill="var(--sage)"
              opacity="0.8"
            />
            <path d="M20 29 L20 33" stroke="var(--sage)" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
        </div>
        <div>
          <h2 className="about-hero-title">MindSense Counselor Support</h2>
          <p className="about-hero-sub">
            Early wellbeing signal detection for licensed mental health practitioners.
            Built to support — not substitute — professional clinical judgement.
            All processing is private and local.
          </p>
        </div>
      </div>

      {/* Ethics cards */}
      <div className="about-ethics-grid">
        {ETHICS.map((item, i) => (
          <div
            key={i}
            className="about-ethics-card card animate-fade-up"
            style={{ animationDelay: `${i * 55}ms` }}
          >
            <div className="about-ethics-icon">{item.icon}</div>
            <h3 className="about-ethics-title">{item.title}</h3>
            <p className="about-ethics-body">{item.body}</p>
          </div>
        ))}
      </div>

      {/* Tool details */}
      <div className="about-details card">
        <span className="label-caps about-section-label">Tool details</span>
        <div className="about-details-grid">
          {TOOL_DETAILS.map((m, i) => (
            <div key={i} className="about-detail-item">
              <span className="label-caps">{m.label}</span>
              <span className="about-detail-value">{m.value}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Crisis resources */}
      <div className="about-resources card">
        <span className="label-caps about-section-label">Indian crisis helplines</span>
        <div className="about-resources-grid">
          {[
            { name: 'iCall (TISS)',           detail: '9152987821',    hours: 'Mon–Sat 8am–10pm' },
            { name: 'Vandrevala Foundation',  detail: '1860-2662-345', hours: '24/7 free' },
            { name: 'Snehi',                  detail: '044-24640050',  hours: '24/7' },
            { name: 'NIMHANS Helpline',       detail: '080-46110007',  hours: '' },
            { name: 'National Emergency',     detail: '112',           hours: '' },
            { name: 'YourDOST',               detail: 'yourdost.com',  hours: 'Online' },
          ].map((r, i) => (
            <div key={i} className="about-resource-item">
              <span className="about-resource-name">{r.name}</span>
              <span className="about-resource-detail">
                {r.detail}
                {r.hours && <em> · {r.hours}</em>}
              </span>
            </div>
          ))}
        </div>
      </div>

      <p className="about-footer">
        MindSense is a research prototype. Always follow your organisation's clinical protocols
        when using any technology-assisted screening tool.
      </p>
    </div>
  );
}
