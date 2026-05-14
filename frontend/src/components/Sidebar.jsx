import './Sidebar.css';

const NAV_ITEMS = [
  {
    id: 'analyze',
    label: 'Analyze Text',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M4 6h12M4 10h8M4 14h10"/>
        <circle cx="15" cy="14" r="3"/>
        <path d="M17.5 16.5l2 2"/>
      </svg>
    ),
  },
  {
    id: 'trends',
    label: 'Trend Analysis',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M3 15l4-6 4 3 4-7"/>
        <path d="M3 17h14"/>
      </svg>
    ),
  },
  {
    id: 'about',
    label: 'About & Ethics',
    icon: (
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <circle cx="10" cy="10" r="7.5"/>
        <path d="M10 9.5v5M10 7h.01"/>
      </svg>
    ),
  },
];

const CRISIS_LINES = [
  { name: 'iCall (TISS)',          detail: '9152987821',    hours: 'Mon–Sat 8am–10pm' },
  { name: 'Vandrevala Foundation', detail: '1860-2662-345', hours: '24 / 7' },
  { name: 'Snehi',                 detail: '044-24640050',  hours: '24 / 7' },
  { name: 'NIMHANS',               detail: '080-46110007',  hours: '' },
  { name: 'Emergency',             detail: '112',           hours: '' },
];

export default function Sidebar({ activeView, onNavigate }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-top">
        {/* Navigation */}
        <div className="sidebar-section">
          <span className="label-caps sidebar-section-label">Views</span>
          <nav className="sidebar-nav">
            {NAV_ITEMS.map((item) => (
              <button
                key={item.id}
                className={`sidebar-nav-item${activeView === item.id ? ' active' : ''}`}
                onClick={() => onNavigate(item.id)}
              >
                <span className="nav-icon">{item.icon}</span>
                <span className="nav-label">{item.label}</span>
                {activeView === item.id && <span className="nav-pip" aria-hidden="true" />}
              </button>
            ))}
          </nav>
        </div>

        {/* Crisis lines */}
        <div className="sidebar-section sidebar-crisis">
          <span className="label-caps sidebar-section-label">Crisis Helplines</span>
          <ul className="crisis-list">
            {CRISIS_LINES.map((line) => (
              <li key={line.name} className="crisis-item">
                <span className="crisis-name">{line.name}</span>
                <span className="crisis-detail">
                  {line.detail}
                  {line.hours && <em> · {line.hours}</em>}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <footer className="sidebar-footer">
        <div className="sidebar-notice">
          <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" width="12" height="12" aria-hidden="true">
            <path d="M7 1l6 10H1L7 1z"/>
            <path d="M7 5.5v3M7 10h.01"/>
          </svg>
          <span>For licensed professionals only</span>
        </div>
        <span className="sidebar-version">v1.0 · Runs offline</span>
      </footer>
    </aside>
  );
}
