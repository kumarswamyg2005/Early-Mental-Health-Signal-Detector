import './Header.css';

export default function Header() {
  return (
    <header className="header">
      <div className="header-brand">
        <div className="header-mark">
          <svg viewBox="0 0 32 32" fill="none" aria-hidden="true">
            <circle cx="16" cy="16" r="14" fill="var(--sage-subtle)" stroke="var(--sage)" strokeWidth="1.2"/>
            <path
              d="M16 10 C16 10, 12 13, 12 16.5 C12 19.5 13.8 21.5 16 22 C18.2 21.5 20 19.5 20 16.5 C20 13 16 10 16 10Z"
              fill="var(--sage)"
              opacity="0.85"
            />
            <path
              d="M16 22 L16 26"
              stroke="var(--sage)"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
            <path
              d="M13 18 C11 17 10 15 11 13"
              stroke="var(--sage-light)"
              strokeWidth="1"
              strokeLinecap="round"
              opacity="0.6"
            />
          </svg>
        </div>
        <div className="header-text">
          <span className="header-title">MindSense</span>
          <span className="header-sub">Counselor Support</span>
        </div>
      </div>

      <div className="header-right">
        <span className="hdr-badge hdr-badge--local">
          <span className="hdr-dot" aria-hidden="true" />
          Local only
        </span>
        <span className="hdr-badge hdr-badge--private">
          <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" width="11" height="11" aria-hidden="true">
            <rect x="2" y="6" width="10" height="7" rx="1.5"/>
            <path d="M4.5 6V4.5a2.5 2.5 0 0 1 5 0V6"/>
          </svg>
          No data stored
        </span>
      </div>
    </header>
  );
}
