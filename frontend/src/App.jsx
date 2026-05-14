import { useState } from 'react';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import AnalyzeView from './components/AnalyzeView';
import TrendView from './components/TrendView';
import AboutView from './components/AboutView';
import './App.css';

export default function App() {
  const [activeView, setActiveView] = useState('analyze');

  const views = {
    analyze: <AnalyzeView />,
    trends: <TrendView />,
    about: <AboutView />,
  };

  return (
    <div className="app-shell">
      <Header />
      <Sidebar activeView={activeView} onNavigate={setActiveView} />
      <main className="app-main">
        <div className="app-content" key={activeView}>
          {views[activeView]}
        </div>
      </main>
    </div>
  );
}
