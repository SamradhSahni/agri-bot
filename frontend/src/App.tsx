import React, { useState } from 'react';
import FarmerProfile, { Profile } from './components/FarmerProfile';
import ChatWindow from './components/ChatWindow';
import SourcesPanel from './components/SourcesPanel';
import { Source } from './services/api';

const App: React.FC = () => {
  const [profile, setProfile] = useState<Profile>({
    name:     '',
    state:    'RAJASTHAN',
    crop:     'wheat',
    language: 'hi',
  });

  const [sources, setSources] = useState<Source[]>([]);
  const [intent,  setIntent]  = useState<string>('crop_advisory');

  return (
    <div className="min-h-screen bg-green-50 p-4">
      <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-4 gap-4 h-screen max-h-screen">

        {/* Left — Profile */}
        <div className="lg:col-span-1 space-y-4">
          <FarmerProfile profile={profile} onChange={setProfile} />

          {/* MSP quick lookup */}
          <div className="bg-white rounded-xl shadow p-4">
            <h2 className="text-green-700 font-bold mb-2">💰 MSP मूल्य</h2>
            <p className="text-xs text-gray-500">
              {profile.crop
                ? `${profile.crop} का MSP देखने के लिए चैट में पूछें।`
                : 'प्रोफाइल में फसल चुनें'}
            </p>
          </div>
        </div>

        {/* Center — Chat */}
        <div className="lg:col-span-2 flex flex-col" style={{ height: 'calc(100vh - 2rem)' }}>
          <ChatWindow
            profile={profile}
            onNewSources={(s, i) => { setSources(s); setIntent(i); }}
          />
        </div>

        {/* Right — Sources */}
        <div className="lg:col-span-1">
          <SourcesPanel sources={sources} intent={intent} />
        </div>

      </div>
    </div>
  );
};

export default App;