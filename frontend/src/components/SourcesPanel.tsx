import React from 'react';
import { Source } from '../services/api';

interface Props {
  sources: Source[];
  intent: string;
}

const INTENT_LABELS: Record<string, string> = {
  pest_id:        '🐛 कीट पहचान',
  disease:        '🍂 रोग',
  crop_advisory:  '🌱 फसल सलाह',
  msp_price:      '💰 MSP मूल्य',
  weather_sowing: '🌦 मौसम / बुवाई',
};

const SourcesPanel: React.FC<Props> = ({ sources, intent }) => {
  if (!sources || sources.length === 0) return null;

  return (
    <div className="bg-white rounded-xl shadow p-4 space-y-3">
      <h2 className="text-green-700 font-bold text-lg">📚 स्रोत</h2>

      <div className="inline-block bg-green-100 text-green-800
                      text-xs font-medium px-2 py-1 rounded-full">
        {INTENT_LABELS[intent] || intent}
      </div>

      {sources.map((src, i) => (
        <div key={i} className="border rounded-lg p-3 text-sm space-y-1">
          <p className="font-medium text-gray-700 text-xs">
            {src.crop && <span className="mr-2">🌾 {src.crop}</span>}
            {src.state && <span>📍 {src.state}</span>}
          </p>
          <p className="text-gray-600 text-xs italic">"{src.query}"</p>
          <p className="text-gray-800 text-xs">{src.answer.slice(0, 120)}...</p>
          <p className="text-gray-400 text-xs">
            स्रोत: {src.source === 'kcc' ? 'किसान कॉल सेंटर' : 'PlantVillage'} |
            स्कोर: {src.score}
          </p>
        </div>
      ))}
    </div>
  );
};

export default SourcesPanel;