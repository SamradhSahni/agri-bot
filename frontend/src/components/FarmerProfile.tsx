import React from 'react';

export interface Profile {
  name: string;
  state: string;
  crop: string;
  language: string;
}

interface Props {
  profile: Profile;
  onChange: (p: Profile) => void;
}

const STATES = [
  'UTTAR PRADESH', 'RAJASTHAN', 'MADHYA PRADESH', 'BIHAR',
  'HARYANA', 'JHARKHAND', 'UTTARAKHAND', 'HIMACHAL PRADESH',
  'CHHATTISGARH', 'DELHI'
];

const CROPS = [
  'paddy (dhan)', 'wheat', 'mustard', 'maize (makka)',
  'cotton (kapas)', 'groundnut', 'bajra', 'sugarcane',
  'tomato', 'onion', 'chillies', 'others'
];

const FarmerProfile: React.FC<Props> = ({ profile, onChange }) => {
  const update = (field: keyof Profile, value: string) =>
    onChange({ ...profile, [field]: value });

  return (
    <div className="bg-white rounded-xl shadow p-4 space-y-3">
      <h2 className="text-green-700 font-bold text-lg">🌾 किसान प्रोफाइल</h2>

      <div>
        <label className="text-sm text-gray-600">नाम</label>
        <input
          className="w-full border rounded-lg px-3 py-2 text-sm mt-1"
          placeholder="आपका नाम"
          value={profile.name}
          onChange={e => update('name', e.target.value)}
        />
      </div>

      <div>
        <label className="text-sm text-gray-600">राज्य</label>
        <select
          className="w-full border rounded-lg px-3 py-2 text-sm mt-1"
          value={profile.state}
          onChange={e => update('state', e.target.value)}
        >
          <option value="">राज्य चुनें</option>
          {STATES.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
      </div>

      <div>
        <label className="text-sm text-gray-600">फसल</label>
        <select
          className="w-full border rounded-lg px-3 py-2 text-sm mt-1"
          value={profile.crop}
          onChange={e => update('crop', e.target.value)}
        >
          <option value="">फसल चुनें</option>
          {CROPS.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
      </div>

      <div>
        <label className="text-sm text-gray-600">भाषा</label>
        <select
          className="w-full border rounded-lg px-3 py-2 text-sm mt-1"
          value={profile.language}
          onChange={e => update('language', e.target.value)}
        >
          <option value="hi">हिंदी</option>
          <option value="en">English</option>
        </select>
      </div>
    </div>
  );
};

export default FarmerProfile;