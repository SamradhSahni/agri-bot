import React, { useState, useRef, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { sendMessage, submitFeedback, ChatResponse, Source } from '../services/api';
import { Profile } from './FarmerProfile';

interface Message {
  id: string;
  role: 'user' | 'bot';
  content: string;
  intent?: string;
  sources?: Source[];
  timestamp: Date;
}

interface Props {
  profile: Profile;
  onNewSources: (sources: Source[], intent: string) => void;
}

const SESSION_ID = uuidv4();

const ChatWindow: React.FC<Props> = ({ profile, onNewSources }) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id:        uuidv4(),
      role:      'bot',
      content:   'नमस्ते किसान भाई! मैं आपका कृषि सलाहकार हूँ। आप अपनी फसल, रोग, कीट या खाद के बारे में कोई भी सवाल पूछ सकते हैं।',
      timestamp: new Date(),
    }
  ]);
  const [input,   setInput]   = useState('');
  const [loading, setLoading] = useState(false);
  const [feedbackGiven, setFeedbackGiven] = useState<Record<string, number>>({});
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleFeedback = async (msgId: string, rating: number) => {
    if (feedbackGiven[msgId]) return;
    setFeedbackGiven(prev => ({ ...prev, [msgId]: rating }));
    try {
      await submitFeedback(0, rating);
    } catch (err) {
      console.error('Feedback submission failed', err);
    }
  };

  // Voice input
  const startVoice = () => {
    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      alert('Voice input not supported in this browser.');
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = profile.language === 'hi' ? 'hi-IN' : 'en-IN';
    recognition.interimResults = false;

    recognition.onresult = (e: any) => {
      setInput(e.results[0][0].transcript);
    };
    recognition.start();
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMsg: Message = {
      id:        uuidv4(),
      role:      'user',
      content:   input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const res: ChatResponse = await sendMessage({
        query:      input,
        session_id: SESSION_ID,
        crop:       profile.crop,
        state:      profile.state,
        language:   profile.language,
      });

      const botMsg: Message = {
        id:        uuidv4(),
        role:      'bot',
        content:   res.answer,
        intent:    res.intent,
        sources:   res.sources,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, botMsg]);

      if (res.sources?.length > 0) {
        onNewSources(res.sources, res.intent);
      }

    } catch (err) {
      setMessages(prev => [...prev, {
        id:        uuidv4(),
        role:      'bot',
        content:   'माफ करें, कुछ तकनीकी समस्या हुई। कृपया दोबारा कोशिश करें।',
        timestamp: new Date(),
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-white rounded-xl shadow">
      {/* Header */}
      <div className="bg-green-600 text-white px-4 py-3 rounded-t-xl">
        <h1 className="font-bold text-lg">🌾 कृषि सलाहकार</h1>
        <p className="text-green-100 text-xs">
          किसान कॉल सेंटर · AI सहायक
        </p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.map(msg => (
          <div
            key={msg.id}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-xs lg:max-w-md px-4 py-2 rounded-2xl text-sm
                ${msg.role === 'user'
                  ? 'bg-green-500 text-white rounded-br-sm'
                  : 'bg-gray-100 text-gray-800 rounded-bl-sm'
                }`}
            >
              <p className="whitespace-pre-wrap">{msg.content}</p>
              <p className={`text-xs mt-1 ${
                msg.role === 'user' ? 'text-green-100' : 'text-gray-400'
              }`}>
                {msg.timestamp.toLocaleTimeString('hi-IN', {
                  hour: '2-digit', minute: '2-digit'
                })}
              </p>
            </div>
            {msg.role === 'bot' && (
              <div className="flex gap-2 mt-1 ml-1">
                <button
                  onClick={() => handleFeedback(msg.id, 1)}
                  className={`text-xs transition-colors ${
                    feedbackGiven[msg.id] === 1
                      ? 'text-green-500'
                      : 'text-gray-400 hover:text-green-500'
                  }`}
                  disabled={!!feedbackGiven[msg.id]}
                  title="Helpful"
                >👍</button>
                <button
                  onClick={() => handleFeedback(msg.id, -1)}
                  className={`text-xs transition-colors ${
                    feedbackGiven[msg.id] === -1
                      ? 'text-red-400'
                      : 'text-gray-400 hover:text-red-400'
                  }`}
                  disabled={!!feedbackGiven[msg.id]}
                  title="Not helpful"
                >👎</button>
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 px-4 py-3 rounded-2xl rounded-bl-sm">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-green-400 rounded-full animate-bounce delay-100" />
                <div className="w-2 h-2 bg-green-400 rounded-full animate-bounce delay-200" />
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div className="p-3 border-t flex items-center gap-2">
        <input
          className="flex-1 border rounded-full px-4 py-2 text-sm
                     focus:outline-none focus:ring-2 focus:ring-green-400"
          placeholder="अपना सवाल यहाँ लिखें..."
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSend()}
          disabled={loading}
        />

        {/* Voice button */}
        <button
          onClick={startVoice}
          className="bg-gray-100 hover:bg-gray-200 text-gray-600
                     rounded-full w-10 h-10 flex items-center justify-center"
          title="Voice input"
        >
          🎤
        </button>

        {/* Send button */}
        <button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          className="bg-green-500 hover:bg-green-600 disabled:bg-gray-300
                     text-white rounded-full w-10 h-10
                     flex items-center justify-center text-lg"
        >
          ➤
        </button>
      </div>
    </div>
  );
};

export default ChatWindow;