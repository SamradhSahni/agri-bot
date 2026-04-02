import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

export interface ChatRequest {
  query: string;
  session_id: string;
  farmer_id?: number;
  crop?: string;
  state?: string;
  language?: string;
}

export interface Source {
  query: string;
  answer: string;
  crop: string;
  state: string;
  source: string;
  score: number;
}

export interface ChatResponse {
  answer: string;
  intent: string;
  language: string;
  sources: Source[];
  session_id: string;
  model_used: string;
}

export interface MSPResponse {
  crop: string;
  price: number;
  season: string;
  year: number;
  source: string;
}

export const sendMessage = async (req: ChatRequest): Promise<ChatResponse> => {
  const res = await axios.post(`${API_BASE}/chat`, req);
  return res.data;
};

export const getMSP = async (crop: string, state: string): Promise<MSPResponse> => {
  const res = await axios.get(`${API_BASE}/msp`, { params: { crop, state } });
  return res.data;
};

export const submitFeedback = async (
  message_id: number,
  rating: number,
  correction?: string
) => {
  const res = await axios.post(`${API_BASE}/feedback`, {
    message_id,
    rating,
    correction,
  });
  return res.data;
};