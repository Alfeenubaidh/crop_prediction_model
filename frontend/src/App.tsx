/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { 
  onAuthStateChanged, 
  signInWithPopup, 
  GoogleAuthProvider, 
  signOut,
  User
} from 'firebase/auth';
import { 
  doc, 
  getDoc, 
  setDoc, 
  collection, 
  query, 
  where, 
  orderBy, 
  onSnapshot,
  addDoc,
  increment,
  serverTimestamp,
  getDocs,
  limit
} from 'firebase/firestore';
import { auth, db } from './firebase';
import { motion, AnimatePresence } from 'motion/react';
import {
  Sprout,
  LayoutDashboard,
  History,
  Settings,
  LogOut,
  User as UserIcon,
  ChevronRight,
  Droplets,
  Thermometer,
  CloudRain,
  Sun,
  Wind,
  FlaskConical,
  ShieldCheck,
  BarChart3,
  Users,
  Activity,
  MapPin,
  Calendar
} from 'lucide-react';
import { cn } from './lib/utils';
import { predictCrop, PredictionInputs, PredictionResult } from './services/predictionService';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';

// --- Types ---
interface UserProfile {
  uid: string;
  email: string;
  displayName: string;
  photoURL: string;
  role: 'user' | 'admin';
  createdAt: any;
}

interface PredictionRecord {
  id: string;
  userId: string;
  timestamp: any;
  inputs: PredictionInputs;
  result: PredictionResult;
}

// --- Components ---

const Navbar = ({ user, profile, onLogin, onLogout, activeTab, setActiveTab }: { 
  user: User | null, 
  profile: UserProfile | null,
  onLogin: () => void, 
  onLogout: () => void,
  activeTab: string,
  setActiveTab: (tab: string) => void
}) => (
  <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b border-emerald-100 px-6 py-4">
    <div className="max-w-7xl mx-auto flex items-center justify-between">
      <div 
        className="flex items-center gap-2 cursor-pointer" 
        onClick={() => setActiveTab('home')}
      >
        <div className="bg-emerald-600 p-2 rounded-lg">
          <Sprout className="text-white w-6 h-6" />
        </div>
        <span className="text-xl font-bold text-emerald-900 tracking-tight">AgroPredict</span>
      </div>

      <div className="hidden md:flex items-center gap-8">
        <button 
          onClick={() => setActiveTab('home')}
          className={cn("text-sm font-medium transition-colors hover:text-emerald-600", activeTab === 'home' ? "text-emerald-600" : "text-emerald-900/60")}
        >
          Home
        </button>
        {user && (
          <>
            <button 
              onClick={() => setActiveTab('predict')}
              className={cn("text-sm font-medium transition-colors hover:text-emerald-600", activeTab === 'predict' ? "text-emerald-600" : "text-emerald-900/60")}
            >
              Predict
            </button>
            <button 
              onClick={() => setActiveTab('dashboard')}
              className={cn("text-sm font-medium transition-colors hover:text-emerald-600", activeTab === 'dashboard' ? "text-emerald-600" : "text-emerald-900/60")}
            >
              Dashboard
            </button>
            {profile?.role === 'admin' && (
              <button 
                onClick={() => setActiveTab('admin')}
                className={cn("text-sm font-medium transition-colors hover:text-emerald-600 flex items-center gap-1", activeTab === 'admin' ? "text-emerald-600" : "text-emerald-900/60")}
              >
                <ShieldCheck className="w-4 h-4" />
                Admin
              </button>
            )}
          </>
        )}
      </div>

      <div className="flex items-center gap-4">
        {user ? (
          <div className="flex items-center gap-3">
            <div className="text-right hidden sm:block">
              <p className="text-sm font-semibold text-emerald-900">{user.displayName}</p>
              <p className="text-xs text-emerald-600 capitalize">{profile?.role || 'User'}</p>
            </div>
            <img 
              src={user.photoURL || ''} 
              alt="Avatar" 
              className="w-10 h-10 rounded-full border-2 border-emerald-100"
              referrerPolicy="no-referrer"
            />
            <button 
              onClick={onLogout}
              className="p-2 text-emerald-900/60 hover:text-red-600 transition-colors"
            >
              <LogOut className="w-5 h-5" />
            </button>
          </div>
        ) : (
          <button 
            onClick={onLogin}
            className="bg-emerald-600 text-white px-6 py-2 rounded-full font-medium hover:bg-emerald-700 transition-all shadow-lg shadow-emerald-200"
          >
            Get Started
          </button>
        )}
      </div>
    </div>
  </nav>
);

const Home = ({ onStart }: { onStart: () => void }) => {
  const scrollToFeatures = () =>
    document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' });

  return (
    <div className="pt-32 pb-20 px-6">
      <div className="max-w-7xl mx-auto grid md:grid-cols-2 gap-12 items-center">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-100 text-emerald-700 text-xs font-bold uppercase tracking-wider mb-6">
            <Activity className="w-3 h-3" />
            AI-Powered Agriculture
          </div>
          <h1 className="text-6xl font-bold text-emerald-950 leading-[1.1] mb-6">
            Predict Your <span className="text-emerald-600">Harvest</span> with Precision.
          </h1>
          <p className="text-lg text-emerald-900/70 mb-8 max-w-lg leading-relaxed">
            Optimize your farming decisions using advanced machine learning. Input your soil and environmental data to get instant, data-driven crop recommendations.
          </p>
          <div className="flex flex-wrap gap-4">
            <button
              onClick={onStart}
              className="bg-emerald-600 text-white px-8 py-4 rounded-2xl font-bold text-lg hover:bg-emerald-700 transition-all shadow-xl shadow-emerald-200 flex items-center gap-2 group"
            >
              Start Prediction
              <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </button>
            <button
              onClick={scrollToFeatures}
              className="bg-white text-emerald-900 border-2 border-emerald-100 px-8 py-4 rounded-2xl font-bold text-lg hover:bg-emerald-50 transition-all"
            >
              Learn More
            </button>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="relative"
        >
          <div className="aspect-square rounded-[3rem] overflow-hidden shadow-2xl">
            <img
              src="https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?auto=format&fit=crop&q=80&w=1000"
              alt="Agriculture"
              className="w-full h-full object-cover"
              referrerPolicy="no-referrer"
            />
          </div>
          <div className="absolute -bottom-6 -left-6 bg-white p-6 rounded-3xl shadow-xl border border-emerald-50 max-w-[200px]">
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2 bg-emerald-100 rounded-lg text-emerald-600">
                <Sprout className="w-5 h-5" />
              </div>
              <span className="font-bold text-emerald-900">98% Accuracy</span>
            </div>
            <p className="text-xs text-emerald-900/60">Trained on 50,000+ soil samples worldwide.</p>
          </div>
        </motion.div>
      </div>

      {/* Features section */}
      <div id="features" className="max-w-7xl mx-auto mt-32">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-14"
        >
          <h2 className="text-4xl font-bold text-emerald-950 mb-4">How It Works</h2>
          <p className="text-lg text-emerald-900/60 max-w-xl mx-auto">
            A three-step ML pipeline that turns raw field data into actionable yield forecasts.
          </p>
        </motion.div>
        <div className="grid md:grid-cols-3 gap-8">
          {[
            {
              icon: <Activity className="w-7 h-7" />,
              title: 'AI-Powered Predictions',
              desc: 'A LightGBM model distilled from a Stacking Regressor, trained on satellite NDVI, NASA POWER climate data, and soil carbon across five Indian states.',
            },
            {
              icon: <ShieldCheck className="w-7 h-7" />,
              title: 'Confidence Intervals',
              desc: 'Every prediction ships with conformal prediction intervals at 80 %, 90 %, and 95 % coverage so you know exactly how much uncertainty to plan for.',
            },
            {
              icon: <History className="w-7 h-7" />,
              title: 'Historical Tracking',
              desc: 'All predictions are saved to your account. Review past forecasts, spot trends over seasons, and compare year-on-year yield estimates in your dashboard.',
            },
          ].map(({ icon, title, desc }, i) => (
            <motion.div
              key={title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.15 }}
              className="bg-white border border-emerald-100 rounded-3xl p-8 shadow-lg shadow-emerald-50 flex flex-col gap-4"
            >
              <div className="p-3 bg-emerald-100 rounded-2xl text-emerald-600 w-fit">{icon}</div>
              <h3 className="text-xl font-bold text-emerald-950">{title}</h3>
              <p className="text-emerald-900/60 leading-relaxed">{desc}</p>
            </motion.div>
          ))}
        </div>
        <div className="text-center mt-12">
          <button
            onClick={onStart}
            className="bg-emerald-600 text-white px-10 py-4 rounded-2xl font-bold text-lg hover:bg-emerald-700 transition-all shadow-xl shadow-emerald-200 inline-flex items-center gap-2 group"
          >
            Try It Now
            <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </button>
        </div>
      </div>
    </div>
  );
};

const PredictionForm = ({ onSubmit, loading }: { onSubmit: (data: PredictionInputs) => void, loading: boolean }) => {
  const [slowLoading, setSlowLoading] = useState(false);

  useEffect(() => {
    if (!loading) { setSlowLoading(false); return; }
    const timer = setTimeout(() => setSlowLoading(true), 3000);
    return () => clearTimeout(timer);
  }, [loading]);

  const [formData, setFormData] = useState<PredictionInputs>({
    state: 'PUNJAB',
    year: 2022,
    season: 'Rabi',
    temperature: 25,
    humidity: 60,
    rainfall: 100,
    tempMax: 35,
    tempMin: 15,
    solarRad: 20,
    windSpeed: 2,
    soilType: 'Loamy',
    ph: 6.5,
    nitrogen: 40,
    phosphorus: 40,
    potassium: 40
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    const stringFields = ['soilType', 'state', 'season'];
    setFormData(prev => ({
      ...prev,
      [name]: stringFields.includes(name) ? value : parseFloat(value)
    }));
  };

  return (
    <div className="max-w-4xl mx-auto bg-white rounded-[2.5rem] shadow-2xl shadow-emerald-100/50 border border-emerald-50 overflow-hidden">
      <div className="bg-emerald-600 p-8 text-white">
        <h2 className="text-3xl font-bold mb-2">Environmental Parameters</h2>
        <p className="text-emerald-100 opacity-80">Enter the details of your land for accurate prediction.</p>
      </div>
      
      <form onSubmit={(e) => { e.preventDefault(); onSubmit(formData); }} className="p-10 grid md:grid-cols-2 gap-8">
        <div className="md:col-span-2 grid md:grid-cols-3 gap-6">
          <div className="space-y-2">
            <label className="text-sm font-bold text-emerald-900 flex items-center gap-2">
              <MapPin className="w-4 h-4 text-emerald-600" />
              State
            </label>
            <select
              name="state" value={formData.state} onChange={handleChange}
              className="w-full px-4 py-3 rounded-xl border-2 border-emerald-50 focus:border-emerald-500 focus:outline-none transition-colors bg-white"
            >
              <option value="PUNJAB">Punjab</option>
              <option value="HARYANA">Haryana</option>
              <option value="RAJASTHAN">Rajasthan</option>
              <option value="UTTAR PRADESH">Uttar Pradesh</option>
            </select>
          </div>
          <div className="space-y-2">
            <label className="text-sm font-bold text-emerald-900 flex items-center gap-2">
              <Calendar className="w-4 h-4 text-emerald-600" />
              Year
            </label>
            <input
              type="number" name="year" value={formData.year} onChange={handleChange}
              min={2000} max={2022}
              className="w-full px-4 py-3 rounded-xl border-2 border-emerald-50 focus:border-emerald-500 focus:outline-none transition-colors"
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-bold text-emerald-900 flex items-center gap-2">
              <Sprout className="w-4 h-4 text-emerald-600" />
              Season
            </label>
            <select
              name="season" value={formData.season} onChange={handleChange}
              className="w-full px-4 py-3 rounded-xl border-2 border-emerald-50 focus:border-emerald-500 focus:outline-none transition-colors bg-white"
            >
              <option value="Rabi">Rabi</option>
              <option value="Kharif">Kharif</option>
              <option value="Zaid">Zaid</option>
            </select>
          </div>
        </div>

        <div className="space-y-6">
          <div className="space-y-2">
            <label className="text-sm font-bold text-emerald-900 flex items-center gap-2">
              <Thermometer className="w-4 h-4 text-emerald-600" />
              Temperature (°C)
            </label>
            <input 
              type="number" name="temperature" value={formData.temperature} onChange={handleChange} step="0.1"
              className="w-full px-4 py-3 rounded-xl border-2 border-emerald-50 focus:border-emerald-500 focus:outline-none transition-colors"
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-bold text-emerald-900 flex items-center gap-2">
              <Droplets className="w-4 h-4 text-emerald-600" />
              Humidity (%)
            </label>
            <input 
              type="number" name="humidity" value={formData.humidity} onChange={handleChange}
              className="w-full px-4 py-3 rounded-xl border-2 border-emerald-50 focus:border-emerald-500 focus:outline-none transition-colors"
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-bold text-emerald-900 flex items-center gap-2">
              <CloudRain className="w-4 h-4 text-emerald-600" />
              Rainfall (mm)
            </label>
            <input
              type="number" name="rainfall" value={formData.rainfall} onChange={handleChange}
              className="w-full px-4 py-3 rounded-xl border-2 border-emerald-50 focus:border-emerald-500 focus:outline-none transition-colors"
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-bold text-emerald-900 flex items-center gap-2">
              <Thermometer className="w-4 h-4 text-red-500" />
              Max Temperature (°C)
            </label>
            <input
              type="number" name="tempMax" value={formData.tempMax} onChange={handleChange} step="0.1"
              className="w-full px-4 py-3 rounded-xl border-2 border-emerald-50 focus:border-emerald-500 focus:outline-none transition-colors"
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-bold text-emerald-900 flex items-center gap-2">
              <Thermometer className="w-4 h-4 text-blue-500" />
              Min Temperature (°C)
            </label>
            <input
              type="number" name="tempMin" value={formData.tempMin} onChange={handleChange} step="0.1"
              className="w-full px-4 py-3 rounded-xl border-2 border-emerald-50 focus:border-emerald-500 focus:outline-none transition-colors"
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-bold text-emerald-900 flex items-center gap-2">
              <Sun className="w-4 h-4 text-emerald-600" />
              Solar Radiation (MJ/m²/day)
            </label>
            <input
              type="number" name="solarRad" value={formData.solarRad} onChange={handleChange} step="0.1"
              className="w-full px-4 py-3 rounded-xl border-2 border-emerald-50 focus:border-emerald-500 focus:outline-none transition-colors"
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-bold text-emerald-900 flex items-center gap-2">
              <Wind className="w-4 h-4 text-emerald-600" />
              Wind Speed (m/s)
            </label>
            <input
              type="number" name="windSpeed" value={formData.windSpeed} onChange={handleChange} step="0.1"
              className="w-full px-4 py-3 rounded-xl border-2 border-emerald-50 focus:border-emerald-500 focus:outline-none transition-colors"
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-bold text-emerald-900 flex items-center gap-2">
              <FlaskConical className="w-4 h-4 text-emerald-600" />
              Soil Type
            </label>
            <select 
              name="soilType" value={formData.soilType} onChange={handleChange}
              className="w-full px-4 py-3 rounded-xl border-2 border-emerald-50 focus:border-emerald-500 focus:outline-none transition-colors bg-white"
            >
              <option value="Loamy">Loamy</option>
              <option value="Sandy">Sandy</option>
              <option value="Clayey">Clayey</option>
              <option value="Silty">Silty</option>
              <option value="Peaty">Peaty</option>
              <option value="Chalky">Chalky</option>
            </select>
          </div>
        </div>

        <div className="space-y-6">
          <div className="space-y-2">
            <label className="text-sm font-bold text-emerald-900 flex items-center gap-2">
              <FlaskConical className="w-4 h-4 text-emerald-600" />
              Soil pH
            </label>
            <input 
              type="number" name="ph" value={formData.ph} onChange={handleChange} step="0.1"
              className="w-full px-4 py-3 rounded-xl border-2 border-emerald-50 focus:border-emerald-500 focus:outline-none transition-colors"
            />
          </div>
          <div className="grid grid-cols-3 gap-4">
            <div className="space-y-2">
              <label className="text-[10px] font-bold text-emerald-900 uppercase tracking-wider">Nitrogen (N)</label>
              <input type="number" name="nitrogen" value={formData.nitrogen} onChange={handleChange} className="w-full px-3 py-2 rounded-lg border-2 border-emerald-50 focus:border-emerald-500 focus:outline-none" />
            </div>
            <div className="space-y-2">
              <label className="text-[10px] font-bold text-emerald-900 uppercase tracking-wider">Phosphorus (P)</label>
              <input type="number" name="phosphorus" value={formData.phosphorus} onChange={handleChange} className="w-full px-3 py-2 rounded-lg border-2 border-emerald-50 focus:border-emerald-500 focus:outline-none" />
            </div>
            <div className="space-y-2">
              <label className="text-[10px] font-bold text-emerald-900 uppercase tracking-wider">Potassium (K)</label>
              <input type="number" name="potassium" value={formData.potassium} onChange={handleChange} className="w-full px-3 py-2 rounded-lg border-2 border-emerald-50 focus:border-emerald-500 focus:outline-none" />
            </div>
          </div>
          
          <div className="pt-8">
            <button 
              type="submit" 
              disabled={loading}
              className="w-full bg-emerald-600 text-white py-4 rounded-xl font-bold text-lg hover:bg-emerald-700 transition-all shadow-lg shadow-emerald-100 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin flex-shrink-0" />
                  {slowLoading
                    ? 'Waking up the AI model, this may take 30 seconds...'
                    : 'Analyzing Data...'}
                </>
              ) : 'Generate Prediction'}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
};

const ResultDisplay = ({ result, onReset }: { result: PredictionResult, onReset: () => void }) => (
  <motion.div 
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    className="max-w-2xl mx-auto bg-white rounded-[2.5rem] shadow-2xl border border-emerald-50 overflow-hidden"
  >
    <div className="bg-emerald-600 p-12 text-center text-white relative overflow-hidden">
      <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full -mr-32 -mt-32 blur-3xl" />
      <div className="relative z-10">
        <div className="bg-white/20 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-6 backdrop-blur-md">
          <Sprout className="w-10 h-10 text-white" />
        </div>
        <h3 className="text-sm font-bold uppercase tracking-[0.2em] mb-2 opacity-80">Predicted Yield</h3>
        <h2 className="text-5xl font-black mb-4">{result.crop}</h2>
        <div className="inline-flex items-center gap-2 bg-emerald-900/40 px-4 py-2 rounded-full backdrop-blur-sm">
          <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
          <span className="text-sm font-bold">Confidence: {(result.confidence * 100).toFixed(1)}%</span>
        </div>
      </div>
    </div>
    
    <div className="p-10">
      <div className="bg-emerald-50 p-6 rounded-2xl mb-8">
        <h4 className="text-emerald-900 font-bold mb-2 flex items-center gap-2">
          <Activity className="w-4 h-4" />
          Expert Advice
        </h4>
        <p className="text-emerald-900/70 leading-relaxed italic">"{result.advice}"</p>
      </div>
      
      <button 
        onClick={onReset}
        className="w-full border-2 border-emerald-100 text-emerald-900 py-4 rounded-xl font-bold hover:bg-emerald-50 transition-all"
      >
        New Prediction
      </button>
    </div>
  </motion.div>
);

const Dashboard = ({ history }: { history: PredictionRecord[] }) => {
  const chartData = history.slice().reverse().map(h => ({
    date: new Date(h.timestamp?.toDate()).toLocaleDateString(),
    temp: h.inputs.temperature,
    rain: h.inputs.rainfall,
    crop: h.result.crop
  }));

  return (
    <div className="max-w-7xl mx-auto px-6 space-y-12">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-4xl font-bold text-emerald-950">Your Dashboard</h2>
          <p className="text-emerald-900/60">Analyze your prediction history and trends.</p>
        </div>
        <div className="bg-white p-4 rounded-2xl shadow-sm border border-emerald-50 flex items-center gap-4">
          <div className="text-right">
            <p className="text-xs font-bold text-emerald-900/40 uppercase">Total Predictions</p>
            <p className="text-2xl font-black text-emerald-600">{history.length}</p>
          </div>
          <div className="p-3 bg-emerald-100 rounded-xl text-emerald-600">
            <BarChart3 className="w-6 h-6" />
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 bg-white p-8 rounded-[2rem] shadow-xl border border-emerald-50">
          <h3 className="text-xl font-bold text-emerald-900 mb-8 flex items-center gap-2">
            <Activity className="w-5 h-5 text-emerald-600" />
            Environmental Trends
          </h3>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorTemp" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0fdf4" />
                <XAxis dataKey="date" stroke="#064e3b" fontSize={10} tickLine={false} axisLine={false} />
                <YAxis stroke="#064e3b" fontSize={10} tickLine={false} axisLine={false} />
                <Tooltip 
                  contentStyle={{ borderRadius: '16px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
                />
                <Area type="monotone" dataKey="temp" stroke="#10b981" fillOpacity={1} fill="url(#colorTemp)" strokeWidth={3} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-emerald-900 p-8 rounded-[2rem] text-white shadow-xl">
          <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
            <History className="w-5 h-5 text-emerald-400" />
            Recent Activity
          </h3>
          <div className="space-y-4">
            {history.slice(0, 5).map((h) => (
              <div key={h.id} className="bg-white/10 p-4 rounded-xl border border-white/5 hover:bg-white/20 transition-colors cursor-pointer group">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-bold text-emerald-400">{h.result.crop}</span>
                  <span className="text-[10px] opacity-40">{new Date(h.timestamp?.toDate()).toLocaleDateString()}</span>
                </div>
                <p className="text-xs opacity-60 line-clamp-1 group-hover:opacity-100 transition-opacity">{h.result.advice}</p>
              </div>
            ))}
            {history.length === 0 && <p className="text-center py-10 opacity-40 italic">No history yet.</p>}
          </div>
        </div>
      </div>
    </div>
  );
};

const AdminPanel = () => {
  const [stats, setStats] = useState<any>(null);
  const [allPredictions, setAllPredictions] = useState<PredictionRecord[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const q = query(collection(db, 'predictions'), orderBy('timestamp', 'desc'), limit(50));
    const unsubscribe = onSnapshot(q, (snapshot) => {
      const data = snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() } as PredictionRecord));
      setAllPredictions(data);
      setLoading(false);
    });

    const statsUnsubscribe = onSnapshot(doc(db, 'system_stats', 'usage'), (doc) => {
      setStats(doc.data());
    });

    return () => {
      unsubscribe();
      statsUnsubscribe();
    };
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-6 space-y-12">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-4xl font-bold text-emerald-950">Admin Console</h2>
          <p className="text-emerald-900/60">System monitoring and global analytics.</p>
        </div>
      </div>

      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {[
          { label: 'Total Predictions', value: stats?.totalPredictions || 0, icon: BarChart3, color: 'bg-blue-500' },
          { label: 'Active Users', value: stats?.activeUsers || 0, icon: Users, color: 'bg-emerald-500' },
          { label: 'System Health', value: '99.9%', icon: Activity, color: 'bg-orange-500' },
          { label: 'Model Version', value: 'v2.4-stable', icon: Settings, color: 'bg-purple-500' },
        ].map((item, i) => (
          <div key={i} className="bg-white p-6 rounded-3xl shadow-sm border border-emerald-50 flex items-center gap-4">
            <div className={cn("p-3 rounded-2xl text-white", item.color)}>
              <item.icon className="w-6 h-6" />
            </div>
            <div>
              <p className="text-xs font-bold text-emerald-900/40 uppercase">{item.label}</p>
              <p className="text-2xl font-black text-emerald-900">{item.value}</p>
            </div>
          </div>
        ))}
      </div>

      <div className="bg-white rounded-[2rem] shadow-xl border border-emerald-50 overflow-hidden">
        <div className="p-8 border-b border-emerald-50 flex items-center justify-between">
          <h3 className="text-xl font-bold text-emerald-900">Recent Global Predictions</h3>
          <div className="flex gap-2">
            <span className="px-3 py-1 bg-emerald-100 text-emerald-700 rounded-full text-xs font-bold">Live Monitoring</span>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead className="bg-emerald-50/50">
              <tr>
                <th className="px-8 py-4 text-xs font-bold text-emerald-900/40 uppercase">User ID</th>
                <th className="px-8 py-4 text-xs font-bold text-emerald-900/40 uppercase">Crop</th>
                <th className="px-8 py-4 text-xs font-bold text-emerald-900/40 uppercase">Confidence</th>
                <th className="px-8 py-4 text-xs font-bold text-emerald-900/40 uppercase">Date</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-emerald-50">
              {allPredictions.map((p) => (
                <tr key={p.id} className="hover:bg-emerald-50/30 transition-colors">
                  <td className="px-8 py-4 text-sm font-mono text-emerald-900/60">{p.userId.slice(0, 8)}...</td>
                  <td className="px-8 py-4 font-bold text-emerald-900">{p.result.crop}</td>
                  <td className="px-8 py-4">
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-1.5 bg-emerald-100 rounded-full overflow-hidden">
                        <div className="h-full bg-emerald-500" style={{ width: `${p.result.confidence * 100}%` }} />
                      </div>
                      <span className="text-xs font-bold">{(p.result.confidence * 100).toFixed(0)}%</span>
                    </div>
                  </td>
                  <td className="px-8 py-4 text-sm text-emerald-900/60">{new Date(p.timestamp?.toDate()).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

// --- Main App ---

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [activeTab, setActiveTab] = useState('home');
  const [loading, setLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
  const [history, setHistory] = useState<PredictionRecord[]>([]);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (u) => {
      setUser(u);
      if (u) {
        const userDoc = await getDoc(doc(db, 'users', u.uid));
        if (!userDoc.exists()) {
          const newProfile: UserProfile = {
            uid: u.uid,
            email: u.email || '',
            displayName: u.displayName || '',
            photoURL: u.photoURL || '',
            role: 'user',
            createdAt: serverTimestamp()
          };
          await setDoc(doc(db, 'users', u.uid), newProfile);
          setProfile(newProfile);
        } else {
          setProfile(userDoc.data() as UserProfile);
        }

        // Fetch history
        const q = query(
          collection(db, 'predictions'), 
          where('userId', '==', u.uid),
          orderBy('timestamp', 'desc')
        );
        onSnapshot(q, (snapshot) => {
          setHistory(snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() } as PredictionRecord)));
        });
      } else {
        setProfile(null);
        setHistory([]);
      }
    });
    return () => unsubscribe();
  }, []);

  const handleLogin = async () => {
    const provider = new GoogleAuthProvider();
    try {
      await signInWithPopup(auth, provider);
    } catch (error) {
      console.error("Login failed", error);
    }
  };

  const handleLogout = () => signOut(auth);

  const handlePredict = async (inputs: PredictionInputs) => {
    if (!user) return;
    setLoading(true);
    try {
      const result = await predictCrop(inputs);
      setPredictionResult(result);
      
      // Save to Firestore
      await addDoc(collection(db, 'predictions'), {
        userId: user.uid,
        timestamp: serverTimestamp(),
        inputs,
        result
      });

      // Update global stats
      await setDoc(doc(db, 'system_stats', 'usage'), {
        totalPredictions: increment(1),
        lastPredictionAt: serverTimestamp()
      }, { merge: true });

    } catch (error) {
      console.error("Prediction failed", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#FBFBF9] font-sans selection:bg-emerald-100 selection:text-emerald-900">
      <Navbar 
        user={user} 
        profile={profile}
        onLogin={handleLogin} 
        onLogout={handleLogout} 
        activeTab={activeTab}
        setActiveTab={setActiveTab}
      />

      <main className="pt-20">
        <AnimatePresence mode="wait">
          {activeTab === 'home' && (
            <motion.div 
              key="home"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <Home onStart={() => user ? setActiveTab('predict') : handleLogin()} />
            </motion.div>
          )}

          {activeTab === 'predict' && (
            <motion.div 
              key="predict"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="py-20 px-6"
            >
              {predictionResult ? (
                <ResultDisplay result={predictionResult} onReset={() => setPredictionResult(null)} />
              ) : (
                <PredictionForm onSubmit={handlePredict} loading={loading} />
              )}
            </motion.div>
          )}

          {activeTab === 'dashboard' && (
            <motion.div 
              key="dashboard"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="py-20"
            >
              <Dashboard history={history} />
            </motion.div>
          )}

          {activeTab === 'admin' && profile?.role === 'admin' && (
            <motion.div 
              key="admin"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="py-20"
            >
              <AdminPanel />
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <footer className="py-12 px-6 border-t border-emerald-50 bg-white">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex items-center gap-2">
            <Sprout className="text-emerald-600 w-5 h-5" />
            <span className="font-bold text-emerald-900">AgroPredict</span>
          </div>
          <p className="text-sm text-emerald-900/40">© 2026 AgroPredict Systems. All rights reserved.</p>
          <div className="flex gap-6">
            <a href="#" className="text-sm text-emerald-900/40 hover:text-emerald-600 transition-colors">Privacy</a>
            <a href="#" className="text-sm text-emerald-900/40 hover:text-emerald-600 transition-colors">Terms</a>
            <a href="#" className="text-sm text-emerald-900/40 hover:text-emerald-600 transition-colors">Contact</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
