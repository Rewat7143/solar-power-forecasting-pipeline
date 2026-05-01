import Link from "next/link";
import Image from "next/image";

export default function LandingPage() {
  return (
    <div className="flex flex-col min-h-screen bg-white">
      {/* Hero Section */}
      <section className="relative pt-20 pb-24 px-6 lg:px-24 bg-gradient-to-b from-[#F8FAFC] to-white overflow-hidden">
        {/* Decorative Background Elements */}
        <div className="absolute top-0 left-0 w-full h-full overflow-hidden z-0 opacity-40 pointer-events-none">
          <div className="absolute -top-[10%] -right-[5%] w-[500px] h-[500px] rounded-full bg-blue-100 blur-3xl"></div>
          <div className="absolute top-[40%] -left-[10%] w-[300px] h-[300px] rounded-full bg-amber-50 blur-3xl"></div>
        </div>

        <div className="max-w-7xl mx-auto flex flex-col lg:flex-row items-center gap-16 relative z-10">
          
          {/* Left Text Content */}
          <div className="flex-1 space-y-8 text-center lg:text-left">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-50 border border-blue-200 text-blue-800 font-bold text-xs tracking-wider uppercase shadow-sm">
              <span className="w-2 h-2 rounded-full bg-blue-600 animate-pulse"></span>
              Academic Research Platform
            </div>
            
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-extrabold tracking-tight text-[#1E293B] leading-[1.15]">
              Next-Generation <br className="hidden lg:block" />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#1E3A8A] to-[#3B82F6]">
                Solar Forecasting
              </span>
            </h1>
            
            <p className="text-lg md:text-xl text-slate-600 leading-relaxed max-w-2xl mx-auto lg:mx-0">
              An advanced machine learning pipeline evaluating the efficacy of deep sequence models 
              against traditional tree-based architectures using high-frequency telemetry and NASA satellite data.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 pt-2 justify-center lg:justify-start">
              <Link href="/dashboard" className="btn-primary shadow-lg shadow-blue-900/20 text-lg px-8 py-4">
                Access Live Predictor &rarr;
              </Link>
              <Link href="/research" className="btn-secondary text-lg px-8 py-4 bg-white">
                View Visualizations
              </Link>
            </div>
          </div>

          {/* Right Hero Graphic */}
          <div className="w-full max-w-lg lg:max-w-xl mx-auto perspective-1000 mt-8 lg:mt-0">
            <div className="relative rounded-2xl bg-white border border-slate-200 shadow-xl overflow-hidden transform transition-transform hover:scale-[1.02] duration-500">
              <div className="absolute top-0 left-0 w-full h-10 bg-slate-50 border-b border-slate-200 flex items-center px-4 gap-2">
                <div className="w-3 h-3 rounded-full bg-red-400"></div>
                <div className="w-3 h-3 rounded-full bg-amber-400"></div>
                <div className="w-3 h-3 rounded-full bg-green-400"></div>
                <div className="mx-auto text-xs font-mono text-slate-400 font-semibold uppercase tracking-wider">System Output.png</div>
              </div>
              <div className="pt-10 relative aspect-[4/3] w-full">
                <Image 
                  src="/research/3_future_forecast_next_24h.png" 
                  alt="Solar Forecast Curve Visualization"
                  fill
                  className="object-contain p-4"
                  priority
                />
              </div>
            </div>
          </div>

        </div>
      </section>

      {/* Stats Strip */}
      <section className="bg-[#1E3A8A] text-white py-12 relative z-20 shadow-xl">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 md:gap-8 text-center divide-x divide-blue-400/30">
            <div className="px-4">
              <div className="text-4xl md:text-5xl font-extrabold mb-2 tracking-tight">7</div>
              <div className="text-[10px] md:text-xs uppercase tracking-widest text-blue-200 font-bold">Trained Models</div>
            </div>
            <div className="px-4">
              <div className="text-4xl md:text-5xl font-extrabold mb-2 tracking-tight text-[#FBBF24]">0.76</div>
              <div className="text-[10px] md:text-xs uppercase tracking-widest text-blue-200 font-bold">Ensemble R&sup2; Score</div>
            </div>
            <div className="px-4">
              <div className="text-4xl md:text-5xl font-extrabold mb-2 tracking-tight text-[#93C5FD]">5-Min</div>
              <div className="text-[10px] md:text-xs uppercase tracking-widest text-blue-200 font-bold">Data Resolution</div>
            </div>
            <div className="px-4">
              <div className="text-4xl md:text-5xl font-extrabold mb-2 tracking-tight text-[#C084FC]">24h</div>
              <div className="text-[10px] md:text-xs uppercase tracking-widest text-blue-200 font-bold">Forecast Horizon</div>
            </div>
          </div>
        </div>
      </section>

      {/* Methodology Overview */}
      <section className="py-24 px-6 lg:px-24 bg-white relative">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-20">
            <span className="eyebrow text-[#1E3A8A]">System Architecture</span>
            <h2 className="text-3xl md:text-5xl font-extrabold mt-2 text-[#1E293B]">Research Methodology</h2>
            <div className="w-24 h-1.5 bg-[#D97706] mx-auto mt-8 rounded-full"></div>
            <p className="mt-6 text-slate-600 max-w-2xl mx-auto text-lg">
              A fully decoupled pipeline spanning from raw data ingestion to comparative real-time inference using state-of-the-art architectures.
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-10">
            {/* Step 1 */}
            <div className="card-panel text-center relative group hover:-translate-y-2 transition-transform duration-300">
              <div className="absolute -top-6 left-1/2 -translate-x-1/2 w-12 h-12 bg-[#1E3A8A] text-white rounded-full flex items-center justify-center font-bold text-xl border-4 border-white shadow-sm">1</div>
              <div className="w-20 h-20 rounded-2xl bg-blue-50 flex items-center justify-center mx-auto mb-6 mt-4 group-hover:scale-110 transition-transform duration-300">
                <svg className="w-10 h-10 text-[#1E3A8A]" width="40" height="40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" />
                </svg>
              </div>
              <h3 className="text-2xl font-bold mb-4 text-[#1E293B]">Data Acquisition</h3>
              <p className="text-slate-600 text-base leading-relaxed">
                Integration of localized weather metrics (Irradiance, Cloud Cover, Temperature) queried directly from NASA POWER APIs alongside high-frequency power telemetry.
              </p>
            </div>
            
            {/* Step 2 */}
            <div className="card-panel text-center relative group hover:-translate-y-2 transition-transform duration-300">
              <div className="absolute -top-6 left-1/2 -translate-x-1/2 w-12 h-12 bg-[#D97706] text-white rounded-full flex items-center justify-center font-bold text-xl border-4 border-white shadow-sm">2</div>
              <div className="w-20 h-20 rounded-2xl bg-amber-50 flex items-center justify-center mx-auto mb-6 mt-4 group-hover:scale-110 transition-transform duration-300">
                <svg className="w-10 h-10 text-[#D97706]" width="40" height="40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h3 className="text-2xl font-bold mb-4 text-[#1E293B]">Feature Engineering</h3>
              <p className="text-slate-600 text-base leading-relaxed">
                Transformation of raw time-series into 26 distinct temporal, cyclical (sin/cos embeddings), and rolling-window statistical features.
              </p>
            </div>
            
            {/* Step 3 */}
            <div className="card-panel text-center relative group hover:-translate-y-2 transition-transform duration-300">
              <div className="absolute -top-6 left-1/2 -translate-x-1/2 w-12 h-12 bg-[#1E3A8A] text-white rounded-full flex items-center justify-center font-bold text-xl border-4 border-white shadow-sm">3</div>
              <div className="w-20 h-20 rounded-2xl bg-blue-50 flex items-center justify-center mx-auto mb-6 mt-4 group-hover:scale-110 transition-transform duration-300">
                <svg className="w-10 h-10 text-[#1E3A8A]" width="40" height="40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
              </div>
              <h3 className="text-2xl font-bold mb-4 text-[#1E293B]">Model Inference</h3>
              <p className="text-slate-600 text-base leading-relaxed">
                Comparative evaluation of deep learning architectures (PatchTST, Temporal Fusion Transformer) and non-negative linear stacking ensembles.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Architecture Graphic Banner */}
      <section className="bg-slate-50 border-y border-slate-200 overflow-hidden">
        <div className="max-w-6xl mx-auto px-6 py-16 flex flex-col lg:flex-row items-center justify-center gap-12">
          <div className="w-full max-w-xl mx-auto">
            <div className="relative h-[250px] md:h-[350px] w-full rounded-xl bg-white border border-slate-200 shadow-sm p-4">
              <Image 
                src="/research/9_system_architecture_diagram.png" 
                alt="Architecture Diagram"
                fill
                className="object-contain"
              />
            </div>
          </div>
          <div className="w-full max-w-lg text-center lg:text-left mx-auto">
            <h2 className="text-3xl font-bold text-[#1E293B]">Designed for Scalability</h2>
            <p className="text-slate-600 leading-relaxed text-lg">
              The SolarCast architecture seamlessly unifies a Python-based machine learning pipeline with a fast, edge-ready Next.js frontend, creating a robust system capable of handling complex multivariate time-series predictions.
            </p>
            <Link href="/about" className="text-[#1E3A8A] font-bold text-lg inline-flex items-center gap-2 hover:underline">
              Read more about the tech stack &rarr;
            </Link>
          </div>
        </div>
      </section>

      {/* Teaser CTA */}
      <section className="py-24 px-6 lg:px-24 bg-white">
        <div className="max-w-4xl mx-auto text-center card-panel bg-gradient-to-br from-white to-blue-50 border-blue-100">
          <div className="w-16 h-16 bg-[#1E3A8A] rounded-full flex items-center justify-center mx-auto mb-6 shadow-lg shadow-blue-900/20">
            <svg className="w-8 h-8 text-white" width="32" height="32" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <h2 className="text-4xl font-extrabold mb-6 text-[#1E293B]">Evaluate the Models Live</h2>
          <p className="text-slate-600 mb-10 max-w-2xl mx-auto text-xl leading-relaxed">
            Input custom timestamps to compare predicted irradiance curves, 
            confidence intervals, and meteorological contexts across all architectures in real time.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/dashboard" className="btn-primary shadow-lg shadow-blue-900/20 text-lg px-8 py-4">
              Launch Live Predictor
            </Link>
            <Link href="/models" className="btn-secondary text-lg px-8 py-4 bg-white">
              View Leaderboard
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
