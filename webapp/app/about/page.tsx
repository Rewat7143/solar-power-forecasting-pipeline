import Image from "next/image";

export default function AboutPage() {
  return (
    <div className="page-frame">
      <div className="mb-12 text-center">
        <span className="eyebrow">Behind the Project</span>
        <h1 className="text-4xl font-bold text-[#1E293B]">About SolarCast</h1>
      </div>

      <div className="max-w-4xl mx-auto space-y-12">
        
        {/* The Team / Author */}
        <section className="card-panel">
          <h2 className="text-2xl font-bold mb-4 text-[#1E3A8A] border-b border-slate-200 pb-2">The Author</h2>
          <p className="text-lg text-slate-700 leading-relaxed">
            This project was developed by <strong>Rewat</strong>. It bridges the gap between advanced deep learning 
            for time-series forecasting and modern web application development, providing a highly scalable 
            and interactive way to visualize AI predictions for renewable energy systems.
          </p>
        </section>

        {/* Architecture */}
        <section className="card-panel">
          <h2 className="text-2xl font-bold mb-6 text-[#1E293B] border-b border-slate-200 pb-2">System Architecture</h2>
          <div className="relative max-w-2xl mx-auto h-[300px] rounded border border-slate-200 overflow-hidden bg-slate-50 mb-8 p-4 shadow-sm">
            <Image 
              src="/research/9_system_architecture_diagram.png" 
              alt="Architecture Diagram"
              fill
              className="object-contain"
            />
          </div>
          <p className="text-slate-700 leading-relaxed text-lg">
            The pipeline is built with a decoupled architecture. The backend consists of a comprehensive Python data 
            science pipeline that ingests NASA POWER weather data, engineers 26 complex features (including rolling statistics 
            and sinusoidal temporal embeddings), and trains 7 different models ranging from Scikit-Learn Random Forests 
            to PyTorch-based Transformers (PatchTST, TFT). 
            <br/><br/>
            The frontend is a Next.js 14 application that communicates with the Python inference engine via child processes, 
            allowing for real-time model evaluation with a polished, interactive UI utilizing Recharts.
          </p>
        </section>

        {/* Tech Stack */}
        <section className="card-panel">
          <h2 className="text-2xl font-bold mb-6 text-[#1E293B] border-b border-slate-200 pb-2">Technology Stack</h2>
          <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-6">
            <div className="bg-slate-50 p-5 rounded-lg border border-slate-200 shadow-sm">
              <div className="font-bold text-[#1E3A8A] mb-1">Next.js 14</div>
              <div className="text-sm text-slate-600 font-medium">React Framework</div>
            </div>
            <div className="bg-slate-50 p-5 rounded-lg border border-slate-200 shadow-sm">
              <div className="font-bold text-[#1E3A8A] mb-1">Recharts</div>
              <div className="text-sm text-slate-600 font-medium">Data Visualization</div>
            </div>
            <div className="bg-slate-50 p-5 rounded-lg border border-slate-200 shadow-sm">
              <div className="font-bold text-[#1E3A8A] mb-1">Tailwind CSS</div>
              <div className="text-sm text-slate-600 font-medium">Styling System</div>
            </div>
            <div className="bg-slate-50 p-5 rounded-lg border border-slate-200 shadow-sm">
              <div className="font-bold text-[#D97706] mb-1">Python 3</div>
              <div className="text-sm text-slate-600 font-medium">Data Science Backend</div>
            </div>
            <div className="bg-slate-50 p-5 rounded-lg border border-slate-200 shadow-sm">
              <div className="font-bold text-[#D97706] mb-1">PyTorch</div>
              <div className="text-sm text-slate-600 font-medium">Deep Learning Models</div>
            </div>
            <div className="bg-slate-50 p-5 rounded-lg border border-slate-200 shadow-sm">
              <div className="font-bold text-[#D97706] mb-1">Scikit-Learn</div>
              <div className="text-sm text-slate-600 font-medium">Tree Models &amp; Ensembles</div>
            </div>
          </div>
        </section>

      </div>
    </div>
  );
}
