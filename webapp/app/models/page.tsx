"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';

export default function ModelsPage() {
  const metrics = [
    { name: "Stacked Ensemble", mae: 10.18, rmse: 15.68, r2: 0.87 },
    { name: "Random Forest", mae: 10.49, rmse: 15.77, r2: 0.87 },
    { name: "PatchTST", mae: 16.99, rmse: 24.32, r2: 0.69 },
    { name: "TFT", mae: 34.02, rmse: 39.24, r2: 0.21 },
    { name: "CNN + Transformer", mae: 42.61, rmse: 52.28, r2: -0.38 },
    { name: "CNN + LSTM", mae: 53.00, rmse: 65.01, r2: -1.14 },
  ];

  return (
    <div className="page-frame">
      <div className="mb-12">
        <span className="eyebrow">Performance Metrics</span>
        <h1 className="text-4xl font-bold text-[#1E293B]">Model Leaderboard</h1>
        <p className="text-slate-600 mt-2 text-lg">
          Evaluation of all 7 models across a 24-hour recursive daylight backtest.
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-8 mb-12">
        <div className="card-panel">
          <h2 className="text-xl font-bold mb-6 text-[#1E293B]">Error Comparison (MAE vs RMSE)</h2>
          <div className="h-[300px] md:h-[350px]">
            {/* @ts-ignore */}
            <ResponsiveContainer width="100%" height="100%">
              {/* @ts-ignore */}
              <BarChart data={metrics} margin={{ top: 20, right: 30, left: 0, bottom: 50 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />
                <XAxis dataKey="name" tick={{ fill: "#64748B", fontSize: 11, fontWeight: 500 }} angle={-45} textAnchor="end" />
                <YAxis tick={{ fill: "#64748B", fontWeight: 500 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#FFFFFF', borderColor: '#E2E8F0', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                  itemStyle={{ color: '#1E293B', fontWeight: 600 }}
                />
                <Legend verticalAlign="top" height={36} wrapperStyle={{ fontWeight: 500, color: '#475569' }} />
                <Bar dataKey="mae" name="MAE (kW)" fill="#1E3A8A" radius={[4, 4, 0, 0]} />
                <Bar dataKey="rmse" name="RMSE (kW)" fill="#93C5FD" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card-panel flex flex-col justify-center">
          <h2 className="text-xl font-bold mb-4 text-[#1E293B]">Stacked Ensemble Weights</h2>
          <p className="text-slate-600 mb-6 leading-relaxed">
            The final prediction utilizes a non-negative linear stacking approach trained on holdout data. 
            The Random Forest model dominates due to its superior stability across varying weather regimes, while PatchTST captures complex sequential dependencies.
          </p>
          <div className="space-y-5 bg-slate-50 p-6 rounded-lg border border-slate-200">
            <div>
              <div className="flex justify-between mb-2 text-sm font-semibold text-slate-700">
                <span>Random Forest</span>
                <span>98.7%</span>
              </div>
              <div className="w-full bg-slate-200 rounded-full h-2.5">
                <div className="bg-[#1E3A8A] h-2.5 rounded-full" style={{ width: '98.7%' }}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between mb-2 text-sm font-semibold text-slate-700">
                <span>PatchTST</span>
                <span>5.1%</span>
              </div>
              <div className="w-full bg-slate-200 rounded-full h-2.5">
                <div className="bg-[#D97706] h-2.5 rounded-full" style={{ width: '5.1%' }}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between mb-2 text-sm font-semibold text-slate-400">
                <span>Other Deep Models</span>
                <span>0.0%</span>
              </div>
              <div className="w-full bg-slate-200 rounded-full h-2.5">
                <div className="bg-slate-300 h-2.5 rounded-full" style={{ width: '0%' }}></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="card-panel">
        <h2 className="text-xl font-bold mb-6 text-[#1E293B]">Detailed Leaderboard</h2>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Model</th>
                <th>MAE (kW)</th>
                <th>RMSE (kW)</th>
                <th>R² Score</th>
                <th>Family</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((m, idx) => (
                <tr key={m.name} className={idx === 0 ? "row-highlight" : ""}>
                  <td className="font-bold text-slate-500">#{idx + 1}</td>
                  <td className="font-bold text-[#1E293B]">{m.name}</td>
                  <td className="mono font-medium text-slate-700">{m.mae.toFixed(2)}</td>
                  <td className="mono font-medium text-slate-700">{m.rmse.toFixed(2)}</td>
                  <td className="mono font-medium text-slate-700">{m.r2.toFixed(3)}</td>
                  <td>
                    <span className={`status-pill ${
                      m.name.includes("Ensemble") ? "bg-blue-100 text-blue-800 border-blue-200" : 
                      m.name.includes("Forest") ? "bg-amber-100 text-amber-800 border-amber-200" : 
                      "bg-purple-100 text-purple-800 border-purple-200"
                    }`}>
                      {m.name.includes("Ensemble") ? "Meta-Model" : m.name.includes("Forest") ? "Tree" : "Sequence"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
