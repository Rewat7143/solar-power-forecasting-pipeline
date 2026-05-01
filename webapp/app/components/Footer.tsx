import Link from "next/link";

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-content flex-col md:flex-row gap-4">
        <div className="text-center md:text-left">
          <strong className="text-slate-800">SolarCast Research Project</strong> <br className="md:hidden" /> 
          <span className="text-slate-500 text-sm">Developed by Rewat &copy; {new Date().getFullYear()}</span>
        </div>
        <div className="flex gap-4">
          <span className="status-chip neutral">Next.js</span>
          <span className="status-chip neutral">Python</span>
          <span className="status-chip neutral">PyTorch</span>
        </div>
        <div>
          <a href="https://github.com/Rewat7143/solar-power-forecasting-pipeline" target="_blank" rel="noopener noreferrer" className="font-medium text-[#1E3A8A] hover:underline">
            View Source on GitHub &rarr;
          </a>
        </div>
      </div>
    </footer>
  );
}
