import "./globals.css";
import { IBM_Plex_Mono, Space_Grotesk } from "next/font/google";
import type { Metadata } from "next";
import type { ReactNode } from "react";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";

export const metadata: Metadata = {
  title: "SolarCast | Academic Solar Forecasting",
  description: "Advanced machine learning pipeline evaluating deep sequence models for high-frequency solar power forecasting.",
};

const spaceGrotesk = Space_Grotesk({ subsets: ["latin"], variable: "--font-sans" });
const ibmPlexMono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "500", "600", "700"], variable: "--font-mono" });

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${spaceGrotesk.variable} ${ibmPlexMono.variable}`}>
      <body>
        <div className="main-content">
          <Navbar />
          {children}
          <Footer />
        </div>
      </body>
    </html>
  );
}
