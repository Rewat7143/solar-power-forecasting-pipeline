"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

export default function Navbar() {
  const pathname = usePathname();
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => setIsMenuOpen(!isMenuOpen);
  const closeMenu = () => setIsMenuOpen(false);

  const NAV_LINKS = [
    { name: "Home", href: "/" },
    { name: "Live Predictor", href: "/dashboard" },
    { name: "Leaderboard", href: "/models" },
    { name: "Visualizations", href: "/research" },
    { name: "About", href: "/about" },
  ];

  return (
    <nav className="navbar relative w-full">
      <div className="w-full max-w-[1200px] mx-auto flex justify-between items-center px-4 md:px-6">
        
        {/* Brand */}
        <Link href="/" className="nav-brand z-50 relative" onClick={closeMenu}>
          Solar<span>Cast</span> Research
        </Link>
        
        {/* Desktop Links */}
        <div className="hidden md:flex gap-8 items-center">
          {NAV_LINKS.map((link) => (
            <Link 
              key={link.name} 
              href={link.href} 
              className={`nav-link ${pathname === link.href ? "active" : ""}`}
            >
              {link.name}
            </Link>
          ))}
          <Link href="/dashboard" className="nav-cta ml-4">
            Access System &rarr;
          </Link>
        </div>

        {/* Mobile Toggle Button */}
        <button 
          className="md:hidden z-50 p-2 text-white focus:outline-none"
          onClick={toggleMenu}
          aria-label="Toggle menu"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            {isMenuOpen ? (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            ) : (
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            )}
          </svg>
        </button>

      </div>

      {/* Mobile Menu Dropdown */}
      <div className={`md:hidden absolute top-full left-0 w-full bg-[#1E3A8A] shadow-xl border-t border-blue-800 transition-all duration-300 ease-in-out origin-top ${isMenuOpen ? "opacity-100 scale-y-100" : "opacity-0 scale-y-0 pointer-events-none"}`}>
        <div className="flex flex-col py-4 px-6 space-y-4">
          {NAV_LINKS.map((link) => (
            <Link 
              key={link.name} 
              href={link.href} 
              onClick={closeMenu}
              className={`text-lg font-medium py-2 border-b border-blue-800/50 ${pathname === link.href ? "text-white" : "text-blue-200"}`}
            >
              {link.name}
            </Link>
          ))}
          <Link 
            href="/dashboard" 
            onClick={closeMenu}
            className="btn-primary mt-4 text-center justify-center bg-[#D97706] hover:bg-[#B45309]"
          >
            Access System &rarr;
          </Link>
        </div>
      </div>
    </nav>
  );
}
