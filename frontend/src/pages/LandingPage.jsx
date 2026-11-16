import React, { useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Star, ArrowRight, Telescope, Cpu, BarChart3, Orbit, ChevronDown } from 'lucide-react'

const CLASS_CARDS = [
  {
    id: 0, name: 'Exoplanet Transit', color: '#00f2fe',
    icon: '🪐',
    desc: 'Shallow, symmetric U-shaped periodic dimming caused by an orbiting planet crossing the stellar disk.'
  },
  {
    id: 1, name: 'Eclipsing Binary', color: '#ff007f',
    icon: '⭐',
    desc: 'Deep V-shaped primary transit with secondary eclipses caused by two mutually orbiting stars.'
  },
  {
    id: 2, name: 'Background Blend', color: '#ffd166',
    icon: '🌌',
    desc: 'Aperture contamination where an unresolved background binary mimics a transit signal.'
  },
  {
    id: 3, name: 'Stellar Activity', color: '#06d6a0',
    icon: '🌟',
    desc: 'Non-transit quasi-periodic signals from stellar rotation, starspots, flares, or instrumental noise.'
  },
]

const STEPS = [
  { icon: Telescope, label: 'Fetch TESS Data', desc: 'Query NASA MAST for public light curve observations' },
  { icon: BarChart3, label: 'Preprocess Signal', desc: 'Remove noise, detrend and normalise the photometric time-series' },
  { icon: Orbit, label: 'BLS Period Search', desc: 'Box Least Squares algorithm detects periodic transit-like signals' },
  { icon: Cpu, label: 'AI Classification', desc: 'Dual-Branch 1D-CNN classifies the signal into 4 astrophysical categories' },
]

export default function LandingPage() {
  const navigate = useNavigate()
  const heroRef = useRef(null)

  // Animate stars twinkling
  useEffect(() => {
    const canvas = document.getElementById('hero-canvas')
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    let frame = 0
    const stars = Array.from({ length: 160 }, () => ({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      r: Math.random() * 1.4 + 0.3,
      speed: Math.random() * 0.4 + 0.05,
      phase: Math.random() * Math.PI * 2
    }))

    const resize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resize()
    window.addEventListener('resize', resize)

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      stars.forEach(s => {
        const alpha = 0.3 + 0.5 * Math.sin(frame * s.speed + s.phase)
        ctx.beginPath()
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(255,255,255,${alpha})`
        ctx.fill()
      })
      frame++
      requestAnimationFrame(draw)
    }
    const raf = requestAnimationFrame(draw)
    return () => { window.removeEventListener('resize', resize); cancelAnimationFrame(raf) }
  }, [])

  return (
    <div style={{ background: '#060a12', minHeight: '100vh', overflowX: 'hidden', position: 'relative' }}>
      {/* Animated star canvas */}
      <canvas id="hero-canvas" style={{ position: 'fixed', inset: 0, zIndex: 0, pointerEvents: 'none' }} />

      {/* Navbar */}
      <nav style={{
        position: 'fixed', top: 0, left: 0, right: 0, zIndex: 50,
        background: 'rgba(6,10,18,0.85)', backdropFilter: 'blur(20px)',
        borderBottom: '1px solid rgba(0,242,254,0.1)', padding: '0 2rem', height: 64,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10,
            background: 'linear-gradient(135deg,#00f2fe,#9055ff)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 0 20px rgba(0,242,254,0.4)'
          }}>
            <Star size={18} fill="#060a12" color="#060a12" />
          </div>
          <span style={{
            fontWeight: 900, letterSpacing: '0.12em', fontSize: '1.05rem',
            background: 'linear-gradient(135deg,#00f2fe,#4facfe,#9055ff)',
            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent'
          }}>STARTRACE</span>
        </div>

        <div style={{ display: 'flex', gap: 24, alignItems: 'center' }}>
          <a href="#how-it-works" style={{ color: '#94a3b8', fontSize: '0.88rem', textDecoration: 'none', fontWeight: 500 }}>How It Works</a>
          <a href="#technology" style={{ color: '#94a3b8', fontSize: '0.88rem', textDecoration: 'none', fontWeight: 500 }}>Technology</a>
          <button className="btn-primary" style={{ padding: '8px 20px', fontSize: '0.85rem' }}
            onClick={() => navigate('/dashboard/analyze')}>
            Launch App <ArrowRight size={15} />
          </button>
        </div>
      </nav>

      {/* Hero */}
      <section ref={heroRef} style={{
        position: 'relative', zIndex: 1, minHeight: '100vh',
        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
        textAlign: 'center', padding: '100px 24px 60px'
      }}>
        {/* Orbital animation */}
        <div style={{ position: 'relative', width: 200, height: 200, marginBottom: 40 }}>
          {/* Central star */}
          <div style={{
            position: 'absolute', top: '50%', left: '50%',
            transform: 'translate(-50%,-50%)',
            width: 56, height: 56, borderRadius: '50%',
            background: 'radial-gradient(circle, #ffe580, #ffa500)',
            boxShadow: '0 0 40px rgba(255,200,80,0.6), 0 0 80px rgba(255,180,60,0.3)'
          }} />
          {/* Orbit ring */}
          <div style={{
            position: 'absolute', top: '50%', left: '50%',
            transform: 'translate(-50%,-50%)',
            width: 170, height: 170, borderRadius: '50%',
            border: '1.5px solid rgba(0,242,254,0.25)'
          }} />
          {/* Orbiting planet */}
          <div style={{
            position: 'absolute', top: '50%', left: '50%',
            width: 170, height: 170, marginTop: -85, marginLeft: -85,
            animation: 'hero-orbit 6s linear infinite'
          }}>
            <div style={{
              position: 'absolute', top: 0, left: '50%', transform: 'translateX(-50%)',
              width: 18, height: 18, borderRadius: '50%',
              background: 'linear-gradient(135deg,#00f2fe,#4facfe)',
              boxShadow: '0 0 14px rgba(0,242,254,0.7)'
            }} />
          </div>
        </div>

        <div style={{ marginBottom: 12, color: '#00f2fe', fontSize: '0.78rem', letterSpacing: '0.3em', textTransform: 'uppercase', fontWeight: 700 }}>
          AI-POWERED ASTRONOMICAL ANALYSIS
        </div>

        <h1 style={{
          fontSize: 'clamp(2.2rem, 6vw, 4.2rem)', fontWeight: 900, lineHeight: 1.1,
          marginBottom: 20, maxWidth: 820, letterSpacing: '-0.02em'
        }}>
          <span className="gradient-text">Discover Worlds</span>
          <br />
          <span style={{ color: '#e2e8f0' }}>Hidden in Starlight.</span>
        </h1>

        <p style={{
          fontSize: '1.08rem', color: '#94a3b8', maxWidth: 620, lineHeight: 1.72,
          marginBottom: 36, fontWeight: 400
        }}>
          StarTrace analyses NASA TESS stellar light curves using a Dual-Branch 1D Convolutional
          Neural Network to detect and classify exoplanet transit signals with astrophysical precision.
        </p>

        <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', justifyContent: 'center' }}>
          <button className="btn-primary" style={{ fontSize: '1rem', padding: '14px 32px' }}
            onClick={() => navigate('/dashboard/analyze')}>
            Start Analyzing <ArrowRight size={18} />
          </button>
          <a href="#how-it-works" className="btn-secondary" style={{ fontSize: '1rem', padding: '14px 28px' }}>
            Explore How It Works
          </a>
        </div>

        <div style={{ marginTop: 64, color: '#4a5568', animation: 'pulse-glow 2s ease infinite' }}>
          <ChevronDown size={22} />
        </div>
      </section>

      {/* Transit Photometry Explainer */}
      <section style={{ position: 'relative', zIndex: 1, padding: '80px 24px', maxWidth: 1100, margin: '0 auto' }}>
        <div style={{ textAlign: 'center', marginBottom: 56 }}>
          <div style={{ color: '#00f2fe', fontSize: '0.75rem', letterSpacing: '0.3em', textTransform: 'uppercase', fontWeight: 700, marginBottom: 12 }}>
            THE SCIENCE
          </div>
          <h2 style={{ fontSize: 'clamp(1.6rem, 4vw, 2.4rem)', fontWeight: 800, color: '#e2e8f0' }}>
            What is Transit Photometry?
          </h2>
          <p style={{ color: '#94a3b8', maxWidth: 640, margin: '16px auto 0', lineHeight: 1.7 }}>
            When a planet passes in front of its host star, it blocks a tiny fraction of the starlight.
            This periodic dimming — measured as a light curve — is the signature of an exoplanet transit.
            StarTrace detects and characterises these signals using Box Least Squares analysis
            and deep learning classification.
          </p>
        </div>

        {/* Signal classification cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))', gap: 20 }}>
          {CLASS_CARDS.map(card => (
            <div key={card.id} className="glass-card" style={{ padding: '1.4rem' }}>
              <div style={{ fontSize: '2rem', marginBottom: 10 }}>{card.icon}</div>
              <div style={{ fontWeight: 700, fontSize: '0.9rem', color: card.color, marginBottom: 6 }}>
                Class {card.id}: {card.name}
              </div>
              <div style={{ color: '#64748b', fontSize: '0.83rem', lineHeight: 1.6 }}>{card.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" style={{ position: 'relative', zIndex: 1, padding: '80px 24px', maxWidth: 1100, margin: '0 auto' }}>
        <div style={{ textAlign: 'center', marginBottom: 56 }}>
          <div style={{ color: '#00f2fe', fontSize: '0.75rem', letterSpacing: '0.3em', textTransform: 'uppercase', fontWeight: 700, marginBottom: 12 }}>
            METHODOLOGY
          </div>
          <h2 style={{ fontSize: 'clamp(1.6rem, 4vw, 2.4rem)', fontWeight: 800, color: '#e2e8f0' }}>
            The StarTrace Pipeline
          </h2>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 20 }}>
          {STEPS.map((step, i) => (
            <div key={i} className="glass-card" style={{ padding: '1.6rem', textAlign: 'center', position: 'relative' }}>
              <div style={{
                width: 48, height: 48, borderRadius: '50%', margin: '0 auto 16px',
                background: 'linear-gradient(135deg,rgba(0,242,254,0.15),rgba(144,85,255,0.15))',
                border: '1px solid rgba(0,242,254,0.25)',
                display: 'flex', alignItems: 'center', justifyContent: 'center'
              }}>
                <step.icon size={22} color="#00f2fe" />
              </div>
              <div style={{
                position: 'absolute', top: 20, right: 20,
                fontSize: '1.6rem', fontWeight: 900, color: 'rgba(0,242,254,0.1)'
              }}>{String(i+1).padStart(2,'0')}</div>
              <div style={{ fontWeight: 700, fontSize: '0.92rem', color: '#e2e8f0', marginBottom: 8 }}>{step.label}</div>
              <div style={{ color: '#64748b', fontSize: '0.82rem', lineHeight: 1.6 }}>{step.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Technology Stack */}
      <section id="technology" style={{ position: 'relative', zIndex: 1, padding: '80px 24px', maxWidth: 1100, margin: '0 auto' }}>
        <div style={{ textAlign: 'center', marginBottom: 48 }}>
          <div style={{ color: '#00f2fe', fontSize: '0.75rem', letterSpacing: '0.3em', textTransform: 'uppercase', fontWeight: 700, marginBottom: 12 }}>
            TECHNOLOGY STACK
          </div>
          <h2 style={{ fontSize: 'clamp(1.4rem, 3.5vw, 2.2rem)', fontWeight: 800, color: '#e2e8f0' }}>
            Built with Modern Scientific Tools
          </h2>
        </div>
        <div className="glass-card" style={{ padding: '2rem', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 24 }}>
          {[
            { label: 'Frontend', items: ['React + Vite', 'Tailwind CSS', 'Plotly.js', 'Lucide Icons'] },
            { label: 'Backend', items: ['Python FastAPI', 'Uvicorn ASGI', 'SQLAlchemy ORM', 'PostgreSQL / SQLite'] },
            { label: 'Machine Learning', items: ['PyTorch 1D-CNN', 'Dual-Branch Architecture', 'Scikit-learn', 'NumPy / Pandas'] },
            { label: 'Astronomy', items: ['Lightkurve (NASA)', 'Astropy BLS', 'Savitzky-Golay Filter', 'TESS / MAST Archive'] },
          ].map(col => (
            <div key={col.label}>
              <div style={{ color: '#00f2fe', fontWeight: 700, fontSize: '0.8rem', letterSpacing: '0.15em', textTransform: 'uppercase', marginBottom: 12 }}>
                {col.label}
              </div>
              {col.items.map(item => (
                <div key={item} style={{ color: '#94a3b8', fontSize: '0.86rem', marginBottom: 7, display: 'flex', alignItems: 'center', gap: 7 }}>
                  <div style={{ width: 5, height: 5, borderRadius: '50%', background: '#00f2fe', flexShrink: 0 }} />
                  {item}
                </div>
              ))}
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section style={{ position: 'relative', zIndex: 1, padding: '80px 24px', textAlign: 'center' }}>
        <div className="glass-card" style={{ maxWidth: 700, margin: '0 auto', padding: '3rem 2rem' }}>
          <h2 style={{ fontSize: 'clamp(1.5rem, 4vw, 2.2rem)', fontWeight: 800, color: '#e2e8f0', marginBottom: 16 }}>
            Ready to Hunt for Exoplanets?
          </h2>
          <p style={{ color: '#64748b', marginBottom: 28, lineHeight: 1.7 }}>
            Enter a TESS TIC ID or upload your own light curve CSV to run the full AI analysis pipeline.
          </p>
          <button className="btn-primary" style={{ fontSize: '1rem', padding: '14px 36px' }}
            onClick={() => navigate('/dashboard/analyze')}>
            Start Analyzing <ArrowRight size={18} />
          </button>
        </div>
      </section>

      {/* Footer */}
      <footer style={{
        position: 'relative', zIndex: 1, textAlign: 'center',
        padding: '24px', color: '#3a4555', fontSize: '0.78rem',
        borderTop: '1px solid rgba(0,242,254,0.06)'
      }}>
        <div style={{ marginBottom: 6 }}>
          <span style={{ fontWeight: 700, letterSpacing: '0.1em' }}>STARTRACE</span>
          {' '}— Discovering Hidden Worlds Through Stellar Light.
        </div>
        <div>
          Astronomical data sourced from{' '}
          <a href="https://mast.stsci.edu" target="_blank" rel="noreferrer"
            style={{ color: '#4facfe', textDecoration: 'none' }}>NASA TESS via MAST Archive</a>.{' '}
          Built as a B.Tech CSE Academic Project using public domain research.
        </div>
      </footer>
    </div>
  )
}
