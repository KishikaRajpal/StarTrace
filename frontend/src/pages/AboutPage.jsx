import React from 'react'
import { Star, ExternalLink, BookOpen, Award, Telescope, Globe } from 'lucide-react'

export default function AboutPage() {
  return (
    <div style={{ padding: '32px 32px 64px', maxWidth: 960, margin: '0 auto' }}>
      <div style={{ marginBottom: 32 }}>
        <h1 style={{ fontWeight: 800, fontSize: '1.7rem', color: '#e2e8f0', marginBottom: 6 }}>
          About StarTrace
        </h1>
        <p style={{ color: '#64748b', fontSize: '0.9rem' }}>
          Discovering Hidden Worlds Through Stellar Light.
        </p>
      </div>

      <div className="glass-card" style={{ padding: '2rem', marginBottom: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
          <div style={{
            width: 40, height: 40, borderRadius: 10,
            background: 'linear-gradient(135deg, #00f2fe, #9055ff)',
            display: 'flex', alignItems: 'center', justifyContent: 'center'
          }}>
            <Star size={20} fill="#060a12" color="#060a12" />
          </div>
          <div>
            <h2 style={{ fontSize: '1.2rem', fontWeight: 800, color: '#e2e8f0', margin: 0 }}>Project Overview</h2>
            <div style={{ fontSize: '0.8rem', color: '#00f2fe' }}>B.Tech Computer Science Academic Mini-Project</div>
          </div>
        </div>

        <p style={{ color: '#94a3b8', fontSize: '0.88rem', lineHeight: 1.75, marginBottom: 14 }}>
          <strong>StarTrace</strong> is a full-stack, AI-enabled astronomical analysis platform designed to identify and classify potential
          exoplanetary transit signals from noisy stellar light curves captured by NASA’s Transiting Exoplanet Survey Satellite (TESS).
        </p>
        <p style={{ color: '#94a3b8', fontSize: '0.88rem', lineHeight: 1.75 }}>
          Traditional exoplanet hunting relies heavily on manual human vetting or computationally intensive manual modeling to distinguish
          true planetary occultations from astrophysical false positives (such as eclipsing binaries and blended background sources).
          StarTrace automates this workflow by integrating live data retrieval, Savitzky-Golay detrending, Box Least Squares (BLS) period
          searching, and a PyTorch Dual-Branch 1D Convolutional Neural Network.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 20, marginBottom: 28 }}>
        <div className="glass-card" style={{ padding: '1.6rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12, color: '#00f2fe' }}>
            <Telescope size={18} />
            <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#e2e8f0', margin: 0 }}>Data Sources</h3>
          </div>
          <p style={{ color: '#94a3b8', fontSize: '0.84rem', lineHeight: 1.7 }}>
            Observational data is retrieved from the Barbara A. Mikulski Archive for Space Telescopes (MAST) at the Space Telescope Science Institute (STScI),
            utilizing Lightkurve. Both Science Processing Operations Center (SPOC) 2-minute cadence and Quick-Look Pipeline (QLP) observations are supported.
          </p>
        </div>

        <div className="glass-card" style={{ padding: '1.6rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12, color: '#9055ff' }}>
            <Award size={18} />
            <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#e2e8f0', margin: 0 }}>Academic Integrity</h3>
          </div>
          <p style={{ color: '#94a3b8', fontSize: '0.84rem', lineHeight: 1.7 }}>
            StarTrace clearly distinguishes between model-based candidate estimations and confirmed physical measurements.
            All planetary metrics are labeled as "Potential Exoplanet Candidate" and include calculated SNR, period uncertainties, and scientific diagnostics.
          </p>
        </div>
      </div>

      {/* Citations & References */}
      <div className="glass-card" style={{ padding: '1.6rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14, color: '#00f2fe' }}>
          <BookOpen size={18} />
          <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#e2e8f0', margin: 0 }}>Scientific References</h3>
        </div>
        <ul style={{ color: '#64748b', fontSize: '0.82rem', lineHeight: 1.8, paddingLeft: 18, margin: 0 }}>
          <li>
            Shallue, C. J., & Vanderburg, A. (2018). <em>Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90.</em> The Astronomical Journal, 155(2), 94.
          </li>
          <li>
            Kovács, G., Zucker, S., & Mazeh, T. (2002). <em>A box-fitting algorithm in the search for periodic transits.</em> Astronomy & Astrophysics, 391(1), 369-377.
          </li>
          <li>
            Lightkurve Collaboration (2018). <em>Lightkurve: Kepler and TESS time series analysis in Python.</em> Astrophysics Source Code Library.
          </li>
          <li>
            Ricker, G. R., et al. (2015). <em>Transiting Exoplanet Survey Satellite (TESS).</em> Journal of Astronomical Telescopes, Instruments, and Systems, 1(1), 014003.
          </li>
        </ul>
      </div>
    </div>
  )
}
