import React, { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import Plot from 'react-plotly.js'
import {
  Download, FileText, ArrowLeft, AlertCircle, Sparkles,
  CheckCircle2, Clock, Orbit, Telescope, Info, ExternalLink
} from 'lucide-react'
import { getAnalysis, getCSVReportUrl, getPDFReportUrl } from '../services/api'

const CLASS_COLORS = {
  'Exoplanet Transit': '#00f2fe',
  'Eclipsing Binary': '#ff007f',
  'Background Blend': '#ffd166',
  'Stellar Noise / Activity': '#06d6a0'
}

const DARK_LAYOUT_BASE = {
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'rgba(9,13,24,0.7)',
  font: { family: 'Inter, sans-serif', color: '#94a3b8', size: 11 },
  margin: { l: 60, r: 25, t: 35, b: 50 },
  xaxis: {
    gridcolor: 'rgba(255,255,255,0.06)',
    zerolinecolor: 'rgba(255,255,255,0.1)',
    tickfont: { color: '#64748b' }
  },
  yaxis: {
    gridcolor: 'rgba(255,255,255,0.06)',
    zerolinecolor: 'rgba(255,255,255,0.1)',
    tickfont: { color: '#64748b' }
  },
  showlegend: false
}

export default function AnalysisResult() {
  const { analysisId } = useParams()
  const navigate = useNavigate()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    getAnalysis(analysisId)
      .then(res => {
        setData(res.data)
        setLoading(false)
      })
      .catch(err => {
        setError(err.message || 'Failed to load analysis record.')
        setLoading(false)
      })
  }, [analysisId])

  if (loading) {
    return (
      <div style={{ padding: '60px 32px', textAlign: 'center' }}>
        <Orbit className="animate-orbit" size={40} color="#00f2fe" style={{ margin: '0 auto 16px' }} />
        <div style={{ fontWeight: 700, color: '#e2e8f0', fontSize: '1.1rem' }}>Retrieving Analysis Data</div>
        <div style={{ color: '#64748b', fontSize: '0.85rem', marginTop: 4 }}>Loading light curves and neural network representations...</div>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div style={{ padding: '40px 32px', maxWidth: 800, margin: '0 auto' }}>
        <div className="alert-error" style={{ marginBottom: 20 }}>
          <AlertCircle size={18} />
          <div>{error || 'Analysis record not found.'}</div>
        </div>
        <button className="btn-secondary" onClick={() => navigate('/dashboard/history')}>
          <ArrowLeft size={16} /> Back to History
        </button>
      </div>
    )
  }

  const clf = data.classification || {}
  const tp = data.transit_parameters || {}
  const bls = data.bls_stats || {}
  const viz = data.visualization || {}
  const predictedClass = clf.predicted_class || 'Unknown'
  const accentColor = CLASS_COLORS[predictedClass] || '#00f2fe'
  const isCandidate = tp.is_candidate

  return (
    <div style={{ padding: '32px 32px 64px', maxWidth: 1200, margin: '0 auto' }}>
      {/* Top action bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24, flexWrap: 'wrap', gap: 12 }}>
        <button className="btn-secondary" onClick={() => navigate('/dashboard')}>
          <ArrowLeft size={15} /> Dashboard
        </button>

        <div style={{ display: 'flex', gap: 10 }}>
          <a
            href={getCSVReportUrl(analysisId)}
            className="btn-secondary"
            download
            style={{ textDecoration: 'none' }}
          >
            <Download size={15} /> Export CSV
          </a>
          <a
            href={getPDFReportUrl(analysisId)}
            className="btn-primary"
            target="_blank"
            rel="noreferrer"
            style={{ textDecoration: 'none' }}
          >
            <FileText size={15} /> Download PDF Report
          </a>
        </div>
      </div>

      {/* Header Banner */}
      <div className="glass-card" style={{ padding: '1.8rem 2rem', marginBottom: 28, position: 'relative', overflow: 'hidden' }}>
        <div style={{
          position: 'absolute', top: 0, left: 0, bottom: 0, width: 6,
          background: accentColor, boxShadow: `0 0 16px ${accentColor}`
        }} />

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 16 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
              <h1 style={{ fontSize: '1.8rem', fontWeight: 900, color: '#e2e8f0', margin: 0 }}>
                TIC {data.tic_id}
              </h1>
              <span style={{
                display: 'inline-flex', alignItems: 'center', gap: 5, padding: '4px 12px',
                borderRadius: 20, background: accentColor + '20', color: accentColor,
                border: `1px solid ${accentColor}40`, fontSize: '0.82rem', fontWeight: 700
              }}>
                {predictedClass}
              </span>
              {data.is_synthetic && (
                <span style={{
                  padding: '3px 8px', borderRadius: 12, background: 'rgba(255,209,102,0.15)',
                  color: '#ffd166', border: '1px solid rgba(255,209,102,0.3)', fontSize: '0.72rem', fontWeight: 600
                }}>
                  Calibrated Demonstration
                </span>
              )}
            </div>

            <div style={{ color: '#64748b', fontSize: '0.84rem', display: 'flex', gap: 20, flexWrap: 'wrap' }}>
              <span>Source: <strong style={{ color: '#94a3b8' }}>{data.data_source}</strong></span>
              <span>Analysis ID: <code style={{ color: '#00f2fe' }}>{data.analysis_id.slice(0, 8)}</code></span>
              {data.completed_at && (
                <span>Completed: {new Date(data.completed_at).toLocaleString()}</span>
              )}
            </div>
          </div>

          {/* Candidate Status Badge */}
          <div style={{
            textAlign: 'right', padding: '10px 16px', borderRadius: 10,
            background: isCandidate ? 'rgba(0,242,254,0.1)' : 'rgba(255,255,255,0.03)',
            border: isCandidate ? '1px solid rgba(0,242,254,0.3)' : '1px solid rgba(255,255,255,0.08)'
          }}>
            <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: '#64748b', fontWeight: 600 }}>
              Physical Classification
            </div>
            <div style={{ fontSize: '1.05rem', fontWeight: 800, color: isCandidate ? '#00f2fe' : '#e2e8f0' }}>
              {tp.status_label || predictedClass}
            </div>
          </div>
        </div>
      </div>

      {/* Section 1: Classification & Probabilities */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 20, marginBottom: 28 }}>
        {/* Prediction summary */}
        <div className="glass-card" style={{ padding: '1.6rem' }}>
          <h2 style={{ fontSize: '0.85rem', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 12 }}>
            AI Classification (Dual-Branch 1D-CNN)
          </h2>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, marginBottom: 10 }}>
            <span style={{ fontSize: '2.4rem', fontWeight: 900, color: accentColor }}>
              {clf.confidence_pct ? clf.confidence_pct.toFixed(1) : 0}%
            </span>
            <span style={{ color: '#94a3b8', fontSize: '0.9rem' }}>Confidence Score</span>
          </div>
          <p style={{ color: '#cbd5e1', fontSize: '0.86rem', lineHeight: 1.65, marginBottom: 16 }}>
            {clf.description || tp.interpretation}
          </p>
          {tp.interpretation && (
            <div style={{
              background: 'rgba(0,242,254,0.05)', border: '1px solid rgba(0,242,254,0.15)',
              borderRadius: 8, padding: '10px 14px', fontSize: '0.82rem', color: '#94c8f0'
            }}>
              <strong>Astrophysical Interpretation:</strong> {tp.interpretation}
            </div>
          )}
        </div>

        {/* 4-class Probability distribution */}
        <div className="glass-card" style={{ padding: '1.6rem' }}>
          <h2 style={{ fontSize: '0.85rem', fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 16 }}>
            Class Probability Distribution
          </h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {Object.entries(clf.probabilities || {}).map(([clsName, prob]) => {
              const cColor = CLASS_COLORS[clsName] || '#64748b'
              const pct = (prob * 100).toFixed(1)
              return (
                <div key={clsName}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: 4 }}>
                    <span style={{ color: clsName === predictedClass ? '#e2e8f0' : '#94a3b8', fontWeight: clsName === predictedClass ? 700 : 500 }}>
                      {clsName}
                    </span>
                    <span style={{ fontFamily: 'JetBrains Mono, monospace', color: cColor, fontWeight: 700 }}>
                      {pct}%
                    </span>
                  </div>
                  <div className="progress-track" style={{ height: 6 }}>
                    <div
                      style={{
                        width: `${pct}%`,
                        height: '100%',
                        background: cColor,
                        borderRadius: 99
                      }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {/* Section 2: Transit Parameters */}
      <div style={{ marginBottom: 32 }}>
        <h2 style={{ fontSize: '1.15rem', fontWeight: 800, color: '#e2e8f0', marginBottom: 14 }}>
          Extracted Astronomical Parameters
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 14 }}>
          <div className="kpi-card">
            <div style={{ fontSize: '0.74rem', textTransform: 'uppercase', color: '#64748b', fontWeight: 600, marginBottom: 6 }}>Orbital Period</div>
            <div style={{ fontSize: '1.35rem', fontWeight: 800, color: '#00f2fe', fontFamily: 'JetBrains Mono, monospace' }}>
              {bls.period_days ? bls.period_days.toFixed(4) : '—'} <span style={{ fontSize: '0.8rem', color: '#64748b' }}>days</span>
            </div>
            {tp.orbital_period_err > 0 && (
              <div style={{ fontSize: '0.72rem', color: '#64748b' }}>± {tp.orbital_period_err} d</div>
            )}
          </div>

          <div className="kpi-card">
            <div style={{ fontSize: '0.74rem', textTransform: 'uppercase', color: '#64748b', fontWeight: 600, marginBottom: 6 }}>Transit Depth</div>
            <div style={{ fontSize: '1.35rem', fontWeight: 800, color: '#00f2fe', fontFamily: 'JetBrains Mono, monospace' }}>
              {bls.depth_pct ? bls.depth_pct.toFixed(3) : '—'} <span style={{ fontSize: '0.8rem', color: '#64748b' }}>%</span>
            </div>
            <div style={{ fontSize: '0.72rem', color: '#64748b' }}>{bls.depth_ppm ? Math.round(bls.depth_ppm) : 0} ppm</div>
          </div>

          <div className="kpi-card">
            <div style={{ fontSize: '0.74rem', textTransform: 'uppercase', color: '#64748b', fontWeight: 600, marginBottom: 6 }}>Transit Duration</div>
            <div style={{ fontSize: '1.35rem', fontWeight: 800, color: '#00f2fe', fontFamily: 'JetBrains Mono, monospace' }}>
              {bls.duration_hours ? bls.duration_hours.toFixed(2) : '—'} <span style={{ fontSize: '0.8rem', color: '#64748b' }}>hours</span>
            </div>
            <div style={{ fontSize: '0.72rem', color: '#64748b' }}>
              {bls.duration_hours ? (bls.duration_hours / 24).toFixed(4) : 0} days
            </div>
          </div>

          <div className="kpi-card">
            <div style={{ fontSize: '0.74rem', textTransform: 'uppercase', color: '#64748b', fontWeight: 600, marginBottom: 6 }}>Detection SNR</div>
            <div style={{ fontSize: '1.35rem', fontWeight: 800, color: bls.snr > 10 ? '#06d6a0' : '#ffd166', fontFamily: 'JetBrains Mono, monospace' }}>
              {bls.snr ? bls.snr.toFixed(1) : '—'}
            </div>
            <div style={{ fontSize: '0.72rem', color: '#64748b' }}>Power: {bls.max_power ? bls.max_power.toFixed(3) : '—'}</div>
          </div>

          <div className="kpi-card">
            <div style={{ fontSize: '0.74rem', textTransform: 'uppercase', color: '#64748b', fontWeight: 600, marginBottom: 6 }}>Transit Epoch</div>
            <div style={{ fontSize: '1.1rem', fontWeight: 800, color: '#e2e8f0', fontFamily: 'JetBrains Mono, monospace' }}>
              {bls.transit_epoch_bjd ? bls.transit_epoch_bjd.toFixed(3) : '—'}
            </div>
            <div style={{ fontSize: '0.72rem', color: '#64748b' }}>BJD</div>
          </div>

          {/* Planetary Estimates if Candidate */}
          {isCandidate && (
            <>
              <div className="kpi-card" style={{ border: '1px solid rgba(0,242,254,0.3)' }}>
                <div style={{ fontSize: '0.74rem', textTransform: 'uppercase', color: '#00f2fe', fontWeight: 600, marginBottom: 6 }}>Est. Planet Radius</div>
                <div style={{ fontSize: '1.35rem', fontWeight: 800, color: '#00f2fe', fontFamily: 'JetBrains Mono, monospace' }}>
                  {tp.estimated_planet_radius_earth ? tp.estimated_planet_radius_earth.toFixed(2) : '—'} <span style={{ fontSize: '0.8rem' }}>R_⊕</span>
                </div>
                <div style={{ fontSize: '0.72rem', color: '#64748b' }}>Rp/R* = {tp.radius_ratio_rp_rstar}</div>
              </div>

              <div className="kpi-card" style={{ border: '1px solid rgba(0,242,254,0.3)' }}>
                <div style={{ fontSize: '0.74rem', textTransform: 'uppercase', color: '#00f2fe', fontWeight: 600, marginBottom: 6 }}>Semi-Major Axis</div>
                <div style={{ fontSize: '1.35rem', fontWeight: 800, color: '#00f2fe', fontFamily: 'JetBrains Mono, monospace' }}>
                  {tp.estimated_semi_major_axis_au ? tp.estimated_semi_major_axis_au.toFixed(3) : '—'} <span style={{ fontSize: '0.8rem' }}>AU</span>
                </div>
                <div style={{ fontSize: '0.72rem', color: '#64748b' }}>Kepler's 3rd Law</div>
              </div>

              <div className="kpi-card" style={{ border: '1px solid rgba(0,242,254,0.3)' }}>
                <div style={{ fontSize: '0.74rem', textTransform: 'uppercase', color: '#00f2fe', fontWeight: 600, marginBottom: 6 }}>Equilibrium Temp</div>
                <div style={{ fontSize: '1.35rem', fontWeight: 800, color: '#00f2fe', fontFamily: 'JetBrains Mono, monospace' }}>
                  {tp.estimated_equilibrium_temp_k ? Math.round(tp.estimated_equilibrium_temp_k) : '—'} <span style={{ fontSize: '0.8rem' }}>K</span>
                </div>
                <div style={{ fontSize: '0.72rem', color: '#64748b' }}>Assuming Albedo 0.3</div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Section 3: Interactive Plotly Visualizations */}
      <div>
        <h2 style={{ fontSize: '1.15rem', fontWeight: 800, color: '#e2e8f0', marginBottom: 16 }}>
          Interactive Scientific Visualizations
        </h2>

        {/* Graph 1: Raw Light Curve */}
        <div className="glass-card" style={{ padding: '1.4rem', marginBottom: 20 }}>
          <div style={{ marginBottom: 10, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#e2e8f0' }}>A. Raw Light Curve</h3>
              <p style={{ color: '#64748b', fontSize: '0.8rem' }}>Original astronomical observations before detrending</p>
            </div>
            <span style={{ fontSize: '0.75rem', color: '#64748b' }}>{viz.raw_time?.length || 0} points displayed</span>
          </div>
          {viz.raw_time?.length ? (
            <Plot
              data={[
                {
                  x: viz.raw_time,
                  y: viz.raw_flux,
                  type: 'scatter',
                  mode: 'markers',
                  marker: { size: 2.5, color: '#4facfe', opacity: 0.7 }
                }
              ]}
              layout={{
                ...DARK_LAYOUT_BASE,
                height: 280,
                xaxis: { ...DARK_LAYOUT_BASE.xaxis, title: 'Time (BJD - Days)' },
                yaxis: { ...DARK_LAYOUT_BASE.yaxis, title: 'Normalized Flux' }
              }}
              config={{ responsive: true, displayModeBar: true, displaylogo: false }}
              style={{ width: '100%' }}
            />
          ) : (
            <div style={{ color: '#64748b', padding: '40px 0', textAlign: 'center' }}>No raw light curve points available.</div>
          )}
        </div>

        {/* Graph 2: Processed & Detrended Light Curve */}
        <div className="glass-card" style={{ padding: '1.4rem', marginBottom: 20 }}>
          <div style={{ marginBottom: 10 }}>
            <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#e2e8f0' }}>B. Processed & Detrended Light Curve</h3>
            <p style={{ color: '#64748b', fontSize: '0.8rem' }}>Cleaned signal with Savitzky-Golay stellar trend removed and baseline re-centered to 1.0</p>
          </div>
          {viz.clean_time?.length ? (
            <Plot
              data={[
                {
                  x: viz.clean_time,
                  y: viz.detrended_flux,
                  type: 'scatter',
                  mode: 'markers',
                  marker: { size: 2.5, color: '#00f2fe', opacity: 0.75 }
                }
              ]}
              layout={{
                ...DARK_LAYOUT_BASE,
                height: 280,
                xaxis: { ...DARK_LAYOUT_BASE.xaxis, title: 'Time (BJD - Days)' },
                yaxis: { ...DARK_LAYOUT_BASE.yaxis, title: 'Relative Detrended Flux' }
              }}
              config={{ responsive: true, displayModeBar: true, displaylogo: false }}
              style={{ width: '100%' }}
            />
          ) : (
            <div style={{ color: '#64748b', padding: '40px 0', textAlign: 'center' }}>No processed data available.</div>
          )}
        </div>

        {/* Graph 3: BLS Periodogram */}
        <div className="glass-card" style={{ padding: '1.4rem', marginBottom: 20 }}>
          <div style={{ marginBottom: 10 }}>
            <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#e2e8f0' }}>C. Box Least Squares (BLS) Periodogram</h3>
            <p style={{ color: '#64748b', fontSize: '0.8rem' }}>Periodogram power spectrum vs trial period. Peak indicates the strongest periodic transit signal.</p>
          </div>
          {viz.bls_periods?.length ? (
            <Plot
              data={[
                {
                  x: viz.bls_periods,
                  y: viz.bls_powers,
                  type: 'scatter',
                  mode: 'lines',
                  line: { color: '#9055ff', width: 1.5 }
                },
                {
                  x: [viz.bls_best_period],
                  y: [viz.bls_max_power],
                  type: 'scatter',
                  mode: 'markers+text',
                  text: [`P = ${viz.bls_best_period.toFixed(3)} d`],
                  textposition: 'top center',
                  textfont: { color: '#00f2fe', size: 12, family: 'JetBrains Mono' },
                  marker: { size: 9, color: '#00f2fe', symbol: 'star' }
                }
              ]}
              layout={{
                ...DARK_LAYOUT_BASE,
                height: 280,
                xaxis: { ...DARK_LAYOUT_BASE.xaxis, title: 'Trial Period (Days)' },
                yaxis: { ...DARK_LAYOUT_BASE.yaxis, title: 'BLS Detection Power' }
              }}
              config={{ responsive: true, displayModeBar: true, displaylogo: false }}
              style={{ width: '100%' }}
            />
          ) : (
            <div style={{ color: '#64748b', padding: '40px 0', textAlign: 'center' }}>No BLS periodogram data available.</div>
          )}
        </div>

        {/* Two-column layout for Folded & Local view */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: 20 }}>
          {/* Graph 4: Phase-Folded Curve */}
          <div className="glass-card" style={{ padding: '1.4rem' }}>
            <div style={{ marginBottom: 10 }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#e2e8f0' }}>D. Phase-Folded Light Curve</h3>
              <p style={{ color: '#64748b', fontSize: '0.8rem' }}>Folded over orbital period {bls.period_days?.toFixed(3)} d (Phase [-0.5, +0.5])</p>
            </div>
            {viz.folded_phase?.length ? (
              <Plot
                data={[
                  {
                    x: viz.folded_phase,
                    y: viz.folded_flux,
                    type: 'scatter',
                    mode: 'markers',
                    name: 'Observations',
                    marker: { size: 2, color: 'rgba(255,255,255,0.25)' }
                  },
                  {
                    x: viz.global_phase,
                    y: viz.global_flux,
                    type: 'scatter',
                    mode: 'lines',
                    name: '2001-Bin Global View',
                    line: { color: '#00f2fe', width: 2 }
                  }
                ]}
                layout={{
                  ...DARK_LAYOUT_BASE,
                  height: 300,
                  xaxis: { ...DARK_LAYOUT_BASE.xaxis, title: 'Orbital Phase' },
                  yaxis: { ...DARK_LAYOUT_BASE.yaxis, title: 'Normalized Flux' }
                }}
                config={{ responsive: true, displayModeBar: true, displaylogo: false }}
                style={{ width: '100%' }}
              />
            ) : (
              <div style={{ color: '#64748b', padding: '40px 0', textAlign: 'center' }}>No phase-folded data available.</div>
            )}
          </div>

          {/* Graph 5: Local Transit View */}
          <div className="glass-card" style={{ padding: '1.4rem' }}>
            <div style={{ marginBottom: 10 }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#e2e8f0' }}>E. Local Transit View</h3>
              <p style={{ color: '#64748b', fontSize: '0.8rem' }}>High-resolution zoom into transit center [-0.05, +0.05] (201 Bins)</p>
            </div>
            {viz.local_phase?.length ? (
              <Plot
                data={[
                  {
                    x: viz.local_phase,
                    y: viz.local_flux,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: '201-Bin Local Transit',
                    line: { color: '#00f2fe', width: 2.5 },
                    marker: { size: 3.5, color: '#00f2fe' }
                  }
                ]}
                layout={{
                  ...DARK_LAYOUT_BASE,
                  height: 300,
                  xaxis: { ...DARK_LAYOUT_BASE.xaxis, title: 'Transit Phase [-0.05, 0.05]' },
                  yaxis: { ...DARK_LAYOUT_BASE.yaxis, title: 'Relative Flux' }
                }}
                config={{ responsive: true, displayModeBar: true, displaylogo: false }}
                style={{ width: '100%' }}
              />
            ) : (
              <div style={{ color: '#64748b', padding: '40px 0', textAlign: 'center' }}>No local transit data available.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
