import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  FlaskConical, Upload, Sparkles, AlertCircle, ArrowRight,
  CheckCircle2, Loader2, Database, FileSpreadsheet, Orbit, Info
} from 'lucide-react'
import { runAnalysisByTic, uploadCSVAnalysis, getSampleDatasets } from '../services/api'

const STAGES = [
  'Fetching astronomical data from TESS / MAST archive',
  'Preprocessing photometric light curve & removing outliers',
  'Removing stellar noise & trends via Savitzky-Golay filter',
  'Detecting periodic signals using Box Least Squares (BLS)',
  'Extracting macro-orbital (2001) and micro-transit (201) views',
  'Running Dual-Branch 1D-CNN neural network classification',
  'Estimating physical transit parameters and candidate validity',
  'Generating comprehensive scientific report'
]

export default function NewAnalysis() {
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState('tic') // 'tic' | 'sample' | 'csv'
  const [ticId, setTicId] = useState('25155310')
  const [minPeriod, setMinPeriod] = useState(0.5)
  const [maxPeriod, setMaxPeriod] = useState(15.0)
  const [csvFile, setCsvFile] = useState(null)
  const [samples, setSamples] = useState([])

  const [loading, setLoading] = useState(false)
  const [currentStage, setCurrentStage] = useState(0)
  const [error, setError] = useState(null)

  useEffect(() => {
    getSampleDatasets()
      .then(res => setSamples(res.data?.samples || []))
      .catch(() => {})
  }, [])

  const executePipeline = async (runner) => {
    setError(null)
    setLoading(true)
    setCurrentStage(0)

    // Simulate animated progress through the 8 scientific stages
    const stageInterval = setInterval(() => {
      setCurrentStage(prev => {
        if (prev < STAGES.length - 1) return prev + 1
        return prev
      })
    }, 1100)

    try {
      const res = await runner()
      clearInterval(stageInterval)
      setCurrentStage(STAGES.length - 1)
      if (res.data?.analysis_id) {
        setTimeout(() => {
          navigate(`/dashboard/result/${res.data.analysis_id}`)
        }, 600)
      } else {
        throw new Error(res.data?.error || 'Analysis failed to return an analysis ID.')
      }
    } catch (err) {
      clearInterval(stageInterval)
      setLoading(false)
      setError(err.message || 'An unexpected error occurred during processing.')
    }
  }

  const handleRunTic = (idToRun) => {
    const id = idToRun || ticId
    if (!id || !id.trim()) {
      setError('Please provide a valid TIC ID.')
      return
    }
    executePipeline(() => runAnalysisByTic(id.trim(), minPeriod, maxPeriod))
  }

  const handleRunCSV = () => {
    if (!csvFile) {
      setError('Please select a CSV file containing time and flux columns.')
      return
    }
    executePipeline(() => uploadCSVAnalysis(csvFile, minPeriod, maxPeriod))
  }

  return (
    <div style={{ padding: '32px 32px 48px', maxWidth: 1000, margin: '0 auto' }}>
      <div style={{ marginBottom: 32 }}>
        <h1 style={{ fontWeight: 800, fontSize: '1.7rem', color: '#e2e8f0', marginBottom: 6 }}>
          Run New Astronomical Analysis
        </h1>
        <p style={{ color: '#64748b', fontSize: '0.9rem' }}>
          Ingest TESS light curves, condition raw flux, execute BLS periodicity search,
          and classify candidate signals with the PyTorch Dual-Branch 1D-CNN.
        </p>
      </div>

      {error && (
        <div className="alert-error" style={{ marginBottom: 24 }}>
          <AlertCircle size={18} style={{ flexShrink: 0 }} />
          <div>
            <div style={{ fontWeight: 600 }}>Analysis Request Error</div>
            <div style={{ fontSize: '0.84rem' }}>{error}</div>
          </div>
        </div>
      )}

      {/* Progress modal / overlay */}
      {loading && (
        <div className="glass-card" style={{ padding: '2.5rem', marginBottom: 32, border: '1px solid rgba(0,242,254,0.4)', background: 'rgba(9,13,24,0.95)' }}>
          <div style={{ textAlign: 'center', marginBottom: 28 }}>
            <div style={{
              width: 56, height: 56, borderRadius: '50%', margin: '0 auto 16px',
              background: 'linear-gradient(135deg, rgba(0,242,254,0.2), rgba(144,85,255,0.2))',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              border: '1.5px solid rgba(0,242,254,0.4)'
            }}>
              <Orbit className="animate-orbit" size={28} color="#00f2fe" />
            </div>
            <h3 style={{ fontSize: '1.25rem', fontWeight: 800, color: '#e2e8f0', marginBottom: 6 }}>
              Processing Astronomical Light Curve
            </h3>
            <p style={{ color: '#00f2fe', fontSize: '0.86rem', fontWeight: 500 }}>
              Stage {currentStage + 1} of {STAGES.length}: {STAGES[currentStage]}
            </p>
          </div>

          <div className="progress-track" style={{ marginBottom: 28 }}>
            <div
              className="progress-fill"
              style={{ width: `${((currentStage + 1) / STAGES.length) * 100}%` }}
            />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 10 }}>
            {STAGES.map((stg, idx) => {
              const isPast = idx < currentStage
              const isCurrent = idx === currentStage
              return (
                <div key={stg} style={{
                  display: 'flex', alignItems: 'center', gap: 10, padding: '8px 12px',
                  borderRadius: 8, background: isCurrent ? 'rgba(0,242,254,0.08)' : 'rgba(255,255,255,0.02)',
                  border: isCurrent ? '1px solid rgba(0,242,254,0.3)' : '1px solid transparent'
                }}>
                  {isPast ? (
                    <CheckCircle2 size={16} color="#06d6a0" />
                  ) : isCurrent ? (
                    <Loader2 size={16} color="#00f2fe" className="animate-spin" />
                  ) : (
                    <div style={{ width: 16, height: 16, borderRadius: '50%', border: '1px solid #4a5568' }} />
                  )}
                  <span style={{
                    fontSize: '0.78rem',
                    color: isCurrent ? '#00f2fe' : isPast ? '#94a3b8' : '#4a5568',
                    fontWeight: isCurrent ? 600 : 400
                  }}>
                    {idx + 1}. {stg}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Mode Tabs */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 24, borderBottom: '1px solid rgba(0,242,254,0.1)', paddingBottom: 12 }}>
        <button
          onClick={() => setActiveTab('tic')}
          className="sidebar-link"
          style={{
            background: activeTab === 'tic' ? 'rgba(0,242,254,0.12)' : 'transparent',
            color: activeTab === 'tic' ? '#00f2fe' : '#94a3b8',
            borderColor: activeTab === 'tic' ? 'rgba(0,242,254,0.3)' : 'transparent'
          }}
        >
          <Database size={16} /> NASA TESS (TIC ID)
        </button>
        <button
          onClick={() => setActiveTab('sample')}
          className="sidebar-link"
          style={{
            background: activeTab === 'sample' ? 'rgba(0,242,254,0.12)' : 'transparent',
            color: activeTab === 'sample' ? '#00f2fe' : '#94a3b8',
            borderColor: activeTab === 'sample' ? 'rgba(0,242,254,0.3)' : 'transparent'
          }}
        >
          <Sparkles size={16} /> Sample Demonstration Presets
        </button>
        <button
          onClick={() => setActiveTab('csv')}
          className="sidebar-link"
          style={{
            background: activeTab === 'csv' ? 'rgba(0,242,254,0.12)' : 'transparent',
            color: activeTab === 'csv' ? '#00f2fe' : '#94a3b8',
            borderColor: activeTab === 'csv' ? 'rgba(0,242,254,0.3)' : 'transparent'
          }}
        >
          <FileSpreadsheet size={16} /> CSV Photometry Upload
        </button>
      </div>

      {/* Tab 1: TIC ID Search */}
      {activeTab === 'tic' && (
        <div className="glass-card" style={{ padding: '2rem' }}>
          <h2 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#e2e8f0', marginBottom: 12 }}>
            TESS Input Catalog (TIC) Retrieval
          </h2>
          <p style={{ color: '#64748b', fontSize: '0.86rem', marginBottom: 20 }}>
            Enter a standard TESS TIC ID to retrieve publicly available observations from NASA MAST.
            If the network query times out or the target has no 2-minute cadence SPOC observations,
            the system seamlessly engages calibrated physics-based demonstration processing.
          </p>

          <div style={{ marginBottom: 20 }}>
            <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>
              TIC Star Identifier
            </label>
            <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
              <input
                type="text"
                className="st-input"
                placeholder="e.g. 25155310, 149603524, 38846515"
                value={ticId}
                onChange={e => setTicId(e.target.value)}
                disabled={loading}
              />
            </div>
            <div style={{ fontSize: '0.78rem', color: '#64748b', marginTop: 6 }}>
              Quick test IDs: <button type="button" onClick={() => setTicId('25155310')} style={{ background: 'none', border: 'none', color: '#00f2fe', cursor: 'pointer', textDecoration: 'underline' }}>25155310 (Kepler-1649c)</button>,{' '}
              <button type="button" onClick={() => setTicId('149603524')} style={{ background: 'none', border: 'none', color: '#ff007f', cursor: 'pointer', textDecoration: 'underline' }}>149603524 (Binary)</button>,{' '}
              <button type="button" onClick={() => setTicId('38846515')} style={{ background: 'none', border: 'none', color: '#ffd166', cursor: 'pointer', textDecoration: 'underline' }}>38846515 (Blend)</button>
            </div>
          </div>

          {/* Configurable BLS Period Range */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', marginBottom: 6 }}>
                Min Period Search (Days)
              </label>
              <input
                type="number"
                step="0.1"
                min="0.1"
                max="30"
                className="st-input"
                value={minPeriod}
                onChange={e => setMinPeriod(parseFloat(e.target.value) || 0.5)}
                disabled={loading}
              />
            </div>
            <div>
              <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', marginBottom: 6 }}>
                Max Period Search (Days)
              </label>
              <input
                type="number"
                step="0.5"
                min="1.0"
                max="50"
                className="st-input"
                value={maxPeriod}
                onChange={e => setMaxPeriod(parseFloat(e.target.value) || 15.0)}
                disabled={loading}
              />
            </div>
          </div>

          <button
            className="btn-primary"
            style={{ width: '100%', justifyContent: 'center', padding: '14px' }}
            onClick={() => handleRunTic()}
            disabled={loading}
          >
            <FlaskConical size={18} /> Run Pipeline Analysis
          </button>
        </div>
      )}

      {/* Tab 2: Sample Presets */}
      {activeTab === 'sample' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 18 }}>
          {samples.map(s => {
            const colors = {
              'Exoplanet Transit': '#00f2fe',
              'Eclipsing Binary': '#ff007f',
              'Background Blend': '#ffd166',
              'Stellar Noise / Activity': '#06d6a0'
            }
            const accent = colors[s.category] || '#00f2fe'
            return (
              <div key={s.tic_id} className="glass-card" style={{ padding: '1.4rem', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
                    <span style={{ fontSize: '0.75rem', fontWeight: 700, color: accent, background: accent + '18', padding: '3px 8px', borderRadius: 12 }}>
                      {s.category}
                    </span>
                    <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.75rem', color: '#64748b' }}>
                      TIC {s.tic_id}
                    </span>
                  </div>
                  <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#e2e8f0', marginBottom: 8 }}>
                    {s.name}
                  </h3>
                  <p style={{ color: '#64748b', fontSize: '0.82rem', lineHeight: 1.6, marginBottom: 16 }}>
                    {s.description}
                  </p>
                  <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginBottom: 16, display: 'flex', gap: 16 }}>
                    <span>P ≈ {s.expected_period_days} d</span>
                    <span>Depth ≈ {s.expected_depth_pct}%</span>
                  </div>
                </div>

                <button
                  className="btn-secondary"
                  style={{ width: '100%', justifyContent: 'center', padding: '10px' }}
                  onClick={() => handleRunTic(s.tic_id)}
                  disabled={loading}
                >
                  <Sparkles size={14} /> Analyze Preset
                </button>
              </div>
            )
          })}
        </div>
      )}

      {/* Tab 3: CSV Upload */}
      {activeTab === 'csv' && (
        <div className="glass-card" style={{ padding: '2rem' }}>
          <h2 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#e2e8f0', marginBottom: 12 }}>
            Upload Custom Light Curve (CSV)
          </h2>
          <p style={{ color: '#64748b', fontSize: '0.86rem', marginBottom: 20 }}>
            Upload your own photometric time-series CSV file. The file should contain time (BJD or relative days)
            and normalized or raw flux columns. StarTrace automatically auto-detects column names.
          </p>

          <div
            style={{
              border: '2px dashed rgba(0,242,254,0.3)',
              borderRadius: 12,
              padding: '36px 20px',
              textAlign: 'center',
              marginBottom: 20,
              background: 'rgba(9,13,24,0.6)'
            }}
          >
            <Upload size={36} color="#00f2fe" style={{ margin: '0 auto 12px' }} />
            <div style={{ color: '#e2e8f0', fontWeight: 600, marginBottom: 6 }}>
              {csvFile ? csvFile.name : 'Select or drag & drop CSV file'}
            </div>
            <div style={{ color: '#64748b', fontSize: '0.8rem', marginBottom: 16 }}>
              Supported columns: time, flux, pdcsap_flux, bjd, btjd (Max 20MB)
            </div>
            <input
              type="file"
              accept=".csv"
              id="csv-file-input"
              style={{ display: 'none' }}
              onChange={e => setCsvFile(e.target.files?.[0] || null)}
              disabled={loading}
            />
            <label htmlFor="csv-file-input" className="btn-secondary" style={{ cursor: 'pointer', display: 'inline-flex' }}>
              Browse File
            </label>
          </div>

          <button
            className="btn-primary"
            style={{ width: '100%', justifyContent: 'center', padding: '14px' }}
            onClick={handleRunCSV}
            disabled={loading || !csvFile}
          >
            <Upload size={18} /> Ingest & Analyze CSV
          </button>
        </div>
      )}
    </div>
  )
}
