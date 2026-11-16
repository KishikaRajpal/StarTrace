import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { FlaskConical, TrendingUp, Zap, AlertCircle, Clock, ExternalLink } from 'lucide-react'
import { getDashboardStats } from '../services/api'

const CLASS_COLORS = {
  'Exoplanet Transit': '#00f2fe',
  'Eclipsing Binary': '#ff007f',
  'Background Blend': '#ffd166',
  'Stellar Noise / Activity': '#06d6a0',
}

function ClassBadge({ cls }) {
  const color = CLASS_COLORS[cls] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 20,
      background: color + '1a', color, border: `1px solid ${color}40`,
      fontSize: '0.74rem', fontWeight: 600
    }}>{cls || '—'}</span>
  )
}

export default function Overview() {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const navigate = useNavigate()

  useEffect(() => {
    getDashboardStats()
      .then(res => { setStats(res.data); setLoading(false) })
      .catch(err => { setError(err.message); setLoading(false) })
  }, [])

  const kpiCards = stats ? [
    { label: 'Total Analyses', value: stats.total_analyses, color: '#00f2fe', icon: FlaskConical },
    { label: 'Exoplanet Candidates', value: stats.exoplanet_candidates, color: '#00f2fe', icon: Zap },
    { label: 'Eclipsing Binaries', value: stats.eclipsing_binaries, color: '#ff007f', icon: TrendingUp },
    { label: 'Stellar Activity', value: stats.stellar_noise, color: '#06d6a0', icon: AlertCircle },
  ] : []

  return (
    <div style={{ padding: '32px 32px 48px', maxWidth: 1200 }}>
      {/* Header */}
      <div style={{ marginBottom: 32 }}>
        <h1 style={{ fontWeight: 800, fontSize: '1.6rem', color: '#e2e8f0', marginBottom: 4 }}>Mission Control</h1>
        <p style={{ color: '#64748b', fontSize: '0.88rem' }}>Real-time overview of all StarTrace analyses and detections.</p>
      </div>

      {error && (
        <div className="alert-error" style={{ marginBottom: 24 }}>
          <AlertCircle size={16} /> Could not load dashboard stats: {error}
        </div>
      )}

      {/* KPI Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16, marginBottom: 36 }}>
        {loading ? Array(4).fill(0).map((_, i) => (
          <div key={i} className="kpi-card skeleton" style={{ height: 100 }} />
        )) : kpiCards.map(({ label, value, color, icon: Icon }) => (
          <div key={label} className="kpi-card">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
              <span style={{ fontSize: '0.74rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: '#64748b', fontWeight: 600 }}>{label}</span>
              <Icon size={16} color={color} />
            </div>
            <div style={{ fontSize: '2.2rem', fontWeight: 900, color, lineHeight: 1 }}>{value}</div>
          </div>
        ))}
      </div>

      {/* Quick actions */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 36, flexWrap: 'wrap' }}>
        <button className="btn-primary" onClick={() => navigate('/dashboard/analyze')}>
          <FlaskConical size={16} /> Run New Analysis
        </button>
        <button className="btn-secondary" onClick={() => navigate('/dashboard/history')}>
          <Clock size={16} /> View Full History
        </button>
      </div>

      {/* Recent analyses */}
      <div className="glass-card" style={{ padding: '1.4rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
          <h2 style={{ fontWeight: 700, fontSize: '1rem', color: '#e2e8f0' }}>Recent Analyses</h2>
          <button onClick={() => navigate('/dashboard/history')} style={{
            color: '#00f2fe', fontSize: '0.78rem', background: 'none', border: 'none', cursor: 'pointer',
            display: 'flex', alignItems: 'center', gap: 4, fontWeight: 600
          }}>
            View all <ExternalLink size={12} />
          </button>
        </div>

        {loading ? (
          <div style={{ padding: '20px 0' }}>
            {Array(4).fill(0).map((_, i) => (
              <div key={i} className="skeleton" style={{ height: 44, marginBottom: 10, borderRadius: 8 }} />
            ))}
          </div>
        ) : !stats?.recent_analyses?.length ? (
          <div style={{ textAlign: 'center', padding: '40px 0', color: '#4a5568' }}>
            <FlaskConical size={36} style={{ margin: '0 auto 12px', display: 'block', opacity: 0.3 }} />
            <div style={{ fontWeight: 600, marginBottom: 6 }}>No analyses yet</div>
            <div style={{ fontSize: '0.84rem' }}>Run your first analysis to populate this dashboard.</div>
          </div>
        ) : (
          <table className="st-table">
            <thead>
              <tr>
                <th>TIC ID</th>
                <th>Classification</th>
                <th>Confidence</th>
                <th>Period (days)</th>
                <th>SNR</th>
                <th>Date</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {stats.recent_analyses.map(a => (
                <tr key={a.analysis_id}>
                  <td style={{ fontFamily: 'JetBrains Mono, monospace', color: '#00f2fe', fontWeight: 600, fontSize: '0.85rem' }}>
                    TIC {a.tic_id}
                  </td>
                  <td><ClassBadge cls={a.predicted_class} /></td>
                  <td style={{ color: '#94a3b8' }}>
                    {a.confidence ? (a.confidence * 100).toFixed(1) + '%' : '—'}
                  </td>
                  <td style={{ color: '#94a3b8', fontFamily: 'JetBrains Mono, monospace', fontSize: '0.83rem' }}>
                    {a.orbital_period_days ? a.orbital_period_days.toFixed(3) : '—'}
                  </td>
                  <td style={{ color: '#94a3b8' }}>
                    {a.snr ? a.snr.toFixed(1) : '—'}
                  </td>
                  <td style={{ color: '#64748b', fontSize: '0.8rem' }}>
                    {a.created_at ? new Date(a.created_at).toLocaleDateString() : '—'}
                  </td>
                  <td>
                    <button onClick={() => navigate(`/dashboard/result/${a.analysis_id}`)}
                      style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#00f2fe' }}>
                      <ExternalLink size={14} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
