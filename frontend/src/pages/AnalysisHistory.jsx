import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  History, Search, Trash2, ExternalLink, Filter,
  AlertCircle, Orbit, CheckCircle2, RefreshCw
} from 'lucide-react'
import { listAnalyses, deleteAnalysis } from '../services/api'

const CLASS_COLORS = {
  'Exoplanet Transit': '#00f2fe',
  'Eclipsing Binary': '#ff007f',
  'Background Blend': '#ffd166',
  'Stellar Noise / Activity': '#06d6a0'
}

export default function AnalysisHistory() {
  const navigate = useNavigate()
  const [items, setItems] = useState([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const [searchTerm, setSearchTerm] = useState('')
  const [selectedClass, setSelectedClass] = useState('')
  const [deletingId, setDeletingId] = useState(null)

  const fetchHistory = () => {
    setLoading(true)
    setError(null)
    const params = {}
    if (selectedClass) params.filter_class = selectedClass
    if (searchTerm.trim()) params.search_tic = searchTerm.trim()

    listAnalyses(params)
      .then(res => {
        setItems(res.data?.items || [])
        setTotal(res.data?.total || 0)
        setLoading(false)
      })
      .catch(err => {
        setError(err.message || 'Failed to retrieve analysis history.')
        setLoading(false)
      })
  }

  useEffect(() => {
    fetchHistory()
  }, [selectedClass])

  const handleSearchSubmit = (e) => {
    e.preventDefault()
    fetchHistory()
  }

  const handleDelete = async (id, e) => {
    e.stopPropagation()
    if (!window.confirm(`Are you sure you want to delete analysis ${id.slice(0, 8)}?`)) return

    setDeletingId(id)
    try {
      await deleteAnalysis(id)
      setItems(prev => prev.filter(item => item.analysis_id !== id))
      setTotal(prev => Math.max(0, prev - 1))
    } catch (err) {
      alert(`Delete failed: ${err.message}`)
    } finally {
      setDeletingId(null)
    }
  }

  return (
    <div style={{ padding: '32px 32px 64px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 28, flexWrap: 'wrap', gap: 16 }}>
        <div>
          <h1 style={{ fontWeight: 800, fontSize: '1.7rem', color: '#e2e8f0', marginBottom: 6 }}>
            Analysis History
          </h1>
          <p style={{ color: '#64748b', fontSize: '0.9rem' }}>
            Persistent historical repository of all TESS photometric analyses ({total} records recorded).
          </p>
        </div>

        <button className="btn-secondary" onClick={fetchHistory} title="Refresh">
          <RefreshCw size={15} /> Refresh
        </button>
      </div>

      {error && (
        <div className="alert-error" style={{ marginBottom: 20 }}>
          <AlertCircle size={18} />
          <div>{error}</div>
        </div>
      )}

      {/* Filters & Search */}
      <div className="glass-card" style={{ padding: '1.2rem', marginBottom: 24, display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between' }}>
        <form onSubmit={handleSearchSubmit} style={{ display: 'flex', gap: 10, flex: 1, minWidth: 260 }}>
          <div style={{ position: 'relative', width: '100%' }}>
            <Search size={16} color="#64748b" style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)' }} />
            <input
              type="text"
              className="st-input"
              placeholder="Search by TIC ID..."
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              style={{ paddingLeft: 36 }}
            />
          </div>
          <button type="submit" className="btn-secondary" style={{ padding: '8px 18px' }}>
            Search
          </button>
        </form>

        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '0.8rem', color: '#64748b', fontWeight: 600 }}>Filter by Class:</span>
          {['', 'Exoplanet Transit', 'Eclipsing Binary', 'Background Blend', 'Stellar Noise / Activity'].map(cls => (
            <button
              key={cls}
              onClick={() => setSelectedClass(cls)}
              style={{
                fontSize: '0.76rem',
                padding: '5px 12px',
                borderRadius: 20,
                border: selectedClass === cls ? '1px solid #00f2fe' : '1px solid rgba(255,255,255,0.08)',
                background: selectedClass === cls ? 'rgba(0,242,254,0.15)' : 'transparent',
                color: selectedClass === cls ? '#00f2fe' : '#94a3b8',
                cursor: 'pointer',
                fontWeight: 600
              }}
            >
              {cls || 'All'}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="glass-card" style={{ padding: '1rem', overflowX: 'auto' }}>
        {loading ? (
          <div style={{ padding: '40px 0', textAlign: 'center' }}>
            <Orbit className="animate-orbit" size={32} color="#00f2fe" style={{ margin: '0 auto 12px' }} />
            <div style={{ color: '#64748b', fontSize: '0.85rem' }}>Querying database records...</div>
          </div>
        ) : items.length === 0 ? (
          <div style={{ padding: '60px 0', textAlign: 'center', color: '#4a5568' }}>
            <History size={40} style={{ margin: '0 auto 12px', opacity: 0.3 }} />
            <div style={{ fontWeight: 600, fontSize: '1rem', color: '#94a3b8', marginBottom: 6 }}>No analyses found</div>
            <div style={{ fontSize: '0.85rem' }}>
              {searchTerm || selectedClass ? 'No records match the current filter.' : 'Run your first analysis to see historical records here.'}
            </div>
          </div>
        ) : (
          <table className="st-table">
            <thead>
              <tr>
                <th>TIC ID</th>
                <th>Classification</th>
                <th>Confidence</th>
                <th>Orbital Period</th>
                <th>SNR</th>
                <th>Data Source</th>
                <th>Timestamp</th>
                <th style={{ textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {items.map(item => {
                const color = CLASS_COLORS[item.predicted_class] || '#64748b'
                return (
                  <tr
                    key={item.analysis_id}
                    onClick={() => navigate(`/dashboard/result/${item.analysis_id}`)}
                    style={{ cursor: 'pointer' }}
                  >
                    <td style={{ fontFamily: 'JetBrains Mono, monospace', color: '#00f2fe', fontWeight: 700 }}>
                      TIC {item.tic_id}
                    </td>
                    <td>
                      <span style={{
                        display: 'inline-block', padding: '3px 10px', borderRadius: 20,
                        background: color + '18', color, border: `1px solid ${color}35`,
                        fontSize: '0.74rem', fontWeight: 600
                      }}>
                        {item.predicted_class || '—'}
                      </span>
                    </td>
                    <td style={{ color: '#e2e8f0', fontWeight: 600 }}>
                      {item.confidence ? (item.confidence * 100).toFixed(1) + '%' : '—'}
                    </td>
                    <td style={{ fontFamily: 'JetBrains Mono, monospace', color: '#94a3b8' }}>
                      {item.orbital_period_days ? item.orbital_period_days.toFixed(3) + ' d' : '—'}
                    </td>
                    <td style={{ color: item.snr > 10 ? '#06d6a0' : '#ffd166', fontWeight: 600 }}>
                      {item.snr ? item.snr.toFixed(1) : '—'}
                    </td>
                    <td style={{ color: '#64748b', fontSize: '0.8rem', maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {item.data_source || '—'}
                    </td>
                    <td style={{ color: '#64748b', fontSize: '0.78rem' }}>
                      {item.created_at ? new Date(item.created_at).toLocaleString() : '—'}
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
                        <button
                          className="btn-secondary"
                          style={{ padding: '6px 10px', fontSize: '0.75rem' }}
                          onClick={(e) => {
                            e.stopPropagation()
                            navigate(`/dashboard/result/${item.analysis_id}`)
                          }}
                          title="View Full Report"
                        >
                          <ExternalLink size={13} />
                        </button>
                        <button
                          className="btn-danger"
                          style={{ padding: '6px 10px', fontSize: '0.75rem' }}
                          onClick={(e) => handleDelete(item.analysis_id, e)}
                          disabled={deletingId === item.analysis_id}
                          title="Delete Analysis"
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
