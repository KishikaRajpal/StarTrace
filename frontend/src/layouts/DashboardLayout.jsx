import React, { useState } from 'react'
import { Outlet, NavLink, useNavigate } from 'react-router-dom'
import {
  LayoutDashboard, FlaskConical, History, BrainCircuit,
  Info, ChevronLeft, ChevronRight, Star, Menu, X
} from 'lucide-react'

const NAV_ITEMS = [
  { to: '/dashboard', icon: LayoutDashboard, label: 'Overview', end: true },
  { to: '/dashboard/analyze', icon: FlaskConical, label: 'New Analysis' },
  { to: '/dashboard/history', icon: History, label: 'Analysis History' },
  { to: '/dashboard/model', icon: BrainCircuit, label: 'Model Info' },
  { to: '/dashboard/about', icon: Info, label: 'About StarTrace' },
]

export default function DashboardLayout() {
  const [collapsed, setCollapsed] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const navigate = useNavigate()

  return (
    <div className="flex min-h-screen" style={{ background: '#060a12' }}>
      {/* Sidebar */}
      <aside
        className="hidden md:flex flex-col flex-shrink-0 transition-all duration-300"
        style={{
          width: collapsed ? 68 : 240,
          background: 'rgba(9,13,24,0.98)',
          borderRight: '1px solid rgba(0,242,254,0.1)',
          position: 'sticky', top: 0, height: '100vh', overflow: 'hidden'
        }}
      >
        {/* Logo */}
        <div className="flex items-center gap-3 px-4 py-5" style={{ borderBottom: '1px solid rgba(0,242,254,0.08)', minHeight: 72 }}>
          <div className="flex-shrink-0 w-9 h-9 rounded-xl flex items-center justify-center"
            style={{ background: 'linear-gradient(135deg,#00f2fe,#9055ff)', boxShadow: '0 0 18px rgba(0,242,254,0.4)' }}>
            <Star size={18} fill="#060a12" color="#060a12" />
          </div>
          {!collapsed && (
            <div>
              <div className="font-black text-sm tracking-wider gradient-text">STARTRACE</div>
              <div className="text-xs" style={{ color: '#64748b', fontSize: '0.68rem' }}>AI Exoplanet Platform</div>
            </div>
          )}
        </div>

        {/* Nav items */}
        <nav className="flex-1 px-3 py-4 flex flex-col gap-1">
          {NAV_ITEMS.map(({ to, icon: Icon, label, end }) => (
            <NavLink
              key={to} to={to} end={end}
              className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}
              title={collapsed ? label : undefined}
            >
              <Icon size={18} className="flex-shrink-0" />
              {!collapsed && <span>{label}</span>}
            </NavLink>
          ))}
        </nav>

        {/* Collapse toggle */}
        <button
          onClick={() => setCollapsed(c => !c)}
          className="mx-auto mb-4 w-8 h-8 rounded-full flex items-center justify-center transition-all"
          style={{ background: 'rgba(0,242,254,0.08)', border: '1px solid rgba(0,242,254,0.2)', color: '#00f2fe' }}
        >
          {collapsed ? <ChevronRight size={15} /> : <ChevronLeft size={15} />}
        </button>
      </aside>

      {/* Mobile header */}
      <div className="md:hidden fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-4 py-3"
        style={{ background: 'rgba(9,13,24,0.98)', borderBottom: '1px solid rgba(0,242,254,0.1)' }}>
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ background: 'linear-gradient(135deg,#00f2fe,#9055ff)' }}>
            <Star size={15} fill="#060a12" color="#060a12" />
          </div>
          <span className="font-black text-sm gradient-text tracking-wider">STARTRACE</span>
        </div>
        <button onClick={() => setMobileOpen(o => !o)} style={{ color: '#94a3b8' }}>
          {mobileOpen ? <X size={22} /> : <Menu size={22} />}
        </button>
      </div>

      {/* Mobile nav drawer */}
      {mobileOpen && (
        <div className="md:hidden fixed inset-0 z-40" onClick={() => setMobileOpen(false)}>
          <div className="fixed top-14 left-0 bottom-0 w-64 px-3 py-4 flex flex-col gap-1"
            style={{ background: 'rgba(9,13,24,0.98)', borderRight: '1px solid rgba(0,242,254,0.1)' }}
            onClick={e => e.stopPropagation()}>
            {NAV_ITEMS.map(({ to, icon: Icon, label, end }) => (
              <NavLink key={to} to={to} end={end}
                className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}
                onClick={() => setMobileOpen(false)}>
                <Icon size={18} />
                <span>{label}</span>
              </NavLink>
            ))}
          </div>
        </div>
      )}

      {/* Main content */}
      <main className="flex-1 md:overflow-y-auto pt-14 md:pt-0" style={{ minWidth: 0 }}>
        <Outlet />
      </main>
    </div>
  )
}
