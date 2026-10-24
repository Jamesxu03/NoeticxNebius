import { useState } from 'react'
import { Compass, Scan, Palate, Heart, Star, Back, Alert } from './icons'

export const price = (tier) => '£'.repeat(Math.max(1, Math.round(tier || 1)))
export const pretty = (s = '') => s.replaceAll('_', ' ').replaceAll('-', ' ')
export const cap = (s = '') => s.charAt(0).toUpperCase() + s.slice(1)

// A photo that degrades to a warm gradient (never a broken image icon).
export function Photo({ src, alt = '', className = '', children }) {
  const [failed, setFailed] = useState(!src)
  return (
    <div className={`photo ${className}`}>
      {!failed && <img src={src} alt={alt} loading="lazy" onError={() => setFailed(true)} />}
      {failed && <div className="photo-fallback" aria-hidden="true" />}
      {children}
    </div>
  )
}

export function Button({ kind = 'primary', children, ...rest }) {
  return <button className={`btn btn-${kind}`} {...rest}>{children}</button>
}

export function Chip({ on, children, ...rest }) {
  return <button type="button" className={on ? 'chip on' : 'chip'} aria-pressed={!!on} {...rest}>{children}</button>
}

export function Switch({ checked, onChange, label, hint }) {
  return (
    <label className="switch-row">
      <span><strong>{label}</strong>{hint && <small>{hint}</small>}</span>
      <input type="checkbox" role="switch" checked={!!checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="switch" aria-hidden="true" />
    </label>
  )
}

export function Stars({ value = 0, onRate, size = 34, label }) {
  return (
    <div className="stars" role="radiogroup" aria-label={label}>
      {[1, 2, 3, 4, 5].map((n) => (
        <button key={n} type="button" role="radio" aria-checked={value === n} aria-label={`${n} star${n > 1 ? 's' : ''}`}
          className={n <= value ? 'on' : ''} onClick={() => onRate(n)}>
          <Star size={size} filled={n <= value} stroke={1.5} />
        </button>
      ))}
    </div>
  )
}

// Large-title navigation header (iOS), with an optional back button and trailing action.
export function NavHeader({ title, subtitle, onBack, trailing, eyebrow }) {
  return (
    <header className="nav-header">
      <div className="nav-row">
        {onBack ? <button className="nav-back" onClick={onBack} aria-label="Back"><Back size={24} /> Back</button> : <span />}
        {trailing}
      </div>
      {eyebrow && <span className="eyebrow">{eyebrow}</span>}
      <h1 className="large-title">{title}</h1>
      {subtitle && <p className="subtitle">{subtitle}</p>}
    </header>
  )
}

// Survey progress header from the brief: where you are and what comes next.
export function ProgressHeader({ steps, index, onBack }) {
  return (
    <div className="progress-header">
      <div className="nav-row">
        {onBack ? <button className="nav-back" onClick={onBack} aria-label="Back"><Back size={24} /></button> : <span />}
        <span className="progress-count">Step {index + 1} of {steps.length}</span>
      </div>
      <div className="progress-track" aria-hidden="true">
        {steps.map((s, i) => <span key={s} className={i < index ? 'done' : i === index ? 'now' : ''} />)}
      </div>
      {steps[index + 1] && <p className="progress-next">Next: {steps[index + 1]}</p>}
    </div>
  )
}

export function Segmented({ options, value, onChange, label }) {
  return (
    <div className="segmented" role="radiogroup" aria-label={label}>
      {options.map(([v, text]) => (
        <button key={String(v)} type="button" role="radio" aria-checked={value === v} className={value === v ? 'on' : ''}
          onClick={() => onChange(v)}>{text}</button>
      ))}
    </div>
  )
}

export function Warnings({ items }) {
  const [open, setOpen] = useState(false)
  if (!items?.length) return null
  return (
    <div className={`warnings ${open ? 'open' : ''}`}>
      <button type="button" onClick={() => setOpen(!open)} aria-expanded={open}>
        <Alert size={16} /> {items.length} {items.length === 1 ? 'thing' : 'things'} to check with the restaurant
      </button>
      {open && <ul>{items.map((w) => <li key={w}>{w}</li>)}</ul>}
    </div>
  )
}

export function Toast({ busy, error, notice, onClose }) {
  const text = busy || error || notice
  if (!text) return null
  return (
    <div className={`toast ${error ? 'error' : busy ? 'busy' : ''}`} role={error ? 'alert' : 'status'} onClick={busy ? undefined : onClose}>
      {busy && <span className="spinner" aria-hidden="true" />}{text}
    </div>
  )
}

const TABS = [['discover', 'Discover', Compass], ['scan', 'Scan', Scan], ['palate', 'My palate', Palate], ['saved', 'Saved', Heart]]

export function TabBar({ tab, onTab }) {
  return (
    <nav className="tab-bar" aria-label="Main">
      {TABS.map(([key, label, Icon]) => (
        <button key={key} className={tab === key ? 'on' : ''} aria-current={tab === key ? 'page' : undefined} onClick={() => onTab(key)}>
          <Icon size={25} stroke={tab === key ? 2.1 : 1.7} />
          <span>{label}</span>
        </button>
      ))}
    </nav>
  )
}

export function matchLabel(score) {
  return score >= 0.68 ? 'Great match' : score >= 0.55 ? 'Good match' : 'Worth a try'
}
