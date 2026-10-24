import { useRef, useState } from 'react'
import { Share } from '../ui/icons'
import { cap, price } from '../ui/kit'

const KEYS = ['spice', 'sweet', 'sour', 'bitter', 'umami', 'richness', 'crunch', 'creaminess']
const C = { ink: '#1f1a17', paprika: '#d9481e', saffron: '#f2a541', grid: '#e9dfd2', muted: '#8a7a6d', paper: '#fbf6ee' }

// Eight-dimension radar (based on Janice's TasteJourney radar), drawn as SVG so it can be exported.
export function Radar({ taste, size = 300, labels = true }) {
  const c = size / 2, r = size * 0.36
  const pt = (i, v) => { const a = (i / KEYS.length) * Math.PI * 2 - Math.PI / 2; return [c + Math.cos(a) * r * v / 10, c + Math.sin(a) * r * v / 10] }
  const poly = (fn) => KEYS.map((k, i) => fn(k, i).join(',')).join(' ')
  return (
    <svg viewBox={`0 0 ${size} ${size}`} width="100%" role="img" aria-label={`Taste radar: ${KEYS.map((k) => `${k} ${Number(taste[k] || 0).toFixed(1)}`).join(', ')}`}>
      {[2.5, 5, 7.5, 10].map((v) => <polygon key={v} points={poly((k, i) => pt(i, v))} fill="none" stroke={C.grid} strokeWidth="1" />)}
      {KEYS.map((k, i) => { const [x, y] = pt(i, 10); return <line key={k} x1={c} y1={c} x2={x} y2={y} stroke={C.grid} /> })}
      <polygon points={poly((k, i) => pt(i, taste[k] || 0))} fill={C.paprika} fillOpacity=".22" stroke={C.paprika} strokeWidth="2.5" strokeLinejoin="round" />
      {KEYS.map((k, i) => { const [x, y] = pt(i, taste[k] || 0); return <circle key={k} cx={x} cy={y} r="3.5" fill={C.paprika} /> })}
      {labels && KEYS.map((k, i) => {
        const [x, y] = pt(i, 12.6)
        return <text key={k} x={x} y={y} textAnchor="middle" dominantBaseline="middle" fontSize={size * 0.04} fill={C.muted}
          fontFamily="Inter Variable, Inter, system-ui, sans-serif">{cap(k)}</text>
      })}
    </svg>
  )
}

// The shareable taste profile: persona, radar, adventurousness, price and vibe.
export default function TasteCard({ map, name, onToast }) {
  const [sharing, setSharing] = useState(false)
  const cardRef = useRef()
  if (!map) return <div className="taste-card skeleton" aria-busy="true" />
  const adv = Number(map.adventurousness || 0)
  const vibes = Object.entries(map.vibe_pref || {}).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([v]) => v)

  async function share() {
    setSharing(true)
    try {
      const blob = await renderCard(map, name)
      const file = new File([blob], 'my-palatable-taste.png', { type: 'image/png' })
      if (navigator.canShare?.({ files: [file] })) await navigator.share({ files: [file], title: 'My Palatable taste profile' })
      else { const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = file.name; a.click(); URL.revokeObjectURL(a.href) }
      onToast?.('Taste card saved as an image.')
    } catch (e) { if (e.name !== 'AbortError') onToast?.('Could not create the image on this device.') }
    finally { setSharing(false) }
  }

  return (
    <article className="taste-card" ref={cardRef}>
      <span className="eyebrow light">{name ? `${name}’s palate` : 'Your palate'}</span>
      <h2 className="persona">{map.persona || 'Your taste profile'}</h2>
      <p className="persona-blurb">{map.persona_blurb}</p>
      <div className="radar-wrap"><Radar taste={map.taste_vector} /></div>
      <div className="top-dims">{map.top_dimensions.map((k, i) => <span key={k} className={i === 0 ? 'lead' : ''}>{cap(k)}</span>)}</div>
      <div className="card-stats">
        <div><strong>{adv.toFixed(1)}</strong><small>Adventurousness</small><span className="meter"><i style={{ width: `${adv * 10}%` }} /></span></div>
        <div><strong>{price(map.price_pref)}</strong><small>Usual spend</small></div>
        <div><strong className="vibe">{vibes.length ? vibes.map(cap).join(' · ') : '—'}</strong><small>Your kind of place</small></div>
      </div>
      <button className="btn btn-glass" onClick={share} disabled={sharing}><Share size={18} /> {sharing ? 'Preparing…' : 'Share my taste card'}</button>
    </article>
  )
}

// Draw the card to a 1080x1350 PNG (Instagram portrait) with canvas.
async function renderCard(map, name) {
  const W = 1080, H = 1350, cv = document.createElement('canvas'); cv.width = W; cv.height = H
  const g = cv.getContext('2d')
  const grad = g.createLinearGradient(0, 0, W, H); grad.addColorStop(0, '#2a1c16'); grad.addColorStop(1, '#5a2414')
  g.fillStyle = grad; g.fillRect(0, 0, W, H)
  await document.fonts?.ready
  g.fillStyle = '#f2a541'; g.font = '600 34px "Inter Variable", Inter, sans-serif'; g.textAlign = 'center'
  g.fillText((name ? `${name.toUpperCase()}’S PALATE` : 'MY PALATE'), W / 2, 130)
  g.fillStyle = '#fbf6ee'; g.font = '600 76px "Fraunces Variable", Fraunces, Georgia, serif'
  g.fillText(map.persona || 'My taste profile', W / 2, 230)
  g.fillStyle = '#e9d9c8'; g.font = '36px "Inter Variable", Inter, sans-serif'; g.fillText(map.persona_blurb || '', W / 2, 295)
  // radar
  const svg = new XMLSerializer().serializeToString(radarSvg(map.taste_vector))
  const img = new Image(); img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg)
  await img.decode(); g.drawImage(img, 140, 340, 800, 800)
  g.fillStyle = '#fbf6ee'; g.font = '600 40px "Inter Variable", Inter, sans-serif'
  g.fillText(`Adventurousness ${Number(map.adventurousness).toFixed(1)}/10  ·  ${price(map.price_pref)}`, W / 2, 1210)
  g.fillStyle = '#c9b6a4'; g.font = '30px "Inter Variable", Inter, sans-serif'; g.fillText('palatable · find your flavour', W / 2, 1290)
  return await new Promise((res) => cv.toBlob(res, 'image/png'))
}

function radarSvg(taste) {
  const size = 400, c = 200, r = 140, ns = 'http://www.w3.org/2000/svg'
  const svg = document.createElementNS(ns, 'svg'); svg.setAttribute('xmlns', ns); svg.setAttribute('viewBox', `0 0 ${size} ${size}`); svg.setAttribute('width', size); svg.setAttribute('height', size)
  const pt = (i, v) => { const a = (i / 8) * Math.PI * 2 - Math.PI / 2; return [c + Math.cos(a) * r * v / 10, c + Math.sin(a) * r * v / 10] }
  const add = (tag, attrs, text) => { const el = document.createElementNS(ns, tag); Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v)); if (text) el.textContent = text; svg.appendChild(el) }
  for (const v of [2.5, 5, 7.5, 10]) add('polygon', { points: KEYS.map((k, i) => pt(i, v).join(',')).join(' '), fill: 'none', stroke: '#ffffff30' })
  add('polygon', { points: KEYS.map((k, i) => pt(i, taste[k] || 0).join(',')).join(' '), fill: '#f2a54155', stroke: '#f2a541', 'stroke-width': 3 })
  KEYS.forEach((k, i) => { const [x, y] = pt(i, 12.7); add('text', { x, y, 'text-anchor': 'middle', 'dominant-baseline': 'middle', 'font-size': 15, fill: '#fbf6ee', 'font-family': 'sans-serif' }, `${cap(k)} ${Number(taste[k] || 0).toFixed(1)}`) })
  return svg
}
