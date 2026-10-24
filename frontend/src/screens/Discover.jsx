import { useEffect, useState } from 'react'
import { api, OCCASIONS } from '../api'
import { NavHeader, Photo, Segmented, Warnings, matchLabel, pretty, price } from '../ui/kit'
import { Heart, Sparkle, Pin } from '../ui/icons'

const FALLBACK = OCCASIONS.map(([id, label]) => ({ id, label, blurb: '', image: `/photos/${id}.jpg` }))

// Brief screen 8: "What occasion are you eating out for?" -> restaurant + dish picks with reasons.
export default function Discover({ profile, sync, run, saved, onSave }) {
  const [occasions, setOccasions] = useState(FALLBACK)
  const [occasion, setOccasion] = useState('')
  const [budget, setBudget] = useState(null)
  const [recs, setRecs] = useState(null)
  useEffect(() => { api.occasions().then(setOccasions).catch(() => {}) }, [])
  useEffect(() => { if (occasion) load(occasion, budget) }, [occasion, budget])

  function load(o, b) {
    run('Finding your next good meal…', async () => {
      const p = await sync(profile)
      setRecs(await api.recommend({ profile_id: p.id, occasion: o, ...(b ? { budget: b } : {}) }))
      document.getElementById('screen-scroll')?.scrollTo({ top: 330, behavior: 'smooth' })
    })
  }

  const hour = new Date().getHours()
  const greeting = hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening'
  return (
    <div className="tab-page">
      <NavHeader eyebrow={new Date().toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'long' })}
        title={`${greeting}, ${profile?.first_name || 'there'}`} subtitle="What are you eating out for?" />
      <div className="occasion-rail" role="radiogroup" aria-label="Occasion">
        {occasions.map((o) => (
          <button key={o.id} role="radio" aria-checked={occasion === o.id} className={`occasion ${occasion === o.id ? 'on' : ''}`} onClick={() => setOccasion(o.id)}>
            <Photo src={o.image} alt="" className="occasion-photo"><span className="occasion-shade" /></Photo>
            <span className="occasion-text"><strong>{o.label}</strong>{o.blurb && <small>{o.blurb}</small>}</span>
          </button>
        ))}
      </div>
      <div className="budget-row">
        <span>Budget</span>
        <Segmented label="Budget" value={budget} onChange={setBudget} options={[[null, 'Any'], [1, '£'], [2, '££'], [3, '£££']]} />
      </div>

      {!recs && <div className="hint-card"><Sparkle size={20} /> Pick an occasion and we’ll match restaurants to your palate, with your allergies always checked.</div>}
      {recs && !recs.length && <div className="hint-card">Nothing fits your diet and budget for this one yet. Try another budget or occasion.</div>}
      <div className="rec-list">
        {recs?.map((r, i) => <RecCard key={r.restaurant.id + r.kind} rec={r} rank={i} saved={saved.some((s) => s.restaurant.id === r.restaurant.id)} onSave={() => onSave(r)} />)}
      </div>
    </div>
  )
}

export function RecCard({ rec: r, rank, saved, onSave }) {
  const stretch = r.kind === 'stretch'
  return (
    <article className={`rec-card ${stretch ? 'stretch' : ''}`}>
      <Photo src={r.restaurant.image} alt={r.restaurant.name} className="rec-photo">
        <span className="rec-shade" />
        <span className="rec-badge">{stretch ? <><Sparkle size={14} /> Try something new</> : rank === 0 ? 'Top pick for you' : matchLabel(r.score)}</span>
        {onSave && <button className={`save ${saved ? 'on' : ''}`} onClick={onSave} aria-label={saved ? 'Remove from saved' : 'Save restaurant'}><Heart size={20} filled={saved} /></button>}
        <span className="rec-title"><strong>{r.restaurant.name}</strong>
          <small><Pin size={13} /> {r.restaurant.area} · {price(r.restaurant.price_tier)} · {r.restaurant.vibe_tags.map(pretty).join(', ')}</small></span>
      </Photo>
      <div className="rec-body">
        <ul className="dish-rows">
          {r.dishes.map((d, j) => (
            <li key={d.id}>
              <Photo src={d.image} alt="" className="dish-thumb" />
              <span><strong>{d.name}</strong><small>{j === 0 ? 'Order this' : d.category === 'dessert' ? 'To finish' : 'On the side'}</small></span>
              <b>£{d.price_gbp.toFixed(2)}</b>
            </li>
          ))}
        </ul>
        <p className="why">{r.explanation}</p>
        {r.because_you_liked && <span className="because"><Heart size={14} filled /> Because you loved {r.because_you_liked}</span>}
        <Warnings items={r.warnings} />
      </div>
    </article>
  )
}
