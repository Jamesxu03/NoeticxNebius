import { useState } from 'react'
import { api, OCCASIONS } from '../api'

// Restaurant + dish picks from the curated catalogue (POST /recommend): taste match, Epicure flavour,
// hard dietary/allergen filters. Warnings are shown exactly as the API returns them.
export default function Recommendations({ sync }) {
  const [occasion,setOccasion]=useState(''), [recs,setRecs]=useState(null), [busy,setBusy]=useState(false), [error,setError]=useState('')
  async function load(o){setOccasion(o);setBusy(true);setError('');try{const p=await sync();setRecs(await api.recommend({profile_id:p.id,occasion:o}))}catch(e){setError(e.message)}finally{setBusy(false)}}
  return <section className="recs">
    <div className="section-heading"><div><h2>Restaurants picked for you</h2><p>Choose the occasion. We match our curated restaurants to your taste profile.</p></div></div>
    <div className="chips">{OCCASIONS.map(([key,label])=><button key={key} className={occasion===key?'chip on':'chip'} disabled={busy} onClick={()=>load(key)}>{label}</button>)}</div>
    {error&&<div role="alert" className="message error">{error}</div>}
    {busy&&<p className="muted">Finding your matches…</p>}
    {recs&&!recs.length&&<p className="safety-note">Nothing on our list fits your dietary needs and allergies yet. Try another occasion or review your restrictions.</p>}
    {!busy&&recs?.map(r=><article key={r.restaurant.id+r.kind} className={`rec-card ${r.kind}`}>
      <div className="rec-head"><div><h3>{r.restaurant.name}</h3><p>{r.restaurant.area} · {'£'.repeat(r.restaurant.price_tier)} · {r.restaurant.vibe_tags.join(', ')}</p></div>{r.kind==='stretch'?<span className="pill stretch-pill">TRY SOMETHING NEW</span>:<span className="match-score">{Math.round(r.score*100)}</span>}</div>
      <ul className="rec-dishes">{r.dishes.map(d=><li key={d.id}><span>{d.name}</span><strong>£{d.price_gbp.toFixed(2)}</strong></li>)}</ul>
      <p className="rec-why">{r.explanation}</p>
      {r.because_you_liked&&<small className="because">Because you loved {r.because_you_liked}</small>}
      {!!r.warnings.length&&<ul className="warnings">{r.warnings.map(w=><li key={w}>{w}</li>)}</ul>}
    </article>)}
  </section>
}
