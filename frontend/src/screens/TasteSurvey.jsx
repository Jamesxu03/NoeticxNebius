import { useState } from 'react'
import { api } from '../api'

// The brief's survey steps 2 and 3: rate ~12 dishes (GET /survey/dishes, POST ratings), then pick between
// the same dish at three price/vibe tiers (GET /survey/venue-rounds, POST venue-choice). Each answer
// updates the backend profile, which is handed back through onProfile.
export default function TasteSurvey({ sync, onProfile }) {
  const [step,setStep]=useState('idle'), [dishes,setDishes]=useState([]), [stars,setStars]=useState({}), [rounds,setRounds]=useState([]), [round,setRound]=useState(0), [busy,setBusy]=useState(false), [error,setError]=useState('')
  async function go(fn){setBusy(true);setError('');try{await fn()}catch(e){setError(e.message)}finally{setBusy(false)}}
  const start=()=>go(async()=>{const p=await sync();setDishes(await api.surveyDishes(p.id));setStars({});setStep('rate')})
  const saveRatings=()=>go(async()=>{const p=await sync();const ratings=Object.entries(stars).map(([dish_id,s])=>({dish_id,stars:s}));if(ratings.length)onProfile(await api.postRatings(p.id,ratings));const r=await api.venueRounds(p.id);setRounds(r);setRound(0);setStep(r.length?'venues':'done')})
  const pick=(r,chosen)=>go(async()=>{const p=await sync();onProfile(await api.postVenueChoice(p.id,{round_id:r.round_id,options:r.options.map(o=>o.restaurant.id),chosen}));if(round+1<rounds.length)setRound(round+1);else setStep('done')})
  return <section className="panel survey-panel">
    <div className="section-heading"><div><h2>Tune your taste profile</h2><p>Rate a few dishes, then tell us where you would rather eat. Two minutes, much better matches.</p></div>{step==='rate'&&<span className="pill">{Object.keys(stars).length}/{dishes.length} RATED</span>}{step==='venues'&&<span className="pill">ROUND {round+1} OF {rounds.length}</span>}</div>
    {error&&<div role="alert" className="message error">{error}</div>}
    {step==='idle'&&<button className="primary" disabled={busy} onClick={start}>Start the taste quiz ↗</button>}
    {step==='rate'&&<><div className="survey-grid">{dishes.map(d=><div className="survey-dish" key={d.id}><strong>{d.name}</strong><small>{d.cuisine} · {d.dish_family}</small><div className="stars" role="radiogroup" aria-label={`Rate ${d.name}`}>{[1,2,3,4,5].map(n=><button key={n} role="radio" aria-checked={stars[d.id]===n} aria-label={`${n} star${n>1?'s':''}`} className={n<=(stars[d.id]||0)?'on':''} onClick={()=>setStars({...stars,[d.id]:n})}>★</button>)}</div></div>)}</div><div className="button-row"><button className="primary" disabled={busy} onClick={saveRatings}>{Object.keys(stars).length?'Save ratings and continue →':'Skip to places →'}</button></div></>}
    {step==='venues'&&rounds[round]&&<><p className="intro">Same dish, three places. Where would you go for <strong>{rounds[round].dish_family.replaceAll('_',' ')}</strong>?</p><div className="venue-options">{rounds[round].options.map(o=><button key={o.restaurant.id} className="venue-option" disabled={busy} onClick={()=>pick(rounds[round],o.restaurant.id)}><span className="pill">{'£'.repeat(o.restaurant.price_tier)}</span><strong>{o.restaurant.name}</strong><small>{o.restaurant.vibe_tags.join(' · ')}</small><span>{o.dish.name} · £{o.dish.price_gbp.toFixed(2)}</span></button>)}</div></>}
    {step==='done'&&<><p className="message">Thanks! Your profile is updated, so your recommendations and menu matches now use it.</p><button className="secondary" onClick={start}>Retake the quiz</button></>}
  </section>
}
