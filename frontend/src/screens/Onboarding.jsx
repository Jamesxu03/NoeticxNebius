import { useEffect, useRef, useState } from 'react'
import { api, ALLERGENS, DIETARY } from '../api'
import { Button, Chip, Photo, ProgressHeader, Stars, Switch, cap, pretty, price } from '../ui/kit'
import { Check, Close, Upload } from '../ui/icons'

export const STEPS = ['About you', 'Your flavours', 'Rate dishes', 'Pick a place', 'Order history']

// Brief screens 2-5: profile, tolerances, dish ratings, venue picks (+ optional Uber Eats import).
// Every step writes to the backend so the next step (and the taste map) learns from it.
export default function Onboarding({ profile, setProfile, sync, dims, run, step, setStep, onDone }) {
  const back = step > 0 ? () => setStep(step - 1) : null
  const body = [
    <AboutYou key="0" profile={profile} onNext={(form) => run('Saving your profile…', async () => {
      const p = await sync({ ...profile, ...form }); setProfile(p); setStep(1)
    })} />,
    <Flavours key="1" dims={dims} initial={profile?.taste_vector} onNext={(taste) => run('Tuning your flavours…', async () => {
      const p = await sync(profile); setProfile(await api.setTolerances(p.id, taste)); setStep(2)
    })} />,
    <RateDishes key="2" profile={profile} run={run} onNext={(ratings) => run('Learning from your ratings…', async () => {
      const p = await sync(profile)
      setProfile(ratings.length ? await api.postRatings(p.id, ratings) : p); setStep(3)
    })} />,
    <VenuePicks key="3" profile={profile} run={run} sync={sync} setProfile={setProfile} onNext={() => setStep(4)} />,
    <OrderHistory key="4" profile={profile} run={run} sync={sync} setProfile={setProfile} onNext={onDone} />,
  ][step]
  return (
    <div className="onboarding">
      <ProgressHeader steps={[...STEPS, 'Your taste profile']} index={step} onBack={back} />
      {body}
    </div>
  )
}

function AboutYou({ profile, onNext }) {
  const [form, setForm] = useState({
    first_name: profile?.first_name || '', dietary: profile?.dietary || [], allergies: profile?.allergies || [],
    disliked_ingredients: profile?.disliked_ingredients || [], strict: !!profile?.strict,
  })
  const [avoid, setAvoid] = useState('')
  const toggle = (field, v) => setForm({ ...form, [field]: form[field].includes(v) ? form[field].filter((x) => x !== v) : [...form[field], v] })
  const addAvoid = () => { const v = avoid.trim().toLowerCase(); if (v && !form.disliked_ingredients.includes(v)) setForm({ ...form, disliked_ingredients: [...form.disliked_ingredients, v] }); setAvoid('') }
  return (
    <section className="step">
      <h1 className="step-title">Good food starts <em>with you.</em></h1>
      <p className="step-lede">We never show you dishes that clash with these. Unconfirmed dishes come with a clear warning.</p>
      <label className="field"><span>First name</span>
        <input value={form.first_name} maxLength={40} onChange={(e) => setForm({ ...form, first_name: e.target.value })} />
      </label>
      <fieldset className="group"><legend>How do you eat?</legend>
        <div className="chips">{DIETARY.map((d) => <Chip key={d} on={form.dietary.includes(d)} onClick={() => toggle('dietary', d)}>{cap(d)}</Chip>)}</div>
      </fieldset>
      <fieldset className="group"><legend>Allergies</legend>
        <div className="chips">{ALLERGENS.map((a) => <Chip key={a} on={form.allergies.includes(a)} onClick={() => toggle('allergies', a)}>{cap(a)}</Chip>)}</div>
      </fieldset>
      <fieldset className="group"><legend>Anything you’d rather avoid?</legend>
        <div className="tag-input">
          {form.disliked_ingredients.map((x) => (
            <span key={x} className="tag">{x}<button type="button" aria-label={`Remove ${x}`} onClick={() => setForm({ ...form, disliked_ingredients: form.disliked_ingredients.filter((y) => y !== x) })}><Close size={14} /></button></span>
          ))}
          <input value={avoid} placeholder={form.disliked_ingredients.length ? 'Add another' : 'e.g. coriander, mushroom'} onChange={(e) => setAvoid(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ',') { e.preventDefault(); addAvoid() } }} onBlur={addAvoid} />
        </div>
      </fieldset>
      <div className="group"><Switch checked={form.strict} onChange={(v) => setForm({ ...form, strict: v })}
        label="Only show confirmed-safe dishes" hint="Stricter: hides anything not yet verified for your diet or allergies." /></div>
      <div className="sticky-cta"><Button disabled={!form.first_name.trim()} onClick={() => onNext({ ...form, first_name: form.first_name.trim() })}>Continue</Button></div>
    </section>
  )
}

function Flavours({ dims, initial, onNext }) {
  const [taste, setTaste] = useState(() => initial || Object.fromEntries(dims.map((d) => [d.key, 5])))
  return (
    <section className="step">
      <h1 className="step-title">Turn up <em>your flavour.</em></h1>
      <p className="step-lede">Slide, or tap an ingredient to set its benchmark. The same anchors train our taste model.</p>
      <div className="flavour-list">
        {dims.map((d) => (
          <div className="flavour" key={d.key}>
            <div className="flavour-head"><strong>{d.label}</strong><output>{Number(taste[d.key]).toFixed(1)}</output></div>
            <input type="range" min="0" max="10" step="0.1" value={taste[d.key]} aria-label={`${d.label}, ${d.meaning}`}
              style={{ '--fill': `${taste[d.key] * 10}%` }} onChange={(e) => setTaste({ ...taste, [d.key]: Number(e.target.value) })} />
            <div className="anchors">
              {d.anchors.map((a) => (
                <button key={a.level} type="button" className={Math.abs(taste[d.key] - a.score) < 0.3 ? 'on' : ''}
                  onClick={() => setTaste({ ...taste, [d.key]: a.score })}>{a.ingredient}</button>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="sticky-cta"><Button onClick={() => onNext(taste)}>Continue</Button></div>
    </section>
  )
}

function RateDishes({ profile, run, onNext }) {
  const [dishes, setDishes] = useState(null)
  const [i, setI] = useState(0)
  const [ratings, setRatings] = useState({})
  const timer = useRef()
  useEffect(() => { run('Choosing dishes for you…', async () => setDishes(await api.surveyDishes(profile.id))) }, [profile?.id])
  useEffect(() => () => clearTimeout(timer.current), [])
  if (!dishes) return <section className="step"><div className="dish-card skeleton" /></section>
  const done = () => onNext(Object.entries(ratings).map(([dish_id, stars]) => ({ dish_id, stars })))
  const d = dishes[i]
  const rate = (n) => { setRatings({ ...ratings, [d.id]: n }); clearTimeout(timer.current); timer.current = setTimeout(() => (i + 1 < dishes.length ? setI(i + 1) : null), 280) }
  return (
    <section className="step">
      <h1 className="step-title">Love it? <em>Rate it.</em></h1>
      <p className="step-lede">Skip anything you haven’t tried. Each rating teaches your profile.</p>
      {d ? (
        <article className="dish-card" key={d.id}>
          <Photo src={d.image} alt={d.name} className="dish-photo"><span className="pill-on-photo">{cap(d.cuisine)} · {pretty(d.dish_family)}</span></Photo>
          <div className="dish-card-body">
            <span className="counter">{i + 1} / {dishes.length}</span>
            <h2>{d.name}</h2>
            <p>{d.ingredients.slice(0, 6).join(' · ')}</p>
            <Stars value={ratings[d.id] || 0} onRate={rate} label={`Rate ${d.name}`} />
            <div className="dish-card-actions">
              <button className="link" disabled={i === 0} onClick={() => setI(i - 1)}>Previous</button>
              <button className="link" onClick={() => (i + 1 < dishes.length ? setI(i + 1) : done())}>{ratings[d.id] ? 'Next' : 'Haven’t tried it'}</button>
            </div>
          </div>
        </article>
      ) : <p className="empty">No dishes match your restrictions yet.</p>}
      <div className="dots" aria-hidden="true">{dishes.map((x, j) => <span key={x.id} className={ratings[x.id] ? 'rated' : j === i ? 'now' : ''} />)}</div>
      <div className="sticky-cta"><Button onClick={done}>{Object.keys(ratings).length ? `Save ${Object.keys(ratings).length} rating${Object.keys(ratings).length > 1 ? 's' : ''}` : 'Skip for now'}</Button></div>
    </section>
  )
}

function VenuePicks({ profile, run, sync, setProfile, onNext }) {
  const [rounds, setRounds] = useState(null)
  const [i, setI] = useState(0)
  const [chosen, setChosen] = useState('')
  useEffect(() => { run('Finding places to compare…', async () => { const r = await api.venueRounds(profile.id); setRounds(r); if (!r.length) onNext() }) }, [profile?.id])
  if (!rounds) return <section className="step"><div className="venue-card skeleton" /></section>
  const round = rounds[i]
  if (!round) return null
  const pick = (id) => {
    setChosen(id)
    run('Learning your kind of place…', async () => {
      const p = await sync(profile)
      setProfile(await api.postVenueChoice(p.id, { round_id: round.round_id, options: round.options.map((o) => o.restaurant.id), chosen: id }))
      setTimeout(() => { setChosen(''); i + 1 < rounds.length ? setI(i + 1) : onNext() }, 350)
    })
  }
  return (
    <section className="step">
      <h1 className="step-title">Same craving. <em>Your kind of place.</em></h1>
      <p className="step-lede">Round {i + 1} of {rounds.length}: where would you go for <strong>{pretty(round.dish_family)}</strong>?</p>
      <div className="venue-list">
        {round.options.map(({ restaurant: r, dish: d }) => (
          <button key={r.id} className={`venue-card ${chosen === r.id ? 'chosen' : ''}`} onClick={() => pick(r.id)} disabled={!!chosen}>
            <Photo src={r.image} alt={r.name} className="venue-photo">
              <span className="price-tag">{price(r.price_tier)}</span>
              {chosen === r.id && <span className="chosen-mark"><Check size={22} /></span>}
            </Photo>
            <span className="venue-info"><strong>{r.name}</strong><small>{r.area} · {r.vibe_tags.map(pretty).join(', ')}</small>
              <span>{d.name} · £{d.price_gbp.toFixed(2)}</span></span>
          </button>
        ))}
      </div>
    </section>
  )
}

function OrderHistory({ profile, run, sync, setProfile, onNext }) {
  const input = useRef()
  const [summary, setSummary] = useState(null)
  const upload = (file) => file && run('Reading your order history…', async () => {
    const p = await sync(profile); const res = await api.importUber(p.id, file); setProfile(res.profile); setSummary(res.summary)
  })
  return (
    <section className="step">
      <h1 className="step-title">Already order <em>on Uber Eats?</em></h1>
      <p className="step-lede">Your past orders are a head start. Upload the zip from Uber’s “Download your data”, or just the Eats order CSV.</p>
      <div className="import-card">
        <Upload size={30} />
        <strong>Import order history</strong>
        <small>Only the food-order file is read. The file isn’t stored, only the flavours we learn from it.</small>
        <Button kind="secondary" onClick={() => input.current.click()}>Choose file</Button>
        <input ref={input} hidden type="file" accept=".zip,.csv" onChange={(e) => { upload(e.target.files[0]); e.target.value = '' }} />
      </div>
      {summary && (
        <div className="import-result">
          <strong><Check size={18} /> {summary.items} items from {summary.restaurants} restaurants</strong>
          <p>{Object.entries(summary.cuisines).map(([c, n]) => `${cap(c)} ${n}`).join(' · ')}</p>
          <div className="chips small">{summary.top_ingredients.slice(0, 6).map((x) => <span key={x} className="chip on static">{x}</span>)}</div>
        </div>
      )}
      <div className="sticky-cta"><Button onClick={onNext}>{summary ? 'See my taste profile' : 'Skip and see my profile'}</Button></div>
    </section>
  )
}
