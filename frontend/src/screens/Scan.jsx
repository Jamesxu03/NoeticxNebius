import { useRef, useState } from 'react'
import { createWorker } from 'tesseract.js'
import { api } from '../api'
import { parseMenu, rankMenu } from '../menu'
import { Button, NavHeader, Warnings } from '../ui/kit'
import { Camera, Upload, Receipt, Plus, Close, Sparkle } from '../ui/icons'

const SAMPLE = 'STARTERS\nBurrata — tomato, basil, olive oil £12\nCalamari fritti — squid, lemon, aioli £10\nMAINS\nLamb vindaloo — lamb, chilli, vinegar, garlic £16\nMushroom risotto — rice, mushroom, parmesan £18\nSalmon teriyaki — salmon, soy, ginger £19\nDESSERTS\nPistachio cannoli — ricotta, pistachio £7\nTiramisu — mascarpone, espresso, savoiardi, cocoa £8'

// At the table: photograph a menu (Janice's on-device OCR) or paste it, check the lines,
// then rank them on the backend (safety rules + taste model + Epicure flavour).
export default function Scan({ profile, sync, run, setBusy, onSaveMenu, initial }) {
  const [items, setItems] = useState(initial?.items || [])
  const [name, setName] = useState(initial?.name || 'Tonight’s menu')
  const [text, setText] = useState('')
  const [preview, setPreview] = useState('')
  const [results, setResults] = useState(null)
  const [offline, setOffline] = useState(false)
  const camera = useRef(), upload = useRef()

  async function ocr(file) {
    if (!file?.type.startsWith('image/')) return
    if (preview) URL.revokeObjectURL(preview)
    setPreview(URL.createObjectURL(file)); setResults(null)
    await run('Reading the menu…', async () => {
      let worker
      try {
        worker = await createWorker('eng', 1, { workerPath: '/ocr/worker.min.js', corePath: '/ocr', langPath: '/ocr',
          logger: (m) => m.status === 'recognizing text' && setBusy(`Reading the menu · ${Math.round(m.progress * 100)}%`) })
        const { data } = await worker.recognize(file)
        setText(data.text); setItems(parseMenu(data.text))
      } finally { if (worker) await worker.terminate() }
    })
  }

  function rank(list = items) {
    run('Finding your dish…', async () => {
      const clean = list.filter((i) => i.name.trim())
      try {
        const p = await sync(profile)
        setResults(await api.rankMenu(p.id, clean)); setOffline(false)
      } catch {
        setResults(rankMenu(clean, profile).map((i) => ({ ...i, reason: i.matches.length ? `Shares ${i.matches.join(', ')} with your orders` : 'No match from your history yet', warnings: [], excluded: false })))
        setOffline(true)
      }
      document.getElementById('screen-scroll')?.scrollTo({ top: 0, behavior: 'smooth' })
    })
  }

  const edit = (id, field, value) => setItems(items.map((i) => (i.id === id ? { ...i, [field]: value } : i)))
  const shown = results?.filter((r) => !r.excluded) || []
  const hidden = results?.filter((r) => r.excluded) || []

  if (results) return (
    <div className="tab-page">
      <NavHeader onBack={() => setResults(null)} eyebrow={name} title="Your dish" subtitle={offline ? 'Offline: ranked by your order history only. Allergy and diet checks did not run.' : 'Ranked for your palate. Anything that clashes with your profile is hidden.'} />
      {shown[0] && (
        <article className="winner">
          <span className="eyebrow light"><Sparkle size={14} /> Order this</span>
          <h2>{shown[0].name}</h2>
          <p>{shown[0].reason}</p>
          {shown[0].price && <b>£{shown[0].price}</b>}
          <Warnings items={shown[0].warnings} />
        </article>
      )}
      <ol className="ranked">
        {shown.slice(1).map((r, i) => (
          <li key={r.id}><span className="rank">{i + 2}</span>
            <div><strong>{r.name}</strong><small>{r.reason}</small><Warnings items={r.warnings} /></div>
            {r.price && <b>£{r.price}</b>}
          </li>
        ))}
      </ol>
      {!!hidden.length && <div className="hidden-list"><h3>Hidden for you</h3>
        {hidden.map((r) => <div key={r.id} className="hidden-item"><strong>{r.name}</strong><small>{r.exclusion_reason}</small></div>)}
      </div>}
      <div className="row-actions">
        <Button kind="secondary" onClick={() => { onSaveMenu({ name, items }); }}>Save menu</Button>
        <Button kind="plain" onClick={() => setResults(null)}>Edit items</Button>
      </div>
    </div>
  )

  return (
    <div className="tab-page">
      <NavHeader eyebrow="At the table" title="Scan a menu" subtitle="Snap the menu and we’ll point you to your dish." />
      <div className="scan-hero">
        <button className="scan-primary" onClick={() => camera.current.click()}><Camera size={30} /><strong>Take a photo</strong><small>Text is read on your device</small></button>
        <button className="scan-secondary" onClick={() => upload.current.click()}><Upload size={22} /> Upload a photo</button>
        <input hidden ref={camera} type="file" accept="image/*" capture="environment" onChange={(e) => { ocr(e.target.files[0]); e.target.value = '' }} />
        <input hidden ref={upload} type="file" accept="image/*" onChange={(e) => { ocr(e.target.files[0]); e.target.value = '' }} />
      </div>
      {preview && <img className="menu-preview" src={preview} alt="Your menu photo" />}
      <details className="paste" open={!items.length && !!text}>
        <summary><Receipt size={18} /> Or paste the menu text</summary>
        <textarea aria-label="Menu text" value={text} onChange={(e) => setText(e.target.value)} placeholder={'One dish per line, e.g.\nBurrata — tomato, basil £12'} />
        <div className="row-actions">
          <Button kind="secondary" onClick={() => setItems(parseMenu(text))} disabled={!text.trim()}>Read these lines</Button>
          <Button kind="plain" onClick={() => { setText(SAMPLE); setItems(parseMenu(SAMPLE)) }}>Try a sample menu</Button>
        </div>
      </details>
      {!!items.length && (
        <section className="review">
          <div className="review-head"><h3>Check what we read</h3><span>{items.length} dishes</span></div>
          <label className="field"><span>Menu name</span><input value={name} onChange={(e) => setName(e.target.value)} /></label>
          {items.map((it) => (
            <div className="menu-line" key={it.id}>
              <input aria-label="Dish name" value={it.name} onChange={(e) => edit(it.id, 'name', e.target.value)} />
              <input aria-label="Listed ingredients" value={it.ingredients} placeholder="Ingredients not listed" onChange={(e) => edit(it.id, 'ingredients', e.target.value)} />
              <input aria-label="Price" className="price" value={it.price} placeholder="£" onChange={(e) => edit(it.id, 'price', e.target.value)} />
              <button aria-label={`Remove ${it.name}`} onClick={() => setItems(items.filter((x) => x.id !== it.id))}><Close size={16} /></button>
            </div>
          ))}
          <button className="link add-line" onClick={() => setItems([...items, { id: crypto.randomUUID(), name: '', ingredients: '', price: '' }])}><Plus size={16} /> Add a dish</button>
          <p className="fine">Only listed ingredients are used. Menus rarely list everything, so always confirm allergens with the restaurant.</p>
          <div className="sticky-cta"><Button onClick={() => rank()}>Find my dish</Button></div>
        </section>
      )}
    </div>
  )
}
