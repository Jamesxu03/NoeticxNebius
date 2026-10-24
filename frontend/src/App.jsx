import { useEffect, useState } from 'react'
import fallbackDims from '../../data/taste_dimensions.json'
import { api, syncProfile } from './api'
import Phone from './ui/Phone'
import { Button, TabBar, Toast } from './ui/kit'
import { Sparkle } from './ui/icons'
import Welcome from './screens/Welcome'
import Onboarding from './screens/Onboarding'
import TasteCard from './screens/TasteCard'
import Discover from './screens/Discover'
import Scan from './screens/Scan'
import Palate from './screens/Palate'
import Saved from './screens/Saved'

// localStorage can be unavailable (private mode); the app still works without it.
const load = (k, fallback) => { try { return JSON.parse(localStorage.getItem(k)) ?? fallback } catch { return fallback } }
const keep = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)) } catch { /* ignore */ } }

export default function App() {
  const [profile, setProfile] = useState(() => load('palatable.profile', null))
  const [stage, setStage] = useState(() => load('palatable.stage', 'welcome')) // welcome | onboarding | reveal | app
  const [step, setStep] = useState(0)
  const [tab, setTab] = useState('discover')
  const [saved, setSaved] = useState(() => load('palatable.saved', []))
  const [menus, setMenus] = useState(() => load('palatable.menus', []))
  const [openMenu, setOpenMenu] = useState(null)
  const [dims, setDims] = useState(fallbackDims.dimensions)
  const [busy, setBusy] = useState(''), [error, setError] = useState(''), [notice, setNotice] = useState('')

  useEffect(() => keep('palatable.profile', profile), [profile])
  useEffect(() => keep('palatable.stage', stage === 'onboarding' ? 'welcome' : stage), [stage])
  useEffect(() => keep('palatable.saved', saved), [saved])
  useEffect(() => keep('palatable.menus', menus), [menus])
  useEffect(() => { api.tasteDimensions().then((d) => setDims(d.dimensions)).catch(() => {}) }, [])
  useEffect(() => { document.getElementById('screen-scroll')?.scrollTo({ top: 0 }) }, [stage, step, tab])
  useEffect(() => { if (!notice && !error) return; const t = setTimeout(() => { setNotice(''); setError('') }, 4500); return () => clearTimeout(t) }, [notice, error])

  // One backend profile per browser (restored after a backend restart, see api.syncProfile).
  const sync = async (local = profile) => { const p = await syncProfile(local, local?.first_name); setProfile(p); return p }
  async function run(label, fn) {
    setBusy(label); setError('')
    try { await fn() } catch (e) { setError(e.message || 'Something went wrong. Please try again.') } finally { setBusy('') }
  }
  const toast = (msg) => setNotice(msg)
  const toggleSave = (rec) => setSaved(saved.some((s) => s.restaurant.id === rec.restaurant.id)
    ? saved.filter((s) => s.restaurant.id !== rec.restaurant.id) : [rec, ...saved])

  let screen
  if (stage === 'welcome') {
    screen = <Welcome returning={profile?.id && profile?.first_name}
      onStart={(name) => { if (name) { setProfile({ ...(profile || {}), first_name: name }); setStep(0); setStage('onboarding') } else setStage('app') }} />
  } else if (stage === 'onboarding') {
    screen = <Onboarding profile={profile} setProfile={setProfile} sync={sync} dims={dims} run={run} step={step} setStep={setStep}
      onDone={() => setStage('reveal')} />
  } else if (stage === 'reveal') {
    screen = <Reveal profile={profile} sync={sync} run={run} toast={toast} onContinue={() => { setTab('discover'); setStage('app') }} />
  } else {
    screen = {
      discover: <Discover profile={profile} sync={sync} run={run} saved={saved} onSave={toggleSave} />,
      scan: <Scan key={openMenu?.id || 'new'} initial={openMenu} profile={profile} sync={sync} run={run} setBusy={setBusy}
        onSaveMenu={({ name, items }) => { setMenus([{ id: crypto.randomUUID(), name, items, date: new Date().toLocaleDateString('en-GB') }, ...menus]); toast('Menu saved.') }} />,
      palate: <Palate profile={profile} setProfile={setProfile} sync={sync} run={run} toast={toast} menus={menus}
        onRetake={() => { setStep(1); setStage('onboarding') }} />,
      saved: <Saved saved={saved} menus={menus} onUnsave={toggleSave} onDiscover={() => setTab('discover')}
        onOpenMenu={(m) => { setOpenMenu(m); setTab('scan') }} onDeleteMenu={(m) => setMenus(menus.filter((x) => x.id !== m.id))} />,
    }[tab]
  }

  return (
    <Phone side={<SidePitch />} tabBar={stage === 'app' && <TabBar tab={tab} onTab={(t) => { if (t === 'scan' && tab !== 'scan') setOpenMenu(null); setTab(t) }} />}>
      <div className={`screen stage-${stage}`}>{screen}</div>
      <Toast busy={busy} error={error} notice={notice} onClose={() => { setError(''); setNotice('') }} />
    </Phone>
  )
}

// Brief screen 7: the taste profile reveal at the end of the survey.
function Reveal({ profile, sync, run, toast, onContinue }) {
  const [map, setMap] = useState(null)
  useEffect(() => { run('Building your taste profile…', async () => { const p = await sync(profile); setMap(await api.tasteMap(p.id)) }) }, [])
  return (
    <div className="reveal">
      <span className="eyebrow light"><Sparkle size={14} /> Your taste profile is ready</span>
      <TasteCard map={map} name={profile?.first_name} onToast={toast} />
      <Button onClick={onContinue}>Find my next meal</Button>
    </div>
  )
}

function SidePitch() {
  return (
    <div className="pitch">
      <span className="wordmark">palatable<Sparkle size={16} /></span>
      <h2>A portable taste profile <em>any delivery app could plug in.</em></h2>
      <ol>
        <li><b>Learn</b> your palate: sliders, dish ratings, venue picks and your Uber Eats history.</li>
        <li><b>Recommend</b> restaurants and dishes for the occasion, scored by taste, Epicure flavour embeddings and TasteNet.</li>
        <li><b>Protect</b>: allergens and diets are checked by code, never guessed by an AI.</li>
      </ol>
      <small>Hackathon prototype · photos: openly licensed, see <a href="/photos/CREDITS.md" target="_blank" rel="noreferrer">credits</a></small>
    </div>
  )
}
