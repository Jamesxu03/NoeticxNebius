import { useEffect, useRef, useState } from 'react'
import { api, ALLERGENS, DIETARY } from '../api'
import { cloud } from '../cloud'
import { Button, Chip, NavHeader, Switch, cap } from '../ui/kit'
import { Chevron, Upload, Sparkle, Cloud, Leaf } from '../ui/icons'
import TasteCard from './TasteCard'

// Brief screen 7 (taste profile) plus settings: restrictions, retake, import, cloud save.
export default function Palate({ profile, setProfile, sync, run, toast, onRetake, menus }) {
  const [map, setMap] = useState(null)
  const [edit, setEdit] = useState(false)
  const file = useRef()
  useEffect(() => { if (profile) run('', async () => { const p = await sync(profile); setMap(await api.tasteMap(p.id)) }) }, [profile?.taste_vector, profile?.dish_ratings?.length, profile?.venue_choices?.length])

  const toggle = (field, v) => setProfile({ ...profile, [field]: (profile[field] || []).includes(v) ? profile[field].filter((x) => x !== v) : [...(profile[field] || []), v] })
  const save = () => run('Saving your preferences…', async () => { setProfile(await sync(profile)); setEdit(false); toast('Preferences saved.') })
  const importFile = (f) => f && run('Reading your order history…', async () => {
    const p = await sync(profile); const res = await api.importUber(p.id, f); setProfile(res.profile)
    toast(`Imported ${res.summary.items} items from ${res.summary.restaurants} restaurants.`)
  })

  return (
    <div className="tab-page">
      <NavHeader eyebrow="Entirely yours" title="My palate" />
      <TasteCard map={map} name={profile?.first_name} onToast={toast} />

      <h3 className="list-title">Diet & allergies</h3>
      <div className="grouped">
        {!edit ? (
          <button className="row" onClick={() => setEdit(true)}>
            <Leaf size={20} /><span><strong>{[...(profile?.dietary || []), ...(profile?.allergies || []).map((a) => `no ${a}`)].map(cap).join(', ') || 'No restrictions'}</strong>
              <small>{profile?.disliked_ingredients?.length ? `Avoiding ${profile.disliked_ingredients.join(', ')}` : 'Tap to edit'}</small></span><Chevron size={18} />
          </button>
        ) : (
          <div className="row-edit">
            <span className="mini-label">Diet</span>
            <div className="chips">{DIETARY.map((d) => <Chip key={d} on={profile.dietary?.includes(d)} onClick={() => toggle('dietary', d)}>{cap(d)}</Chip>)}</div>
            <span className="mini-label">Allergies</span>
            <div className="chips">{ALLERGENS.map((a) => <Chip key={a} on={profile.allergies?.includes(a)} onClick={() => toggle('allergies', a)}>{cap(a)}</Chip>)}</div>
            <label className="field"><span>Avoid (comma separated)</span>
              <input value={(profile.disliked_ingredients || []).join(', ')} onChange={(e) => setProfile({ ...profile, disliked_ingredients: e.target.value.split(',').map((s) => s.trimStart()) })} /></label>
            <Button onClick={save}>Save</Button>
          </div>
        )}
        <div className="row"><Switch checked={profile?.strict} onChange={(v) => { setProfile({ ...profile, strict: v }); run('', async () => setProfile(await sync({ ...profile, strict: v }))) }}
          label="Only confirmed-safe dishes" hint="Hide anything not yet verified for you" /></div>
      </div>

      <h3 className="list-title">Teach it more</h3>
      <div className="grouped">
        <button className="row" onClick={onRetake}><Sparkle size={20} /><span><strong>Retake the taste quiz</strong><small>{profile?.dish_ratings?.length || 0} dishes rated so far</small></span><Chevron size={18} /></button>
        <button className="row" onClick={() => file.current.click()}><Upload size={20} /><span><strong>Import Uber Eats history</strong><small>{Object.keys(profile?.ordered_ingredients || {}).length ? 'Imported, add more anytime' : 'Zip or Eats order CSV, not stored'}</small></span><Chevron size={18} /></button>
        <input ref={file} hidden type="file" accept=".zip,.csv" onChange={(e) => { importFile(e.target.files[0]); e.target.value = '' }} />
      </div>

      <h3 className="list-title">Take it with you</h3>
      <CloudRow profile={profile} menus={menus} setProfile={setProfile} run={run} toast={toast} />

      <button className="link danger" onClick={() => { if (confirm('Start over? This clears your profile on this device.')) { localStorage.clear(); location.reload() } }}>Start over</button>
    </div>
  )
}

function CloudRow({ profile, menus, setProfile, run, toast }) {
  const [session, setSession] = useState(null)
  const [email, setEmail] = useState(''), [password, setPassword] = useState('')
  useEffect(() => {
    if (!cloud) return
    cloud.auth.getSession().then(({ data }) => setSession(data.session))
    const { data } = cloud.auth.onAuthStateChange((_, s) => setSession(s))
    return () => data.subscription.unsubscribe()
  }, [])
  if (!cloud) return <div className="grouped"><div className="row"><Cloud size={20} /><span><strong>Saved on this device</strong><small>Cloud sync isn’t connected for this demo.</small></span></div></div>
  if (session) return (
    <div className="grouped">
      <div className="row"><Cloud size={20} /><span><strong>{session.user.email}</strong><small>Signed in</small></span></div>
      <div className="row-actions pad">
        <Button kind="secondary" onClick={() => run('Saving to your account…', async () => {
          const { error } = await cloud.from('palatable_profiles').upsert({ user_id: session.user.id, profile, menus, updated_at: new Date().toISOString() })
          if (error) throw error; toast('Saved to your account.')
        })}>Save to cloud</Button>
        <Button kind="plain" onClick={() => cloud.auth.signOut()}>Sign out</Button>
      </div>
    </div>
  )
  const auth = (fn, msg) => run('Signing in…', async () => { const { error } = await fn({ email, password }); if (error) throw error; setPassword(''); if (msg) toast(msg) })
  return (
    <form className="grouped pad" onSubmit={(e) => { e.preventDefault(); auth((c) => cloud.auth.signInWithPassword(c)) }}>
      <input type="email" required placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} />
      <input type="password" required minLength={8} placeholder="Password (8+ characters)" value={password} onChange={(e) => setPassword(e.target.value)} />
      <div className="row-actions"><Button type="submit">Sign in</Button>
        <Button type="button" kind="plain" onClick={() => auth((c) => cloud.auth.signUp(c), 'Check your email to confirm your account.')}>Create account</Button></div>
    </form>
  )
}
