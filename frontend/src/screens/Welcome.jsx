import { useState } from 'react'
import { Button, Photo } from '../ui/kit'
import { Sparkle } from '../ui/icons'

// Screen 1 (brief: login). A light sign-in: name only, stored on this device.
export default function Welcome({ onStart, returning }) {
  const [name, setName] = useState('')
  const [asking, setAsking] = useState(false)
  return (
    <div className="welcome">
      <Photo src="/photos/hero.jpg" className="welcome-photo">
        <div className="welcome-shade" />
        <div className="welcome-brand"><span className="wordmark">palatable</span><Sparkle size={18} /></div>
      </Photo>
      <div className="welcome-body">
        <h1 className="display">Less guessing.<br /><em>More your flavour.</em></h1>
        <p>A two-minute taste quiz, then restaurant picks and menu matches made for your palate, with your allergies always respected.</p>
        {!asking ? (
          <>
            <Button onClick={() => setAsking(true)}>Get started</Button>
            {returning && <Button kind="plain" onClick={() => onStart(null)}>Continue as {returning}</Button>}
          </>
        ) : (
          <form className="welcome-form" onSubmit={(e) => { e.preventDefault(); if (name.trim()) onStart(name.trim()) }}>
            <label className="field"><span>What should we call you?</span>
              <input autoFocus required maxLength={40} autoComplete="given-name" value={name} onChange={(e) => setName(e.target.value)} placeholder="First name" />
            </label>
            <Button type="submit" disabled={!name.trim()}>Continue</Button>
          </form>
        )}
        <small className="fine">No password needed. Your profile stays on this device unless you choose to save it to the cloud.</small>
      </div>
    </div>
  )
}
