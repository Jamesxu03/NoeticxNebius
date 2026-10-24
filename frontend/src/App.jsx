import { useState } from 'react'

// Screen order from BRIEF.md section 2. Replace each placeholder with a real component in src/screens/.
const SCREENS = [
  { key: 'login', title: 'Log in' },
  { key: 'profile', title: 'About you' },
  { key: 'tolerances', title: 'Your tolerances', survey: true },
  { key: 'ratings', title: 'Rate these dishes', survey: true },
  { key: 'venues', title: 'Pick a place', survey: true },
  { key: 'taste-map', title: 'Your taste profile' },
  { key: 'recommend', title: 'What are you eating out for?' },
]

function ProgressHeader({ index }) {
  const steps = SCREENS.filter((s) => s.survey)
  const pos = steps.findIndex((s) => s.key === SCREENS[index].key)
  if (pos < 0) return null
  const next = SCREENS[index + 1]?.title
  return (
    <div className="progress">
      <div className="bar"><div style={{ width: `${((pos + 1) / steps.length) * 100}%` }} /></div>
      <small>Step {pos + 1} of {steps.length}{next ? ` · next: ${next}` : ''}</small>
    </div>
  )
}

export default function App() {
  const [index, setIndex] = useState(0)
  const screen = SCREENS[index]
  return (
    <div className="stage">
      <div className="phone">
        <div className="notch" />
        <div className="screen">
          <ProgressHeader index={index} />
          <h1>{screen.title}</h1>
          <p className="placeholder">TODO: {screen.key} screen</p>
          <div className="nav">
            <button disabled={index === 0} onClick={() => setIndex(index - 1)}>Back</button>
            <button disabled={index === SCREENS.length - 1} onClick={() => setIndex(index + 1)}>Next</button>
          </div>
        </div>
      </div>
    </div>
  )
}
