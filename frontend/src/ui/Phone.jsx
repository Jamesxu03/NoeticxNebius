import { useEffect, useState } from 'react'
import { StatusGlyphs } from './icons'

// iPhone shell. On a desktop the app sits in a device frame with a short pitch beside it;
// on a real phone (or a narrow window) it fills the screen and respects the safe areas.
export default function Phone({ children, tabBar, side }) {
  const [time, setTime] = useState(clock())
  useEffect(() => { const t = setInterval(() => setTime(clock()), 15000); return () => clearInterval(t) }, [])
  return (
    <div className="stage">
      {side && <aside className="stage-side">{side}</aside>}
      <div className="device" role="application" aria-label="Palatable iPhone app">
        <div className="device-screen">
          <div className="status-bar" aria-hidden="true">
            <span className="status-time">{time}</span>
            <span className="island" />
            <StatusGlyphs />
          </div>
          <div className="screen-scroll" id="screen-scroll">{children}</div>
          {tabBar}
          <div className="home-indicator" aria-hidden="true" />
        </div>
      </div>
    </div>
  )
}

function clock() {
  return new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })
}
