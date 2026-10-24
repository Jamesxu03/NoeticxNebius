import { NavHeader } from '../ui/kit'
import { Receipt, Chevron, Heart } from '../ui/icons'
import { RecCard } from './Discover'

export default function Saved({ saved, menus, onUnsave, onOpenMenu, onDeleteMenu, onDiscover }) {
  return (
    <div className="tab-page">
      <NavHeader eyebrow="Kept close" title="Saved" />
      <h3 className="list-title">Restaurants</h3>
      {!saved.length && <div className="hint-card"><Heart size={18} /> Tap the heart on a recommendation to keep it here. <button className="link" onClick={onDiscover}>Discover</button></div>}
      <div className="rec-list">{saved.map((r, i) => <RecCard key={r.restaurant.id} rec={r} rank={i} saved onSave={() => onUnsave(r)} />)}</div>
      <h3 className="list-title">Menus</h3>
      {!menus.length && <div className="hint-card"><Receipt size={18} /> Scanned menus you save appear here.</div>}
      <div className="grouped">
        {menus.map((m) => (
          <div className="row" key={m.id}>
            <Receipt size={20} />
            <button className="row-main" onClick={() => onOpenMenu(m)}><strong>{m.name}</strong><small>{m.items.length} dishes · {m.date}</small></button>
            <button className="link danger small" aria-label={`Delete ${m.name}`} onClick={() => onDeleteMenu(m)}>Delete</button>
            <Chevron size={18} />
          </div>
        ))}
      </div>
    </div>
  )
}
