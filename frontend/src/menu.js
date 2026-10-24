export function parseMenu(text) {
  const rows = []
  for (const line of text.split('\n').map(s => s.trim()).filter(Boolean)) {
    if (/^(menu|starters|mains|desserts|sides|drinks|our menu)$/i.test(line)) continue
    const match = line.match(/(?:£|\$|€)\s*(\d+(?:[.,]\d{1,2})?)\s*$/)
    const clean = line.replace(/(?:£|\$|€)\s*\d+(?:[.,]\d{1,2})?\s*$/, '').trim()
    const parts = clean.split(/\s+[—–-]\s+|\s*:\s*/)
    if (parts.length === 1 && !match && rows.length && /^[a-z]/.test(clean)) {
      rows[rows.length - 1].ingredients = [rows.at(-1).ingredients, clean].filter(Boolean).join(', ')
    } else if (clean.length > 2) rows.push({ id: crypto.randomUUID(), name: parts[0], ingredients: parts.slice(1).join(', '), price: match ? match[1].replace(',', '.') : '' })
  }
  return rows
}
export function rankMenu(items, profile) {
  const weights = profile?.ordered_ingredients || {}
  return items.filter(x => x.name.trim()).map(item => {
    const text = `${item.name} ${item.ingredients}`.toLowerCase()
    const matches = Object.keys(weights).filter(key => new RegExp(`\\b${key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'i').test(text))
    const score = matches.reduce((n, key) => n + weights[key], 0)
    return {...item, score, matches}
  }).sort((a,b) => b.score-a.score)
}
