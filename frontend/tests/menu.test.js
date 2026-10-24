import { test } from 'node:test'
import assert from 'node:assert/strict'
import { parseMenu, rankMenu } from '../src/menu.js'

test('extracts names, explicitly listed ingredients and prices without inventing ingredients', () => {
  const items = parseMenu('MAINS\nBurrata — tomato, basil £12\nMushroom risotto — rice, parmesan £18\nMystery dish £10')
  assert.equal(items.length,3)
  assert.equal(items[0].name,'Burrata')
  assert.equal(items[0].ingredients,'tomato, basil')
  assert.equal(items[0].price,'12')
  assert.equal(items[2].ingredients,'')
})
test('ranks only reviewed menu items by imported ingredient overlap', () => {
  const items = parseMenu('Burrata — tomato, basil £12\nRisotto — mushroom, rice £18')
  const ranked = rankMenu(items,{ordered_ingredients:{mushroom:4}})
  assert.equal(ranked[0].name,'Risotto')
  assert.deepEqual(new Set(ranked.map(i=>i.id)),new Set(items.map(i=>i.id)))
  assert.deepEqual(ranked[0].matches,['mushroom'])
})
test('without overlap preserves menu order and does not mark a preference match', () => {
  const items = parseMenu('Burrata £12\nRisotto £18')
  const ranked = rankMenu(items,null)
  assert.deepEqual(ranked.map(i=>i.name),items.map(i=>i.name))
  assert.ok(ranked.every(i=>i.score===0&&i.matches.length===0))
})
test('matches ingredient words rather than unrelated substrings and excludes blank entries', () => {
  const ranked = rankMenu([{id:'1',name:'Nutmeg noodles',ingredients:''},{id:'2',name:'',ingredients:'nut'}],{ordered_ingredients:{nut:4}})
  assert.equal(ranked.length,1)
  assert.equal(ranked[0].score,0)
})
