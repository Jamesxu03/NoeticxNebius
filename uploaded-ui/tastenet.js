let modelStatus = 'Loading live TasteNet predictions…';
let predictingCatalogue = false;
function predictionInput(dish) {
  return {name: dish.name, description: dish.description || '', cuisine: dish.cuisine, ingredients: dish.ingredients || []};
}
async function modelRequest(path, body) {
  const response = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'Please check the dish details.');
  return result;
}
function modelPanel() {
  return `<section class="model-panel"><p class="eyebrow">TASTELAYER · TASTENET</p><p id="model-status" role="status">${esc(modelStatus)}</p><button class="secondary" onclick="refreshModelCatalogue()">Refresh model predictions ↻</button><details><summary>Predict a dish’s taste</summary><form onsubmit="predictDish(event)"><label for="predict-name">Dish name</label><input id="predict-name" required maxlength="200" placeholder="e.g. Chilli paneer"><label for="predict-cuisine">Cuisine</label><select id="predict-cuisine"><option value="indian">Indian</option><option value="italian">Italian</option></select><label for="predict-ingredients">Ingredients, separated by commas</label><input id="predict-ingredients" placeholder="paneer, chilli, tomato"><button class="primary" id="predict-submit">Predict taste →</button></form><div id="predict-result" aria-live="polite"></div></details></section>`;
}
async function refreshModelCatalogue() {
  if (predictingCatalogue || !data) return;
  predictingCatalogue = true;
  modelStatus = 'Loading live TasteNet predictions…'; updateModelStatus();
  try {
    const predictions = await modelRequest('/api/taste/batch', {dishes:data.dishes.map(predictionInput)});
    if (predictions.length !== data.dishes.length) throw new Error('Incomplete model response.');
    data.dishes = data.dishes.map((dish, i) => ({...dish, ...predictions[i], taste_source:'tastenet'}));
    modelStatus = `Live ${predictions[0].taste_model_version} · ${predictions.length} dishes predicted locally. Taste matches and survey now use these scores.`;
  } catch (error) {
    modelStatus = data.dishes.some(d=>d.taste_source==='tastenet')
      ? 'Refresh failed. Keeping the last successful TasteNet predictions. '+error.message
      : 'Using uploaded catalogue scores; live predictions are unavailable. '+error.message;
  } finally { predictingCatalogue=false; updateModelStatus(); if(screen==='home' && !document.querySelector('.model-panel details[open]')) render(); }
}
async function predictDish(event) {
  event.preventDefault();
  const button=document.getElementById('predict-submit'), output=document.getElementById('predict-result');
  const dish={name:document.getElementById('predict-name').value.trim(), cuisine:document.getElementById('predict-cuisine').value, ingredients:document.getElementById('predict-ingredients').value.split(',').map(s=>s.trim()).filter(Boolean)};
  button.disabled=true; output.textContent='Predicting locally…';
  try {
    const result=await modelRequest('/api/taste',dish);
    output.innerHTML=`<p><b>${esc(dish.name)}</b> · ${esc(result.taste_model_version)}</p>${keys.map(k=>`<div class="prediction-row"><label>${esc(k)}</label><meter min="0" max="10" value="${Number(result.taste[k])}"></meter><span>${Number(result.taste[k])}/10 · ${Math.round(result.taste_confidence[k]*100)}% confidence</span></div>`).join('')}<p class="notice">Taste predictions describe flavour. They do not verify allergens or dietary suitability.</p>`;
  } catch(error) {output.textContent=error.message;} finally {button.disabled=false;}
}

function updateModelStatus() { const el=document.getElementById('model-status'); if(el)el.textContent=modelStatus; }
