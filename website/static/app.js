'use strict';
const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const html = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = (value, digits=2) => value == null ? 'Not supplied' : `${(100*value).toFixed(digits)}%`;
const num = value => Number(value).toLocaleString('en-US');
const label = value => String(value).replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
const shortLabel = value => ({bacterial_spot:'Bacterial spot',cercospora_leaf_spot:'Cercospora',healthy:'Healthy',leaf_curl:'Leaf curl',nutrient_deficiency:'Nutrient def.',powdery_mildew:'Powdery mildew'}[value] || label(value));
let training, selectedFile, previewURL, prediction, busy = false;
let serverConnected = false, connectionCheck = null;
const fileInput = $('#leaf-file');
const dropZone = $('#drop-zone');
const SERVER_HELP = 'Cannot reach the prediction server. Run Start-PepperLens.cmd, keep its terminal open, then click Check connection. Your photo is not rejected for being new.';

function setConnection(connected, message=SERVER_HELP) {
  serverConnected = connected;
  $('#connection').innerHTML = connected ? '<span class="status-dot"></span>Trained model connected' : 'Server unreachable';
  $('#connection-message').textContent = connected ? '' : message;
  $('#connection-error').hidden = connected;
  $('#analyze').disabled = busy || !selectedFile || !training || !connected;
}

async function requestJSON(path, options={}, timeout=10000) {
  const controller = new AbortController();
  const timer = setTimeout(()=>controller.abort(),timeout);
  try {
    let response;
    try {
      response = await fetch(path,{...options,signal:controller.signal,cache:'no-store'});
    } catch(cause) {
      const error = new Error(cause.name==='AbortError'
        ? 'The server did not respond in time. Check its terminal, then retry. A slow response is not a disease prediction.'
        : SERVER_HELP);
      error.connectionFailure = cause.name!=='AbortError';
      throw error;
    }
    let payload;
    try { payload = await response.json(); }
    catch { throw new Error(`The server returned an incomplete or unreadable response (HTTP ${response.status}). Check its terminal and retry.`); }
    if (!response.ok) throw new Error(payload.error || `Analysis failed (HTTP ${response.status}).`);
    return payload;
  } finally { clearTimeout(timer); }
}

async function connectServer(refreshReport=false) {
  if (connectionCheck) return connectionCheck;
  connectionCheck = (async()=>{
    $('#retry-connection').disabled = true;
    try {
      const health = await requestJSON('/api/health',{},6000);
      if (health.application!=='PepperLens' || health.status!=='ready') throw new Error('A different service is using this address. Start PepperLens on an available port.');
      if (!training || !serverConnected || refreshReport) {
        training = await requestJSON('/api/training');
        renderTraining(training);
      }
      setConnection(true);
      return true;
    } catch(error) {
      setConnection(false,error.connectionFailure ? SERVER_HELP : `${error.message} Run Start-PepperLens.cmd and keep its terminal open.`);
      return false;
    } finally { $('#retry-connection').disabled = false; }
  })();
  try { return await connectionCheck; }
  finally { connectionCheck = null; }
}

function navigate() {
  const target = ['training','dsp'].includes(location.hash.slice(1)) ? location.hash.slice(1) : 'lab';
  $$('.view').forEach(view => view.hidden = view.id !== `view-${target}`);
  $$('.nav-link').forEach(link => link.classList.toggle('active', link.dataset.view === target));
  window.scrollTo({top:0,behavior:'instant'});
}
window.addEventListener('hashchange', navigate);
navigate();

function showError(message) {
  $('#upload-error').textContent = message;
  $('#upload-error').hidden = !message;
}

function resetPrediction() {
  prediction = null;
  $('#result-empty').hidden = false;
  $('#result-content').hidden = true;
  $('#prediction-workspace').hidden = true;
}

function chooseFile(file) {
  if (busy) return;
  showError('');
  if (!file) return;
  if (file.size > 15*1024*1024) { showError('Choose an image up to 15 MB.'); return; }
  if (!file.type.startsWith('image/') && !/\.(jpe?g|png|webp|bmp|tiff?|gif)$/i.test(file.name)) {
    showError('Choose a JPG, PNG or WebP image.'); return;
  }
  resetPrediction();
  selectedFile = file;
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = URL.createObjectURL(file);
  $('#leaf-preview').src = previewURL;
  $('#leaf-preview').hidden = false;
  $('#preview-overlay').hidden = false;
  $('#upload-prompt').hidden = true;
  $('#file-name').textContent = `${file.name} · ${(file.size/1024/1024).toFixed(2)} MB`;
  $('#clear-file').hidden = false;
  $('#analyze').disabled = !training || !serverConnected;
}

dropZone.addEventListener('click', () => { if (!busy) fileInput.click(); });
fileInput.addEventListener('change', () => chooseFile(fileInput.files[0]));
for (const type of ['dragenter','dragover']) dropZone.addEventListener(type, event => {
  event.preventDefault(); if (!busy) dropZone.classList.add('dragging');
});
for (const type of ['dragleave','drop']) dropZone.addEventListener(type, event => {
  event.preventDefault(); dropZone.classList.remove('dragging');
});
dropZone.addEventListener('drop', event => chooseFile(event.dataTransfer.files[0]));
window.addEventListener('dragover', event => event.preventDefault());
window.addEventListener('drop', event => event.preventDefault());
$('#clear-file').addEventListener('click', () => {
  if (busy) return;
  selectedFile = null; fileInput.value = '';
  if (previewURL) URL.revokeObjectURL(previewURL);
  $('#leaf-preview').hidden = true;
  $('#leaf-preview').removeAttribute('src');
  $('#preview-overlay').hidden = true;
  $('#upload-prompt').hidden = false;
  $('#file-name').textContent = 'One clear leaf. A simple background.';
  $('#clear-file').hidden = true; $('#analyze').disabled = true;
  resetPrediction(); showError('');
});

$('#analyze').addEventListener('click', async () => {
  if (!selectedFile || busy) return;
  busy = true; showError('');
  $('#analyze').disabled = true; $('#clear-file').disabled = true; dropZone.disabled = true;
  $('#analyze-label').textContent = 'Reading the signal';
  $('#analyze-label').classList.add('loading-dots');
  try {
    if (!await connectServer()) throw new Error(SERVER_HELP);
    const result = await requestJSON('/api/predict', {method:'POST', headers:{'Content-Type':'application/octet-stream'}, body:selectedFile},120000);
    prediction = result;
    renderPrediction(result);
  } catch(error) {
    if (error.connectionFailure) setConnection(false);
    showError(error.message);
  }
  finally {
    busy = false; $('#analyze').disabled = !selectedFile || !training || !serverConnected;
    $('#clear-file').disabled = false; dropZone.disabled = false;
    $('#analyze-label').textContent = 'Analyze leaf';
    $('#analyze-label').classList.remove('loading-dots');
  }
});

function renderPrediction(result) {
  const main = result.overall;
  const score = main.model_score;
  $('#result-empty').hidden = true; $('#result-content').hidden = false;
  $('#result-content').innerHTML = `
    <div class="prediction-header"><div><span class="prediction-category">${html(main.condition_type.toUpperCase())}</span><h3>${html(main.condition)}</h3><p>Selected classifier · ${html(main.selected_model)}</p></div>
    ${score == null ? '' : `<div class="confidence-ring" style="--score:${100*score}"><div><strong>${pct(score,1)}</strong><small>model score*</small></div></div>`}</div>
    <p class="prediction-explanation">This is the ${html(main.selected_model)} classification among the six trained conditions. The model score below describes this prediction; it is not measured accuracy for the photograph.</p>
    <div class="prediction-distribution">${result.distribution.map(row => `<div class="class-probability"><span>${html(shortLabel(row.label))}</span><div class="track"><div class="fill" style="width:${100*row.score}%"></div></div><strong>${pct(row.score,1)}</strong></div>`).join('')}</div>
    <p class="prediction-score-note">* Confidence = uncalibrated class score. A high score does not establish crop identity or rule out an unseen condition.</p>`;
  $('#prediction-workspace').hidden = false;
  $('#prediction-rows').innerHTML = result.per_model.map(row => {
    const isMain = row.model === main.selected_model;
    const recorded = training.comparison.find(model => model.model === row.model);
    return `<tr class="${isMain?'selected-row':''}"><td>${html(row.model)}${isMain?'<span class="main-badge">MAIN</span>':''}</td><td>${html(row.condition)}</td><td class="score-cell">${row.model_score == null ? 'Not supplied (SVM)' : pct(row.model_score)}</td><td class="score-cell">${pct(recorded?.test_accuracy)}</td></tr>`;
  }).join('');
  const agreeing = main.votes_for_selected_class;
  $('#agreement-note').textContent = `${agreeing} of ${main.total_models} classifiers selected the main condition. Voting is a comparison; ${main.selected_model} makes the main prediction because it was selected on validation.`;
  $('#processing-meta').textContent = `${result.original_size.join(' × ')} input · ${result.processing_seconds.toFixed(2)} s`;
  $('#coverage-note').textContent = `The candidate leaf mask covers ${pct(result.mask_coverage,1)} of the resized image. This is not disease severity or a percentage of diseased tissue. Check that the mask follows your leaf.`;
  $('#stage-grid').innerHTML = result.stages.map((stage,index) => {
    const group = ['sobel','dft'].includes(stage.id) ? 'diagnostic' : ['green','brown','union','closing','mask','segmented'].includes(stage.id) ? 'mask' : 'colour';
    return `<button type="button" class="stage-card" data-group="${group}" data-stage-index="${index}" aria-label="Enlarge ${html(stage.title)}"><img src="${stage.image}" alt="${html(stage.title)}" loading="lazy"><div><h3>${String(index+1).padStart(2,'0')} · ${html(stage.title)}</h3><p>${html(stage.caption)}</p></div></button>`;
  }).join('');
  $('#prediction-figures').innerHTML = result.figures.map((figure,index) => figureCard(figure.title, figure.image, 'Generated from your uploaded photograph.', `data-prediction-figure="${index}"`)).join('');
  $('#feature-tables').innerHTML = `
    <p>The classifier receives ${result.features.count} values: 512 masked HSV histogram bins, 13 Haralick texture values and seven log-transformed Hu moments. Histogram-bin sum: ${result.features.histogram.reduce((a,b)=>a+b,0).toFixed(6)}.</p>
    ${featureTable('Haralick texture',result.features.haralick,'Haralick')}${featureTable('Shape moments',result.features.hu,'Log Hu')}`;
  $$('[data-stage-group]').forEach(button => button.classList.toggle('active',button.dataset.stageGroup==='all'));
}

function featureTable(title, values, name) {
  return `<div><h3>${html(title)}</h3><table><thead><tr><th>Descriptor</th><th>Value</th></tr></thead><tbody>${values.map((value,index)=>`<tr><td>${name} ${index+1}</td><td>${value.toPrecision(6)}</td></tr>`).join('')}</tbody></table></div>`;
}

$$('[data-stage-group]').forEach(button => button.addEventListener('click', () => {
  $$('[data-stage-group]').forEach(item => item.classList.toggle('active', item===button));
  $$('.stage-card').forEach(card => card.hidden = button.dataset.stageGroup !== 'all' && card.dataset.group !== button.dataset.stageGroup);
}));

function figureCard(title, source, caption, attributes) {
  return `<article class="figure-card"><figure><button type="button" ${attributes} aria-label="Enlarge ${html(title)}"><img src="${html(source)}" alt="${html(title)}" loading="lazy"><span class="expand" aria-hidden="true">↗</span></button><figcaption><h3>${html(title)}</h3><p>${html(caption)}</p></figcaption></figure></article>`;
}

function openImage(title, source) {
  $('#dialog-title').textContent = title;
  $('#dialog-image').src = source;
  $('#dialog-image').alt = title;
  $('#dialog-download').href = source;
  $('#image-dialog').showModal();
}
$('#close-dialog').addEventListener('click', () => $('#image-dialog').close());
$('#image-dialog').addEventListener('click', event => { if (event.target === $('#image-dialog')) $('#image-dialog').close(); });
document.addEventListener('click', event => {
  const stage = event.target.closest('[data-stage-index]');
  const saved = event.target.closest('[data-saved-figure]');
  const fresh = event.target.closest('[data-prediction-figure]');
  if (stage && prediction) { const item=prediction.stages[Number(stage.dataset.stageIndex)]; openImage(item.title,item.image); }
  if (fresh && prediction) { const item=prediction.figures[Number(fresh.dataset.predictionFigure)]; openImage(item.title,item.image); }
  if (saved && training) { const item=training.plots[Number(saved.dataset.savedFigure)]; openImage(item.title,item.url); }
});

$('#download-prediction').addEventListener('click', () => {
  if (!prediction) return;
  const {stages,figures,...compact} = prediction;
  const blob = new Blob([JSON.stringify({image_name:selectedFile.name,...compact},null,2)],{type:'application/json'});
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a'); anchor.href=url; anchor.download='PepperLens_prediction.json'; anchor.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
});

function renderTraining(report) {
  const m = report.metrics;
  $$('[data-field="test-accuracy"]').forEach(element=>element.textContent=pct(m.test_accuracy_counting_rejections_as_errors));
  $('#main-model').textContent = m.selected_model.toUpperCase();
  $('#model-note').textContent = `${m.selected_model} · Selected using validation data · ${num(m.test_total)} test images`;
  $('#condition-chips').innerHTML = m.labels.map(condition=>`<span class="condition-chip">${html(label(condition))}</span>`).join('');
  $('#training-kpis').innerHTML = [
    ['Selected-model test accuracy',pct(m.test_accuracy_counting_rejections_as_errors),`${num(report.test_correct)} of ${num(m.test_total)} test images correct`],
    ['Test balanced accuracy',pct(m.test_balanced_accuracy),'Equal weight to recall across the six classes'],
    ['Retained unique images',num(m.unique_images),'After conflicting labels and duplicate cleanup'],
    ['Test macro F1',pct(m.classification_report['macro avg']['f1-score']),`${m.selected_model} · Six trained conditions`],
  ].map(([title,value,note])=>`<div class="kpi"><span>${html(title)}</span><strong>${html(value)}</strong><small>${html(note)}</small></div>`).join('');
  $('#benchmark-bars').innerHTML = report.comparison.map(row=>`<div class="benchmark-row"><div class="benchmark-name">${html(row.model)}${row.selected_for_main_prediction?'<small>VALIDATION WINNER · MAIN MODEL</small>':''}</div><div class="benchmark-tracks"><span style="width:${100*row.validation_balanced_accuracy}%"></span><span style="width:${100*row.test_balanced_accuracy}%"></span></div><div class="benchmark-values"><span>${pct(row.validation_balanced_accuracy,1)}</span><strong>${pct(row.test_balanced_accuracy,1)}</strong></div></div>`).join('');
  $('#comparison-rows').innerHTML = report.comparison.map(row=>`<tr class="${row.selected_for_main_prediction?'selected-row':''}"><td>${html(row.model)}${row.selected_for_main_prediction?'<span class="main-badge">MAIN</span>':''}</td>${['validation_balanced_accuracy','validation_macro_f1','test_accuracy','test_balanced_accuracy','test_macro_f1'].map(key=>`<td class="score-cell">${pct(row[key])}</td>`).join('')}</tr>`).join('');
  $('#report-model').innerHTML = report.comparison.map(row=>`<option ${row.model===m.selected_model?'selected':''}>${html(row.model)}</option>`).join('');
  renderClassMetrics(m.selected_model);
  const audit=m.data_audit;
  $('#audit-flow').innerHTML = [
    [audit.readable_files_before_cleanup,'Readable source images'],
    [audit.conflicting_files_excluded,'Conflicting-label copies excluded'],
    [audit.same_label_duplicate_copies_removed,'Same-label duplicate copies removed'],
    [audit.retained_unique_images,'Unique images retained'],
  ].map(([value,title])=>`<div class="audit-item"><strong>${num(value)}</strong><span>${html(title)}</span></div>`).join('');
  const maximum=Math.max(...Object.values(audit.counts_after_cleanup));
  $('#class-distribution').innerHTML=m.labels.map(condition=>`<div class="distribution-row"><div><span>${html(label(condition))}</span><strong>${num(audit.counts_after_cleanup[condition])}</strong></div><div class="distribution-track"><span style="width:${100*audit.counts_after_cleanup[condition]/maximum}%"></span></div><small>${num(audit.counts_before_cleanup[condition])} before cleanup</small></div>`).join('');
  const partitions=['train','validation','test'];
  const colours=['#597a45','#a2b987','#d0dcbc'];
  const sums=partitions.map(split=>m.labels.reduce((sum,condition)=>sum+m.split_counts[condition][split],0));
  $('#split-chart').innerHTML=`<div class="split-stack">${sums.map((value,index)=>`<span style="width:${100*value/m.unique_images}%;background:${colours[index]}" title="${partitions[index]}: ${num(value)}"></span>`).join('')}</div><div class="split-legend">${sums.map((value,index)=>`<span>${label(partitions[index])}<strong>${num(value)}</strong>${pct(value/m.unique_images,1)}</span>`).join('')}</div>`;
  $('#split-rows').innerHTML=m.labels.map(condition=>`<tr><td>${html(shortLabel(condition))}</td>${partitions.map(split=>`<td>${num(m.split_counts[condition][split])}</td>`).join('')}</tr>`).join('');
  $('#feature-failure-note').textContent=`Counts above are before feature filtering. ${num(m.feature_failures)} image(s) failed feature extraction; ${num(m.training_rows_including_synthetic_views)} usable training rows were fitted. ${num(audit.unreadable_files)} files failed decoding. No synthetic views were added in this run.`;
  $('#training-figures').innerHTML=report.plots.map((figure,index)=>figureCard(figure.title,figure.url,figure.caption,`data-saved-figure="${index}"`)).join('');
  const versions=report.environment;
  const stats=[
    ['Primary classifier',m.selected_model],['Selection metric','Validation balanced accuracy'],
    ['Selection score',pct(m.selection_score)],['Accepted test images / total',`${num(m.test_accepted)} / ${num(m.test_total)}`],
    ['Feature coverage on test',pct(m.test_coverage)],['Primary test errors',num(m.test_total-report.test_correct)],
    ['Image-level Wilson 95% interval',`${pct(m.image_level_wilson_95[0])} – ${pct(m.image_level_wilson_95[1])}`],
    ['Optional majority-vote accuracy (all test rows)',pct(m.ensemble_accuracy_counting_rejections_and_abstentions_as_errors)],
    ['Voting decisions / accepted test images',`${num(m.ensemble_decided)} / ${num(m.test_accepted)}`],
    ['Vote accuracy on decided images only',pct(m.ensemble_accuracy_on_decided)],
    ['Detected conflicting-image groups',num(audit.conflicting_image_groups)],
    ['Grouping basis','Image hashes; plant / parent IDs not reviewed'],
    ['Dataset source','Mendeley · Pepper Bell Leaf Disease v1'],
    ['Training environment',`Python ${versions.python} · sklearn ${versions.sklearn} · NumPy ${versions.numpy}`],
    ['Image-processing environment',`OpenCV ${versions.opencv} · mahotas ${versions.mahotas}`],
    ['Website runtime',`Python ${report.runtime.python} · sklearn ${report.runtime.sklearn} · NumPy ${report.runtime.numpy}`],
  ];
  $('#extra-statistics').innerHTML=stats.map(([title,value])=>`<div class="stat-line"><span>${html(title)}</span><strong>${html(value)}</strong></div>`).join('');
  $('#report-files').innerHTML=report.report_files.map(name=>`<a href="/reports/${encodeURIComponent(name)}" download>${html(name)} <span>↓</span></a>`).join('');
  $('#dsp-saved-figures').innerHTML=report.plots.map((figure,index)=>({figure,index})).filter(({figure})=>figure.filename.startsWith('dsp_')).map(({figure,index})=>figureCard(figure.title,figure.url,'Saved training example. Upload a leaf in the lab to generate the same analysis for your own photograph.',`data-saved-figure="${index}"`)).join('');
}

function renderClassMetrics(modelName) {
  const m=training.metrics;
  const evaluation=m.test_per_model[modelName];
  const matrix=evaluation.confusion_matrix;
  const maximum=Math.max(...matrix.flat());
  let content='<div class="matrix"><span></span>'+m.labels.map(condition=>`<span class="matrix-label top">${html(shortLabel(condition))}</span>`).join('');
  m.labels.forEach((condition,row)=>{
    content+=`<span class="matrix-label row-label">${html(shortLabel(condition))}</span>`;
    m.labels.forEach((predicted,col)=>{
      const value=matrix[row][col];
      const strength=value ? Math.max(.13,value/maximum) : 0;
      const bg=value ? `rgba(51,91,47,${(.12+.78*strength).toFixed(3)})` : '#f3f6ee';
      content+=`<span style="background:${bg};color:${strength>.55?'#fff':'#6b855a'}" title="Actual ${html(label(condition))}; predicted ${html(label(predicted))}: ${value}">${value}</span>`;
    });
  });
  $('#confusion-matrix').innerHTML=content+'</div>';
  $('#matrix-title').textContent=`Confusion matrix · ${modelName}`;
  $('#class-metric-rows').innerHTML=m.labels.map(condition=>{
    const row=evaluation.classification_report[condition];
    return `<tr><td>${html(shortLabel(condition))}</td><td>${pct(row.precision,1)}</td><td>${pct(row.recall,1)}</td><td>${pct(row['f1-score'],1)}</td><td>${num(row.support)}</td></tr>`;
  }).join('');
  $('#model-summary').innerHTML=[['Accuracy',evaluation.accuracy],['Balanced accuracy',evaluation.balanced_accuracy],['Macro F1',evaluation.classification_report['macro avg']['f1-score']]].map(([title,value])=>`<div><strong>${pct(value)}</strong><span>${title}</span></div>`).join('');
}
$('#report-model').addEventListener('change',event=>renderClassMetrics(event.target.value));

$('#dsp-pipeline').innerHTML=[
  ['01','Sample & represent','Decode the photograph, apply camera orientation and resample it to a 256 × 256 grid. Keep RGB and BGR channel conventions consistent.','SPATIAL RESAMPLING'],
  ['02','Smooth the signal','Apply a 3 × 3 Gaussian convolution with sigma 0.7 pixels. A spatial low-pass filter attenuates fine intensity changes.','2-D FIR FILTERING'],
  ['03','Isolate the leaf','Convert RGB to HSV, threshold green/brown candidates, combine with OR, close small gaps and fill the largest contour.','SEGMENTATION & MORPHOLOGY'],
  ['04','Measure colour','Calculate a masked 8 × 8 × 8 HSV histogram with the correct OpenCV hue range. Its 512 values describe the selected pixels.','HISTOGRAM / QUANTIZATION'],
  ['05','Measure texture & shape','Thirteen Haralick values describe grayscale co-occurrence. Seven log-transformed Hu moments describe the binary silhouette.','SPATIAL STATISTICS & MOMENTS'],
  ['06','Classify the descriptors','Seven trained algorithms classify the same 532 numerical features. Validation selects the main classifier; a separate grouped test set measures performance.','SUPERVISED CLASSIFICATION'],
].map(([number,title,body,concept])=>`<article class="pipeline-step"><span>${number}</span><h2>${title}</h2><p>${body}</p><small>${concept}</small></article>`).join('');

$('#retry-connection').addEventListener('click',()=>connectServer(true));
window.addEventListener('focus',()=>{ if (!busy) connectServer(); });
document.addEventListener('visibilitychange',()=>{ if (!document.hidden && !busy) connectServer(); });
setInterval(()=>{ if (!document.hidden && !busy) connectServer(); },15000);
connectServer();
