const input = document.querySelector('#file-input');
const preview = document.querySelector('#preview');
const previewBox = document.querySelector('#preview-box');
const statusLine = document.querySelector('#status');
const allOutcomes = document.querySelector('#all-outcomes');
const rescanButton = document.querySelector('#rescan-button');
const cameraArea = document.querySelector('#camera-area');
const video = document.querySelector('#camera');

// Wide Model XAI Elements
const xaiSingleView = document.querySelector('#xai-single-view');
const xaiGridView = document.querySelector('#xai-grid-view');
const xaiGridContainer = document.querySelector('#xai-grid-container');
const xaiBaseImg = document.querySelector('#xai-transformed-base');
const xaiHeatImg = document.querySelector('#xai-heatmap-layer');
const opacitySlider = document.querySelector('#opacity-slider');
const opacityVal = document.querySelector('#opacity-val');
const xaiMethodTitle = document.querySelector('#xai-method-title');
const xaiMethodDesc = document.querySelector('#xai-method-desc');

// Local Model Patch Heatmap Elements
const patchBaseImg = document.querySelector('#patch-transformed-base');
const patchHeatImg = document.querySelector('#patch-heatmap-layer');
const patchOpacitySlider = document.querySelector('#patch-opacity-slider');
const patchOpacityVal = document.querySelector('#patch-opacity-val');
const patchHighlightBox = document.querySelector('#patch-highlight-box');
const patchMatrixGrid = document.querySelector('#patch-matrix-grid');
const patchInspectText = document.querySelector('#patch-inspect-text');
const topPatchesList = document.querySelector('#top-patches-list');

let selectedFile = null;
let stream = null;
let currentXAIData = null;
let currentMethod = 'gradcam';

// Top Tab Switching (Scan Image vs Model Data)
document.querySelectorAll('.tab').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('.tab, .panel').forEach(item => item.classList.remove('active'));
  button.classList.add('active');
  document.querySelector(`#${button.dataset.tab}`).classList.add('active');
}));

input.addEventListener('change', () => setFile(input.files[0]));
rescanButton.addEventListener('click', () => analyze());

function setFile(file) {
  if (!file) return;
  selectedFile = file;
  preview.src = URL.createObjectURL(file);
  previewBox.hidden = false;
  rescanButton.disabled = false;
  allOutcomes.hidden = true;
  setStatus(`Selected: ${file.name}. Ready to analyze.`);
  analyze();
}

async function analyze() {
  if (!selectedFile) return;
  allOutcomes.hidden = true;
  setStatus('Analyzing image across Wide Model, Local Model, and XAI Heatmaps (512x512)...');
  const form = new FormData();
  form.append('image', selectedFile, selectedFile.name);
  try {
    const response = await fetch('/api/analyze', { method: 'POST', body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Analysis failed.');
    renderResults(data);
    setStatus(`Analysis complete for: ${data.filename}`);
  } catch (error) {
    setStatus(error.message, true);
  }
}

function renderResults(data) {
  const pct = v => `${(Number(v) * 100).toFixed(2)}%`;
  const wide = data.wide_model || {};
  const local = data.local_model || {};
  const fusion = data.fusion || {};

  // 1. WIDE MODEL PREDICTION
  document.querySelector('#wide-verdict-badge').textContent = wide.prediction || '--';
  document.querySelector('#wide-val-verdict').textContent = wide.prediction || '--';
  document.querySelector('#wide-val-conf').textContent = pct(wide.confidence || 0);
  document.querySelector('#wide-val-real').textContent = pct(wide.real_probability || 0);
  document.querySelector('#wide-val-fake').textContent = pct(wide.fake_probability || 0);

  // XAI EXPLAINABILITY ON 512x512 TRANSFORMED FACE
  currentXAIData = data;
  if (data.transformed_image) {
    xaiBaseImg.src = data.transformed_image;
  } else {
    xaiBaseImg.src = preview.src;
  }

  if (currentMethod === 'compare') {
    showCompareGrid();
  } else {
    showSingleMethod(currentMethod);
  }

  // 2. LOCAL MODEL PREDICTION
  document.querySelector('#local-verdict-badge').textContent = local.prediction || '--';
  document.querySelector('#local-val-verdict').textContent = local.prediction || '--';
  document.querySelector('#local-val-conf').textContent = pct(local.confidence || 0);
  document.querySelector('#local-val-real').textContent = pct(local.real_probability || 0);
  document.querySelector('#local-val-fake').textContent = pct(local.fake_probability || 0);
  document.querySelector('#local-val-count').textContent = local.patch_count || 36;
  document.querySelector('#local-val-mean').textContent = Number(local.mean || 0).toFixed(5);
  document.querySelector('#local-val-std').textContent = Number(local.std_dev || 0).toFixed(5);
  document.querySelector('#local-val-var').textContent = Number(local.variance || 0).toFixed(5);

  // RENDER LOCAL MODEL PATCH HEATMAP & 6x6 MATRIX
  renderPatchHeatmap(local, data.transformed_image || preview.src);

  // 3. ENSEMBLE OUTCOME
  document.querySelector('#ensemble-verdict-badge').textContent = data.prediction || '--';
  document.querySelector('#ens-val-verdict').textContent = data.prediction || '--';
  document.querySelector('#ens-val-conf').textContent = pct(data.confidence || 0);
  document.querySelector('#ens-val-real').textContent = pct(data.real_probability || 0);
  document.querySelector('#ens-val-fake').textContent = pct(data.fake_probability || 0);
  document.querySelector('#ens-val-wideweight').textContent = Number(fusion.wide_weight || 1.0).toFixed(6);
  document.querySelector('#ens-val-localweight').textContent = Number(fusion.patch_weight || 1.0).toFixed(6);

  allOutcomes.hidden = false;
}

function showSingleMethod(methodKey) {
  if (!currentXAIData || !currentXAIData.heatmaps) return;

  const heatmaps = currentXAIData.raw_heatmaps || currentXAIData.heatmaps;
  const descriptions = currentXAIData.descriptions || {};

  if (!heatmaps[methodKey]) {
    methodKey = Object.keys(heatmaps)[0];
  }
  currentMethod = methodKey;

  document.querySelectorAll('.xai-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.method === methodKey);
  });

  xaiSingleView.hidden = false;
  xaiGridView.hidden = true;

  if (currentXAIData.raw_heatmaps && currentXAIData.raw_heatmaps[methodKey]) {
    xaiHeatImg.src = currentXAIData.raw_heatmaps[methodKey];
  } else {
    xaiHeatImg.src = currentXAIData.heatmaps[methodKey];
  }

  updateOpacity();

  const info = descriptions[methodKey] || {
    name: methodKey.toUpperCase(),
    concept: 'Model Activation Heatmap',
    description: 'Visual attribution map on 512x512 transformed image.',
  };

  xaiMethodTitle.textContent = `${info.name} — ${info.concept}`;
  xaiMethodDesc.textContent = info.description;
}

function showCompareGrid() {
  if (!currentXAIData || !currentXAIData.heatmaps) return;
  currentMethod = 'compare';

  document.querySelectorAll('.xai-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.method === 'compare');
  });

  xaiSingleView.hidden = true;
  xaiGridView.hidden = false;

  const overlays = currentXAIData.heatmaps;
  const descriptions = currentXAIData.descriptions || {};

  let html = '';
  for (const [key, overlaySrc] of Object.entries(overlays)) {
    const info = descriptions[key] || { name: key, concept: '' };
    html += `
      <div class="xai-grid-item">
        <h4>${info.name}</h4>
        <div class="grid-img-wrap">
          <img src="${overlaySrc}" alt="${info.name}" width="240" height="240">
        </div>
        <p class="grid-concept">${info.concept}</p>
      </div>
    `;
  }
  xaiGridContainer.innerHTML = html;
}

// Opacity Slider for Wide Model
if (opacitySlider) {
  opacitySlider.addEventListener('input', () => updateOpacity());
}

function updateOpacity() {
  if (!opacitySlider || !xaiHeatImg) return;
  const val = opacitySlider.value;
  opacityVal.textContent = `${val}%`;
  xaiHeatImg.style.opacity = (val / 100).toString();
}

// Opacity Slider for Local Patch Model
if (patchOpacitySlider) {
  patchOpacitySlider.addEventListener('input', () => {
    if (patchHeatImg) patchHeatImg.style.opacity = (patchOpacitySlider.value / 100).toString();
    if (patchOpacityVal) patchOpacityVal.textContent = patchOpacitySlider.value + '%';
  });
}

function renderPatchHeatmap(local, baseSrc) {
  if (!patchBaseImg || !patchHeatImg) return;
  patchBaseImg.src = baseSrc;
  patchHeatImg.src = local.patch_raw_heatmap || local.patch_heatmap_overlay || '';
  if (patchOpacitySlider) {
    patchHeatImg.style.opacity = (patchOpacitySlider.value / 100).toString();
  }

  const patches = local.patch_grid || [];
  if (patchMatrixGrid) {
    let gridHtml = '';
    patches.forEach(p => {
      const fakePct = (p.fake_probability * 100).toFixed(0);
      const isFake = p.fake_probability >= 0.5;
      // Exact color coordination: RED for AI/Fake, BLUE for Real
      const bg = isFake
        ? `rgba(220, 30, 30, ${Math.min(0.92, 0.3 + p.fake_probability * 0.6)})`
        : `rgba(20, 80, 220, ${Math.min(0.92, 0.3 + p.real_probability * 0.6)})`;
      gridHtml += `<div class="patch-cell" data-idx="${p.index}" style="background:${bg}; color:#ffffff;">
        <div class="patch-cell-id">P${p.index}</div>
        <div class="patch-cell-prob">${fakePct}%</div>
      </div>`;
    });
    patchMatrixGrid.innerHTML = gridHtml;

    // Attach hover listeners to cells for live inspection & bounding box highlight
    patchMatrixGrid.querySelectorAll('.patch-cell').forEach(cell => {
      const idx = parseInt(cell.dataset.idx, 10);
      const p = patches.find(item => item.index === idx);
      if (!p) return;

      cell.addEventListener('mouseenter', () => {
        patchInspectText.innerHTML = `<strong>Patch #${p.index} (Row ${p.row}, Col ${p.col}):</strong> Fake Prob: <strong>${(p.fake_probability * 100).toFixed(2)}%</strong> | Real Prob: <strong>${(p.real_probability * 100).toFixed(2)}%</strong> | Verdict: <strong>${p.verdict}</strong> [Position: X=${p.x}, Y=${p.y} (112x112)]`;
        if (patchHighlightBox) {
          patchHighlightBox.style.left = p.x + 'px';
          patchHighlightBox.style.top = p.y + 'px';
          patchHighlightBox.style.width = p.size + 'px';
          patchHighlightBox.style.height = p.size + 'px';
          patchHighlightBox.hidden = false;
        }
      });
      cell.addEventListener('mouseleave', () => {
        if (patchHighlightBox) patchHighlightBox.hidden = true;
      });
    });
  }

  // Render Top 5 suspicious patches
  if (topPatchesList && patches.length > 0) {
    const sorted = [...patches].sort((a, b) => b.fake_probability - a.fake_probability).slice(0, 5);
    topPatchesList.innerHTML = sorted.map(p =>
      `<li><strong>Patch #${p.index}</strong> (Row ${p.row}, Col ${p.col}): <strong>${(p.fake_probability * 100).toFixed(2)}% Fake</strong> (Position: X=${p.x}, Y=${p.y})</li>`
    ).join('');
  }
}

// XAI Method Navigation
document.querySelectorAll('.xai-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const method = btn.dataset.method;
    if (method === 'compare') {
      showCompareGrid();
    } else {
      showSingleMethod(method);
    }
  });
});

function setStatus(message, isError = false) {
  statusLine.textContent = message;
  statusLine.className = isError ? 'status error' : 'status';
}

// Camera controls
document.querySelector('#camera-button').addEventListener('click', async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
    video.srcObject = stream;
    cameraArea.hidden = false;
  } catch {
    setStatus('Camera access unavailable. Check browser permissions.', true);
  }
});
document.querySelector('#close-camera').addEventListener('click', stopCamera);
document.querySelector('#capture-button').addEventListener('click', () => {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  canvas.toBlob(blob => {
    setFile(new File([blob], 'camera-photo.jpg', { type: 'image/jpeg' }));
    stopCamera();
  }, 'image/jpeg', 0.92);
});
function stopCamera() {
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  cameraArea.hidden = true;
}

// Model Report
async function loadReport() {
  try {
    const data = await (await fetch('/api/model-report')).json();
    renderMarkdown(data.report);
    document.querySelector('#report-status').hidden = true;
    document.querySelector('#report').hidden = false;
  } catch {
    document.querySelector('#report-status').textContent = 'Model report could not be loaded.';
  }
}
function inline(text) {
  return text.replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
}
function renderMarkdown(markdown) {
  const report = document.querySelector('#report');
  const lines = markdown.split('\n');
  let html = '', inTable = false, inList = false;
  for (const line of lines) {
    if (/^\|/.test(line)) {
      const cells = line.split('|').slice(1, -1).map(cell => cell.trim());
      if (/^[-:| ]+$/.test(line.replace(/\|/g, ''))) continue;
      if (!inTable) {
        html += '<table class="simple-table"><thead><tr>' + cells.map(c => `<th>${inline(c)}</th>`).join('') + '</tr></thead><tbody>';
        inTable = true;
      } else {
        html += '<tr>' + cells.map(c => `<td>${inline(c)}</td>`).join('') + '</tr>';
      }
      continue;
    }
    if (inTable) {
      html += '</tbody></table>';
      inTable = false;
    }
    if (/^### /.test(line)) html += `<h3>${inline(line.slice(4))}</h3>`;
    else if (/^## /.test(line)) html += `<h2>${inline(line.slice(3))}</h2>`;
    else if (/^- /.test(line)) {
      if (!inList) { html += '<ul>'; inList = true; }
      html += `<li>${inline(line.slice(2))}</li>`;
    } else {
      if (inList) { html += '</ul>'; inList = false; }
      if (line.trim() && !/^---/.test(line) && !/^>/.test(line) && !/^\$/.test(line)) html += `<p>${inline(line)}</p>`;
    }
  }
  if (inTable) html += '</tbody></table>';
  if (inList) html += '</ul>';
  report.innerHTML = html;
}
loadReport();
