// Section 1 - State
const state = {
  currentFile: null,
  files: [],
  pdfDoc: null,
  currentPage: 1,
  totalPages: 0,
  zoom: 1.0,
  zoomLevels: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
  editor: null,
  markerRanges: [],
  activeMarkerIndex: -1,
  isDirty: false,
  isSaving: false,
};

const MARKER_START = '<!-- ⚠️ REVIEW NEEDED';
const MARKER_END = '<!-- END REVIEW SECTION -->';

let activeRenderTasks = [];
let renderSequence = 0;
let scanTimer = null;
let suppressEditorChange = false;
let pdfScrollTicking = false;

// Section 2 - Init
document.addEventListener('DOMContentLoaded', init);

async function init() {
  state.editor = CodeMirror.fromTextArea(
    document.getElementById('editor-textarea'),
    { mode: 'markdown', lineNumbers: true, lineWrapping: true }
  );
  state.editor.setSize('100%', '100%');

  attachEventListeners();
  await loadFileList();
}

async function loadFileList() {
  setHeaderMessage('');
  setPdfPlaceholder('Loading files...');

  try {
    const status = await fetchJson('/api/status');
    showDirectoryWarnings(status);

    state.files = await fetchJson('/api/files');
    populateFileSelect();

    if (!state.files.length) {
      setMarkerBadge(0);
      setPdfPlaceholder('No reviewable file pairs found');
      if (!getHeaderMessage()) {
        setHeaderMessage('No matching data/*.pdf and output/*.md pairs found');
      }
      return;
    }

    await loadFile(state.files[0].name);
  } catch (error) {
    console.error(error);
    setHeaderMessage('Unable to load file list');
    setPdfPlaceholder('Unable to load file list');
    setMarkerBadge(0);
  }
}

function attachEventListeners() {
  document.getElementById('file-select').addEventListener('change', (event) => {
    loadFile(event.target.value);
  });

  document.getElementById('save-btn').addEventListener('click', saveFile);
  document.getElementById('resolve-btn').addEventListener('click', resolveActiveMarker);

  document.getElementById('btn-prev').addEventListener('click', () => goToPage(state.currentPage - 1));
  document.getElementById('btn-next').addEventListener('click', () => goToPage(state.currentPage + 1));
  document.getElementById('page-input').addEventListener('change', function () {
    goToPage(parseInt(this.value, 10) || 1);
  });
  document.getElementById('btn-zoom-in').addEventListener('click', () => stepZoom(1));
  document.getElementById('btn-zoom-out').addEventListener('click', () => stepZoom(-1));
  document.getElementById('pdf-panel').addEventListener('scroll', updateCurrentPageFromScroll);

  state.editor.on('cursorActivity', handleCursorActivity);
  state.editor.on('change', () => {
    if (suppressEditorChange) return;
    state.isDirty = true;
    scheduleMarkerScan();
  });

  document.addEventListener('keydown', handleKeyboardShortcuts);
  window.addEventListener('beforeunload', warnIfDirty);

  attachDividerDrag();
  attachShortcutTooltip();
  updatePageControls();
}

function populateFileSelect() {
  const select = document.getElementById('file-select');
  select.innerHTML = '';
  select.disabled = state.files.length === 0;

  if (!state.files.length) {
    const option = document.createElement('option');
    option.textContent = 'No files';
    select.appendChild(option);
    return;
  }

  for (const file of state.files) {
    const option = document.createElement('option');
    option.value = file.name;
    option.dataset.name = file.name;
    option.textContent = fileOptionText(file.name, file.marker_count);
    select.appendChild(option);
  }
}

function showDirectoryWarnings(status) {
  const missing = [];
  if (!status.data_exists) missing.push(`data not found: ${status.data_dir}`);
  if (!status.output_exists) missing.push(`output not found: ${status.output_dir}`);
  setHeaderMessage(missing.join(' | '));
}

// Section 3 - File Loading
async function loadFile(name) {
  if (!name) return;
  if (state.isDirty && name !== state.currentFile) {
    const shouldSwitch = confirm('Unsaved changes - switch anyway?');
    if (!shouldSwitch) {
      syncSelectedFile();
      return;
    }
  }

  setHeaderMessage('');
  state.currentFile = name;
  syncSelectedFile();
  setPdfPlaceholder('Loading PDF...');
  cancelActiveRenderTasks();
  state.pdfDoc = null;
  state.totalPages = 0;
  state.currentPage = 1;
  updatePageControls();
  document.getElementById('resolve-btn').style.display = 'none';

  try {
    const textRes = await fetch(`/api/markdown/${encodeURIComponent(name)}`);
    if (!textRes.ok) throw new Error(`Markdown request failed: ${textRes.status}`);
    const text = await textRes.text();

    suppressEditorChange = true;
    state.editor.setValue(text);
    state.editor.clearHistory();
    suppressEditorChange = false;
    state.activeMarkerIndex = -1;
    scanMarkers();

    const pdfRes = await fetch(`/api/pdf/${encodeURIComponent(name)}`);
    if (!pdfRes.ok) throw new Error(`PDF request failed: ${pdfRes.status}`);
    const pdfBytes = await pdfRes.arrayBuffer();
    await loadPDF(pdfBytes);

    state.isDirty = false;
    setSaveButtonState('idle');
  } catch (error) {
    console.error(error);
    setHeaderMessage(`Unable to load ${name}`);
    setPdfPlaceholder('Unable to load PDF');
  } finally {
    suppressEditorChange = false;
  }
}

// Section 4 - Marker Scanning
function scanMarkers() {
  for (const marker of state.markerRanges) {
    marker.textMarker.clear();
  }
  state.markerRanges = [];

  const lines = state.editor.getValue().split('\n');
  let line = 0;

  while (line < lines.length) {
    if (!lines[line].includes(MARKER_START)) {
      line += 1;
      continue;
    }

    const startLine = line;
    let endLine = -1;
    let pageNum = 0;
    let commentClosed = false;

    for (let scanLine = startLine; scanLine < lines.length; scanLine += 1) {
      const pageMatch = lines[scanLine].match(/Page:\s*(\d+)/);
      if (pageMatch) pageNum = parseInt(pageMatch[1], 10);
      if (!commentClosed && lines[scanLine].includes('-->')) commentClosed = true;
      if (commentClosed && lines[scanLine].includes(MARKER_END)) {
        endLine = scanLine;
        break;
      }
    }

    if (endLine !== -1) {
      const textMarker = state.editor.markText(
        { line: startLine, ch: 0 },
        { line: endLine, ch: lines[endLine].length },
        { className: 'cm-marker-highlight', inclusiveLeft: true, inclusiveRight: true }
      );
      state.markerRanges.push({ from: startLine, to: endLine, pageNum, textMarker });
      line = endLine + 1;
    } else {
      line = startLine + 1;
    }
  }

  state.activeMarkerIndex = -1;
  document.getElementById('resolve-btn').style.display = 'none';
  updateMarkerBadge();
}

function scheduleMarkerScan() {
  clearTimeout(scanTimer);
  scanTimer = setTimeout(() => {
    scanMarkers();
    handleCursorActivity();
  }, 250);
}

function updateMarkerBadge() {
  setMarkerBadge(state.markerRanges.length);
}

function setMarkerBadge(count) {
  const badge = document.getElementById('marker-badge');
  badge.className = count > 0 ? 'badge-warning' : 'badge-success';
  badge.textContent = count > 0 ? `⚠️ ${count} markers remaining` : '✓ All markers resolved';
}

// Section 5 - Cursor -> PDF Sync
function handleCursorActivity() {
  const line = state.editor.getCursor().line;
  const idx = state.markerRanges.findIndex((marker) => line >= marker.from && line <= marker.to);

  if (idx !== -1 && idx !== state.activeMarkerIndex) {
    state.activeMarkerIndex = idx;
    positionResolveButton(state.markerRanges[idx]);
    const pageNum = state.markerRanges[idx].pageNum;
    if (pageNum > 0) goToPage(pageNum);
  }

  if (idx === -1 && state.activeMarkerIndex !== -1) {
    state.activeMarkerIndex = -1;
    document.getElementById('resolve-btn').style.display = 'none';
  }
}

function positionResolveButton(marker) {
  const btn = document.getElementById('resolve-btn');
  const coords = state.editor.charCoords({ line: marker.to, ch: 0 }, 'window');
  btn.style.top = `${coords.bottom + 4}px`;
  btn.style.left = `${coords.left + 8}px`;
  btn.style.display = 'block';
}

// Section 6 - PDF Rendering
async function loadPDF(pdfBytes) {
  state.pdfDoc = await pdfjsLib.getDocument({ data: new Uint8Array(pdfBytes) }).promise;
  state.totalPages = state.pdfDoc.numPages;
  state.currentPage = 1;
  setPdfPlaceholder('');
  await renderAllPages();
}

async function renderAllPages() {
  if (!state.pdfDoc) return;

  const sequence = ++renderSequence;
  cancelActiveRenderTasks();

  const pages = document.getElementById('pdf-pages');
  pages.innerHTML = '';

  for (let pageNum = 1; pageNum <= state.totalPages; pageNum += 1) {
    const pageShell = document.createElement('div');
    pageShell.className = 'pdf-page';
    pageShell.dataset.page = String(pageNum);

    const label = document.createElement('div');
    label.className = 'pdf-page-label';
    label.textContent = `Page ${pageNum}`;

    const canvas = document.createElement('canvas');
    canvas.className = 'pdf-page-canvas';
    canvas.dataset.page = String(pageNum);

    pageShell.append(label, canvas);
    pages.appendChild(pageShell);
  }

  updatePageControls();

  for (let pageNum = 1; pageNum <= state.totalPages; pageNum += 1) {
    if (sequence !== renderSequence) return;

    const page = await state.pdfDoc.getPage(pageNum);
    const viewport = page.getViewport({ scale: state.zoom });
    const canvas = pages.querySelector(`canvas[data-page="${pageNum}"]`);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = viewport.width;
    canvas.height = viewport.height;

    const renderTask = page.render({ canvasContext: ctx, viewport });
    activeRenderTasks.push(renderTask);

    try {
      await renderTask.promise;
    } catch (error) {
      if (error && error.name === 'RenderingCancelledException') return;
      throw error;
    } finally {
      activeRenderTasks = activeRenderTasks.filter((task) => task !== renderTask);
    }
  }

  if (sequence !== renderSequence) return;
  goToPage(state.currentPage, false);
}

function cancelActiveRenderTasks() {
  for (const task of activeRenderTasks) {
    task.cancel();
  }
  activeRenderTasks = [];
}

function goToPage(num, smooth = true) {
  if (!state.pdfDoc) return;

  const targetPage = Math.max(1, Math.min(num, state.totalPages));
  const page = document.querySelector(`.pdf-page[data-page="${targetPage}"]`);
  state.currentPage = targetPage;
  updatePageControls();

  if (!page) return;
  const panel = document.getElementById('pdf-panel');
  panel.scrollTo({
    top: Math.max(0, page.offsetTop - 12),
    behavior: smooth ? 'smooth' : 'auto',
  });
}

function stepZoom(direction) {
  const current = state.zoomLevels.indexOf(state.zoom);
  const next = Math.max(0, Math.min(current + direction, state.zoomLevels.length - 1));
  if (next === current) return;

  state.zoom = state.zoomLevels[next];
  const pageToKeep = state.currentPage;
  renderAllPages()
    .then(() => goToPage(pageToKeep, false))
    .catch((error) => {
      console.error(error);
      setHeaderMessage('Unable to render PDF zoom level');
    });
}

function updateCurrentPageFromScroll() {
  if (!state.pdfDoc || pdfScrollTicking) return;

  pdfScrollTicking = true;
  requestAnimationFrame(() => {
    pdfScrollTicking = false;

    const panel = document.getElementById('pdf-panel');
    const pageEls = Array.from(document.querySelectorAll('.pdf-page'));
    if (!pageEls.length) return;

    const panelTop = panel.getBoundingClientRect().top;
    let bestPage = state.currentPage;
    let bestDistance = Infinity;

    for (const pageEl of pageEls) {
      const rect = pageEl.getBoundingClientRect();
      const distance = Math.abs(rect.top - panelTop - 12);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestPage = parseInt(pageEl.dataset.page, 10);
      }
    }

    if (bestPage !== state.currentPage) {
      state.currentPage = bestPage;
      updatePageControls();
    }
  });
}

function updatePageControls() {
  const hasPdf = Boolean(state.pdfDoc);
  document.getElementById('page-input').value = state.currentPage;
  document.getElementById('page-total').textContent = `/ ${state.totalPages}`;
  document.getElementById('zoom-display').textContent = `${Math.round(state.zoom * 100)}%`;
  document.getElementById('btn-prev').disabled = !hasPdf || state.currentPage <= 1;
  document.getElementById('btn-next').disabled = !hasPdf || state.currentPage >= state.totalPages;
  document.getElementById('page-input').disabled = !hasPdf;
  document.getElementById('btn-zoom-in').disabled = !hasPdf || state.zoom === state.zoomLevels[state.zoomLevels.length - 1];
  document.getElementById('btn-zoom-out').disabled = !hasPdf || state.zoom === state.zoomLevels[0];
}

// Section 7 - Resolve Marker
function resolveActiveMarker() {
  if (state.activeMarkerIndex === -1) return;
  const marker = state.markerRanges[state.activeMarkerIndex];
  const lines = state.editor.getValue().split('\n');

  let blockStart = -1;
  let commentClose = -1;
  let blockEnd = -1;

  for (let i = marker.from; i <= Math.min(marker.to + 5, lines.length - 1); i += 1) {
    if (lines[i].includes(MARKER_START) && blockStart === -1) blockStart = i;
    if (blockStart !== -1 && commentClose === -1 && lines[i].trim() === '-->') commentClose = i;
    if (lines[i].includes(MARKER_END)) {
      blockEnd = i;
      break;
    }
  }

  if (blockStart === -1 || commentClose === -1 || blockEnd === -1) return;

  const tableLines = lines.slice(commentClose + 1, blockEnd);
  const newLines = [
    ...lines.slice(0, blockStart),
    ...tableLines,
    ...lines.slice(blockEnd + 1),
  ];

  suppressEditorChange = true;
  state.editor.setValue(newLines.join('\n'));
  suppressEditorChange = false;
  document.getElementById('resolve-btn').style.display = 'none';
  state.activeMarkerIndex = -1;
  scanMarkers();
  state.isDirty = true;
  saveFile();
}

// Section 8 - Save
async function saveFile() {
  if (!state.currentFile || state.isSaving) return;
  scanMarkers();
  handleCursorActivity();
  state.isSaving = true;
  setSaveButtonState('saving');

  try {
    const res = await fetch(`/api/markdown/${encodeURIComponent(state.currentFile)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain' },
      body: state.editor.getValue(),
    });
    if (!res.ok) throw new Error(`Save failed: ${res.status}`);
    const data = await res.json();
    state.isDirty = false;
    setSaveButtonState('saved');
    setTimeout(() => setSaveButtonState('idle'), 2000);
    updateFileBadge(state.currentFile, data.marker_count);
  } catch (error) {
    console.error(error);
    setSaveButtonState('error');
    setTimeout(() => setSaveButtonState('idle'), 3000);
  } finally {
    state.isSaving = false;
  }
}

function setSaveButtonState(mode) {
  const button = document.getElementById('save-btn');
  button.classList.remove('is-saved', 'is-error');
  button.disabled = mode === 'saving';

  if (mode === 'idle') button.textContent = '💾 Save';
  if (mode === 'saving') button.textContent = 'Saving...';
  if (mode === 'saved') {
    button.textContent = '✓ Saved';
    button.classList.add('is-saved');
  }
  if (mode === 'error') {
    button.textContent = '✗ Error';
    button.classList.add('is-error');
  }
}

function updateFileBadge(name, markerCount) {
  const file = state.files.find((item) => item.name === name);
  if (file) file.marker_count = markerCount;

  const option = document.querySelector(`#file-select option[data-name="${cssEscape(name)}"]`);
  if (option) option.textContent = fileOptionText(name, markerCount);
}

function fileOptionText(name, markerCount) {
  return markerCount > 0 ? `${name}  ⚠️ ${markerCount}` : `${name}  ✓`;
}

// Section 9 - Keyboard Shortcuts
function handleKeyboardShortcuts(e) {
  const ctrl = e.ctrlKey || e.metaKey;

  if (ctrl && e.key.toLowerCase() === 's') {
    e.preventDefault();
    saveFile();
  }
  if (e.altKey && e.key.toLowerCase() === 'n') {
    e.preventDefault();
    jumpToMarker(+1);
  }
  if (e.altKey && e.key.toLowerCase() === 'p') {
    e.preventDefault();
    jumpToMarker(-1);
  }
  if (e.altKey && e.key.toLowerCase() === 'r') {
    e.preventDefault();
    resolveActiveMarker();
  }
  if (e.altKey && e.key === '1') {
    e.preventDefault();
    document.getElementById('pdf-panel').focus();
  }
  if (e.altKey && e.key === '2') {
    e.preventDefault();
    state.editor.focus();
  }
}

function jumpToMarker(dir) {
  if (!state.markerRanges.length) return;
  let next = state.activeMarkerIndex + dir;
  if (state.activeMarkerIndex === -1) next = dir > 0 ? 0 : state.markerRanges.length - 1;
  next = Math.max(0, Math.min(next, state.markerRanges.length - 1));
  const marker = state.markerRanges[next];
  state.editor.setCursor({ line: marker.from + 1, ch: 0 });
  state.editor.scrollIntoView({ line: marker.from, ch: 0 }, 150);
}

// Section 10 - Resizable Divider
function attachDividerDrag() {
  const divider = document.getElementById('divider');
  const pdfPanel = document.getElementById('pdf-panel');
  const mdPanel = document.getElementById('md-panel');
  const container = document.getElementById('main-container');
  let dragging = false;

  divider.addEventListener('mousedown', () => {
    dragging = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  });

  document.addEventListener('mousemove', (event) => {
    if (!dragging) return;
    const rect = container.getBoundingClientRect();
    const ratio = (event.clientX - rect.left) / rect.width;
    const pct = Math.min(Math.max(ratio * 100, 20), 80);
    pdfPanel.style.width = `${pct}%`;
    pdfPanel.style.flexBasis = `${pct}%`;
    mdPanel.style.width = `${100 - pct - 0.5}%`;
  });

  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    state.editor.refresh();
  });
}

// Section 11 - Keyboard Shortcut Help Tooltip
function attachShortcutTooltip() {
  const button = document.getElementById('help-btn');
  const tooltip = document.getElementById('shortcut-tooltip');

  button.addEventListener('click', (event) => {
    event.stopPropagation();
    tooltip.hidden = !tooltip.hidden;
  });

  tooltip.addEventListener('click', (event) => {
    event.stopPropagation();
  });

  document.addEventListener('click', () => {
    tooltip.hidden = true;
  });
}

// Shared helpers
async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} failed: ${res.status}`);
  return res.json();
}

function setHeaderMessage(message) {
  const el = document.getElementById('header-message');
  const badge = document.getElementById('marker-badge');
  el.textContent = message;
  el.title = message;
  el.hidden = !message;
  badge.hidden = Boolean(message);
}

function getHeaderMessage() {
  return document.getElementById('header-message').textContent;
}

function setPdfPlaceholder(message) {
  const placeholder = document.getElementById('pdf-placeholder');
  const pages = document.getElementById('pdf-pages');
  placeholder.textContent = message;
  placeholder.hidden = !message;
  pages.hidden = Boolean(message);
  if (message) pages.innerHTML = '';
}

function syncSelectedFile() {
  const select = document.getElementById('file-select');
  if (select.value !== state.currentFile) select.value = state.currentFile || '';
}

function warnIfDirty(event) {
  if (!state.isDirty) return;
  event.preventDefault();
  event.returnValue = '';
}

function cssEscape(value) {
  if (window.CSS && CSS.escape) return CSS.escape(value);
  return value.replace(/"/g, '\\"');
}
