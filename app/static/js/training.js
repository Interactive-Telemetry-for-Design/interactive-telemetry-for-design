/********************************************************************
 * GLOBAL DATA
 ********************************************************************/
// We'll store label definitions: { id, name, color }
let labels = [];

// We'll store GT blocks: { id, labelId, labelName, color, start, end, element, leftHandle, rightHandle }
let blocks = [];

// AI blocks for adopt: { id, labelName, color, startPx, widthPx, element, adoptButton }
let aiBlocks = [];

// Ci blocks: { labelName, startPx, widthPx, avgConf, element }
let ciBlocks = [];

// For naming new labels
let labelCounter = 1;

// Active GT block
let activeBlockId = null;

// Selected AI block
let selectedAIBlockId = null;

/********************************************************************
 * DOM REFERENCES
 ********************************************************************/
const video = document.getElementById('video');
const groundTruthTimeline = document.getElementById('groundTruthTimeline');
const groundTruthPlayhead = groundTruthTimeline.querySelector('.playhead');

const aiTimeline = document.getElementById('aiTimeline');
const ciTimeline = document.getElementById('ciTimeline');

const labelList = document.getElementById('labelList');
const addLabelButton = document.getElementById('addLabelButton');
const sendAnnotationsButton = document.getElementById('sendAnnotationsButton');
const finishButton = document.getElementById('finishButton');
const epochInput = document.getElementById('epochInput');

const timelinesContainer = document.querySelector('.timelines-container');

/********************************************************************
 * CONFIG
 ********************************************************************/
// Hardcoded FPS
const FPS = 60000 / 1001;

// 1 px = 15 frames
const FRAMES_PER_PIXEL = 15;

// We'll ensure timeline width is at least some fallback
const MIN_TIMELINE_WIDTH = 1600;

// Default block width in px for new GT block
const DEFAULT_BLOCK_WIDTH_PX = 70;

/********************************************************************
 * INIT TIMELINES
 ********************************************************************/
video.addEventListener('loadedmetadata', () => {
  initTimelines();
});

function initTimelines() {
  const totalFrames = video.duration * FPS;
  const computedWidth = Math.floor(totalFrames / FRAMES_PER_PIXEL);
  const finalWidth = Math.max(MIN_TIMELINE_WIDTH, computedWidth);

  groundTruthTimeline.style.width = finalWidth + 'px';
  aiTimeline.style.width = finalWidth + 'px';
  ciTimeline.style.width = finalWidth + 'px';
}

/********************************************************************
 * HELPER FUNCTIONS
 ********************************************************************/
/** Local X coordinate in timeline, accounting for horizontal scroll. */
function getLocalX(e, timelineEl) {
  const rect = timelineEl.getBoundingClientRect();
  const scrollOffset = timelinesContainer.scrollLeft;
  return e.pageX - rect.left;
//   return e.pageX - rect.left + scrollOffset;
//   return e.pageX - e.currentTarget.offsetLeft + scrollOffset;
//   return e.clientX - rect.left + scrollOffset;
}

function getTimelineWidth() {
  return groundTruthTimeline.offsetWidth;
}

function timeToFrame(time) {
  return Math.floor(time * FPS);
}

function frameToPixels(frame) {
  const totalFrames = video.duration * FPS;
  const w = getTimelineWidth();
  if (!w || !totalFrames) return 0;
  return (frame / totalFrames) * w;
}

function pixelsToFrame(px) {
  const totalFrames = video.duration * FPS;
  const w = getTimelineWidth();
  if (!w || !totalFrames) return 0;
  const clamped = Math.min(Math.max(px, 0), w);
  return Math.floor((clamped / w) * totalFrames);
}

function pxToFrames(px) {
  const totalFrames = video.duration * FPS;
  const w = getTimelineWidth();
  if (!w || !totalFrames) return 0;
  return Math.floor((px / w) * totalFrames);
}

/********************************************************************
 * GROUND TRUTH BLOCKS
 ********************************************************************/
function dimOtherBlocksGT(blockId) {
  blocks.forEach(b => {
    if (b.id === blockId) b.element.classList.remove('dimmed');
    else b.element.classList.add('dimmed');
  });
}

function undimAllBlocksGT() {
  blocks.forEach(b => b.element.classList.remove('dimmed'));
}

function hideAllHandlesGT() {
  blocks.forEach(b => {
    b.leftHandle.classList.add('hidden');
    b.rightHandle.classList.add('hidden');
  });
}

/**
 * Overlap check: new block [start..end] overlaps existing block [b.start..b.end] if
 * (start< b.end) && (b.start< end)
 */
function canMoveBlockGT(blockId, newStart, newEnd) {
  if (newStart >= newEnd) return false;
  for (let b of blocks) {
    if (b.id === blockId) continue;
    if (newStart < b.end && b.start < newEnd) {
      return false;
    }
  }
  return true;
}

/** Removes any GT blocks that overlap [start..end]. */
function removeOverlappingGTBlocks(start, end) {
  blocks = blocks.filter(b => {
    // Overlap if (start < b.end && b.start < end)
    const doesOverlap = (start < b.end && b.start < end);
    if (doesOverlap) {
      // remove from DOM
      b.element.remove();
      return false;
    }
    return true;
  });
}

/********************************************************************
 * VIDEO => MOVE GT PLAYHEAD
 ********************************************************************/
video.addEventListener('timeupdate', () => {
  const currentFrame = timeToFrame(video.currentTime);
  const xPos = frameToPixels(currentFrame);
  groundTruthPlayhead.style.left = xPos + 'px';
});

/********************************************************************
 * GROUND TRUTH TIMELINE EVENTS
 ********************************************************************/
// Clicking an empty GT timeline => seek
groundTruthTimeline.addEventListener('click', (e) => {
  if (e.target.classList.contains('timeline-block') ||
      e.target.classList.contains('handle')) {
    return;
  }
  undimAllBlocksGT();
  hideAllHandlesGT();
  activeBlockId = null;

  const localX = getLocalX(e, groundTruthTimeline);
  const frame = pixelsToFrame(localX);
  video.currentTime = frame / FPS;
});

// For dragging the GT playhead
let isDraggingPlayhead = false;

groundTruthTimeline.addEventListener('mousedown', (e) => {
  if (e.target === groundTruthTimeline) {
    isDraggingPlayhead = true;
    movePlayhead(e);
  }
});
document.addEventListener('mousemove', (e) => {
  if (isDraggingPlayhead) {
    movePlayhead(e);
  }
});
document.addEventListener('mouseup', () => {
  isDraggingPlayhead = false;
});

function movePlayhead(e) {
  const localX = getLocalX(e, groundTruthTimeline);
  const frame = pixelsToFrame(localX);
  video.currentTime = frame / FPS;
}

/********************************************************************
 * SETUP GT BLOCK: REFAC TO SUPPORT BOTH "CREATE" AND "ADOPT"
 ********************************************************************/
function setupGTBlockResizing(blockObj) {
  // hooking up click => "edit" mode
  blockObj.element.addEventListener('click', (e) => {
    e.stopPropagation();
    activeBlockId = blockObj.id;
    dimOtherBlocksGT(blockObj.id);
    hideAllHandlesGT();
    blockObj.leftHandle.classList.remove('hidden');
    blockObj.rightHandle.classList.remove('hidden');
  });

  // local flags
  let isDraggingLeft = false;
  let isDraggingRight = false;

  blockObj.leftHandle.addEventListener('mousedown', (ev) => {
    ev.stopPropagation();
    isDraggingLeft = true;
  });
  blockObj.rightHandle.addEventListener('mousedown', (ev) => {
    ev.stopPropagation();
    isDraggingRight = true;
  });

  document.addEventListener('mousemove', (ev) => {
    if (!isDraggingLeft && !isDraggingRight) return;
    const localX = getLocalX(ev, groundTruthTimeline);
    const newFrame = pixelsToFrame(localX);
    const totalFrames = Math.floor(video.duration * FPS);

    if (isDraggingLeft) {
      let newStart = newFrame;
      if (newStart < 0) newStart = 0;
      if (newStart > blockObj.end) newStart = blockObj.end - 1;

      if (canMoveBlockGT(blockObj.id, newStart, blockObj.end)) {
        blockObj.start = newStart;
        blockObj.element.style.left = frameToPixels(blockObj.start) + 'px';
        blockObj.element.style.width = (frameToPixels(blockObj.end) - frameToPixels(blockObj.start)) + 'px';
        video.currentTime = blockObj.start / FPS;
      }
    }
    if (isDraggingRight) {
      let newEnd = newFrame;
      if (newEnd > totalFrames) newEnd = totalFrames;
      if (newEnd < blockObj.start) newEnd = blockObj.start + 1;

      if (canMoveBlockGT(blockObj.id, blockObj.start, newEnd)) {
        blockObj.end = newEnd;
        blockObj.element.style.width = (frameToPixels(blockObj.end) - frameToPixels(blockObj.start)) + 'px';
        video.currentTime = blockObj.end / FPS;
      }
    }
  });

  document.addEventListener('mouseup', () => {
    isDraggingLeft = false;
    isDraggingRight = false;
  });
}

/**
 * Creates a new GT block [start..end], removing overlaps if needed.
 */
function addGTBlock(labelId, labelName, color, start, end) {
  // remove any blocks that overlap
  removeOverlappingGTBlocks(start, end);

  // create the DOM
  const blockEl = document.createElement('div');
  blockEl.classList.add('timeline-block');
  blockEl.style.backgroundColor = color;
  blockEl.style.left = frameToPixels(start) + 'px';
  blockEl.style.width = (frameToPixels(end) - frameToPixels(start)) + 'px';

  const leftHandle = document.createElement('div');
  leftHandle.classList.add('handle', 'handle-left', 'hidden');
  const rightHandle = document.createElement('div');
  rightHandle.classList.add('handle', 'handle-right', 'hidden');
  blockEl.appendChild(leftHandle);
  blockEl.appendChild(rightHandle);

  groundTruthTimeline.appendChild(blockEl);

  const blockObj = {
    id: Date.now() + '-' + Math.random(),
    labelId,
    labelName,
    color,
    start,
    end,
    element: blockEl,
    leftHandle,
    rightHandle
  };

  // store in blocks array
  blocks.push(blockObj);

  // setup resizing logic
  setupGTBlockResizing(blockObj);

  // highlight it
  activeBlockId = blockObj.id;
  dimOtherBlocksGT(blockObj.id);
  leftHandle.classList.remove('hidden');
  rightHandle.classList.remove('hidden');
  return blockObj;
}

/********************************************************************
 * CREATE NEW GT BLOCK (user clicks label)
 ********************************************************************/
function createBlock(label) {
  const startFrame = timeToFrame(video.currentTime);
  const blockWidthFrames = pxToFrames(DEFAULT_BLOCK_WIDTH_PX);
  if (blockWidthFrames <= 0) {
    alert('Video not ready or timeline zero width. Cannot create block.');
    return;
  }

  const totalFrames = Math.floor(video.duration * FPS);
  let s = startFrame;
  let e = startFrame + blockWidthFrames;
  let shift = 0;

  while (!canMoveBlockGT(null, s, e)) {
    if (e > totalFrames) {
      alert('No space left to create a new block!');
      return;
    }
    shift++;
    if (shift > 20000) {
      alert('No space left to create a new block (shift>20000)!');
      return;
    }
    s++;
    e++;
  }

  addGTBlock(label.id, label.name, label.color, s, e);
}

/********************************************************************
 * LABELS
 ********************************************************************/
function getRandomColor() {
  const letters = '0123456789ABCDEF';
  let color = '#';
  for (let i = 0; i < 6; i++) {
    color += letters[Math.floor(Math.random() * 16)];
  }
  return color;
}

addLabelButton.addEventListener('click', () => {
  const labelName = `Label ${labelCounter++}`;
  const color = getRandomColor();
  const labelId = Date.now() + '-' + Math.random();
  const labelObj = { id: labelId, name: labelName, color };

  labels.push(labelObj);

  const button = document.createElement('button');
  button.classList.add('label-button');
  button.style.backgroundColor = color;
  button.innerText = labelName;

  // On click => create new GT block at current video time
  button.addEventListener('click', () => {
    createBlock(labelObj);
  });

  labelList.appendChild(button);
});

/********************************************************************
 * TRAIN & PREDICT => GET PREDICTIONS => CHUNK => RENDER
 ********************************************************************/
sendAnnotationsButton.addEventListener('click', () => {
  const epochCount = parseInt(epochInput.value, 10) || 5;

  // gather GT blocks
  const result = blocks.map(block => ({
    label: block.labelName,
    frame_start: block.start,
    frame_end: block.end
  }));

  const payload = {
    blocks: result,
    epochs: epochCount
  };

  fetch('/process_blocks', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  .then(r => r.json())
  .then(data => {
    console.log('Server response:', data);
    alert('Blocks + epochs sent to backend!');

    if (data.predictions) {
      chunkAndRenderPredictions(data.predictions);
    }
  })
  .catch(err => console.error('Error:', err));
});

/********************************************************************
 * FINISH
 ********************************************************************/
finishButton.addEventListener('click', () => {
  window.location.href = '/finished.html';
});

/********************************************************************
 * CHUNK + RENDER
 ********************************************************************/
function chunkAndRenderPredictions(predList) {
  // separate AI vs Ci
  predList.sort((a,b) => {
    if (a.source < b.source) return -1;
    if (a.source > b.source) return 1;
    return a.frame_number - b.frame_number;
  });

  const aiFrames = [];
  const ciFrames = [];

  for (const p of predList) {
    if (p.source === 'AI') aiFrames.push(p);
    else if (p.source === 'Ci') ciFrames.push(p);
  }

  const aiChunks = chunkConsecutive(aiFrames, 60);
  const ciChunks = chunkConsecutive(ciFrames, 60);

  renderAIChunks(aiChunks);
  renderCiChunks(ciChunks);
}

/**
 * chunkConsecutive => array of {start, end, label, avgConfidence}
 */
function chunkConsecutive(frames, minSize) {
  if (!frames.length) return [];
  frames.sort((a,b)=>a.frame_number - b.frame_number);

  let result = [];
  let startFrame = frames[0].frame_number;
  let currentLabel = frames[0].label;
  let sumConf = frames[0].confidence;
  let count = 1;

  for (let i=1; i<frames.length; i++){
    const f = frames[i];
    if (f.label===currentLabel && f.frame_number===frames[i-1].frame_number+1) {
      sumConf += f.confidence;
      count++;
    } else {
      // ended chunk
      const prevFrame = frames[i-1].frame_number;
      const length = prevFrame - startFrame +1;
      if (length >= minSize) {
        const avgConf = sumConf / count;
        result.push({
          start: startFrame,
          end: prevFrame,
          label: currentLabel,
          avgConfidence: avgConf
        });
      }
      // new chunk
      startFrame = f.frame_number;
      currentLabel = f.label;
      sumConf = f.confidence;
      count=1;
    }
  }
  // last chunk
  const lastFrame = frames[frames.length-1].frame_number;
  const length = lastFrame - startFrame +1;
  if(length>=minSize){
    const avgConf = sumConf / count;
    result.push({
      start: startFrame,
      end: lastFrame,
      label: currentLabel,
      avgConfidence: avgConf
    });
  }
  return result;
}

/********************************************************************
 * RENDER AI / CI
 ********************************************************************/
function renderAIChunks(chunks) {
  aiTimeline.innerHTML = '';
  aiBlocks = [];

  for (const c of chunks) {
    const {start, end, label, avgConfidence} = c;
    const color = getColorForLabel(label);

    const leftPx = frameToPixelsAI(start, aiTimeline);
    const rightPx = frameToPixelsAI(end, aiTimeline);
    const blockWidth = rightPx - leftPx;

    const blockEl = document.createElement('div');
    blockEl.classList.add('timeline-block');
    blockEl.style.backgroundColor = color;
    blockEl.style.left = leftPx + 'px';
    blockEl.style.width = blockWidth + 'px';

    const adoptBtn = document.createElement('button');
    adoptBtn.classList.add('adopt-button','hidden');
    adoptBtn.innerText='adopt';
    blockEl.appendChild(adoptBtn);

    const blockObj = {
      id: Date.now() + '-' + Math.random(),
      labelName: label,
      color,
      startPx: leftPx,
      widthPx: blockWidth,
      element: blockEl,
      adoptButton: adoptBtn
    };
    aiBlocks.push(blockObj);

    // click => select
    blockEl.addEventListener('click', (e) => {
      e.stopPropagation();
      selectAIBlock(blockObj);
    });
    adoptBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      adoptAIBlock(blockObj);
    });

    aiTimeline.appendChild(blockEl);
  }
}

function renderCiChunks(chunks) {
  ciTimeline.innerHTML = '';
  ciBlocks=[];

  for (const c of chunks) {
    const {start, end, label, avgConfidence} = c;
    const [r,g,b] = confidenceToRGB(avgConfidence);

    const leftPx = frameToPixelsAI(start, ciTimeline);
    const rightPx = frameToPixelsAI(end, ciTimeline);
    const blockWidth = rightPx - leftPx;

    const blockEl = document.createElement('div');
    blockEl.classList.add('timeline-block');
    blockEl.style.backgroundColor = `rgb(${r},${g},${b})`;
    blockEl.style.left = leftPx + 'px';
    blockEl.style.width = blockWidth + 'px';

    ciBlocks.push({
      labelName: label,
      startPx: leftPx,
      widthPx: blockWidth,
      avgConf: avgConfidence,
      element: blockEl
    });
    ciTimeline.appendChild(blockEl);
  }
}

function confidenceToRGB(conf) {
  // 0 => red, 1 => green
  const r = Math.floor((1-conf)*255);
  const g = Math.floor(conf*255);
  const b = 0;
  return [r,g,b];
}

/** Convert frames->px for AI or Ci timeline */
function frameToPixelsAI(frame, timelineEl){
  const totalFrames = video.duration * FPS;
  const w = timelineEl.offsetWidth;
  if (!w || !totalFrames) return 0;
  return (frame / totalFrames) * w;
}

function getColorForLabel(label) {
  const found = labels.find(l=>l.name===label);
  if (found) return found.color;
  return getRandomColor();
}

/********************************************************************
 * AI SELECT / DESELECT / ADOPT
 ********************************************************************/
function selectAIBlock(blockObj) {
  selectedAIBlockId = blockObj.id;
  aiBlocks.forEach(b => {
    if (b.id === blockObj.id) {
      b.element.classList.remove('dimmed');
      b.adoptButton.classList.remove('hidden');
    } else {
      b.element.classList.add('dimmed');
      b.adoptButton.classList.add('hidden');
    }
  });
}

// unselect if user clicks outside AI timeline
document.addEventListener('click', (e) => {
  if (!aiTimeline.contains(e.target)) {
    unselectAIBlock();
  }
});
function unselectAIBlock(){
  selectedAIBlockId = null;
  aiBlocks.forEach(b=>{
    b.element.classList.remove('dimmed');
    b.adoptButton.classList.add('hidden');
  });
}

/**
 * ADOPT => create GT block with correct start/end frames from pixel,
 * removing any overlapping blocks in GT.
 */
function adoptAIBlock(aiBlock){
  console.log('Adopting AI block => GT timeline', aiBlock);

  // find or create label
  let found = labels.find(l=>l.name===aiBlock.labelName);
  if (!found) {
    const labelId = Date.now()+'-'+Math.random();
    found = {id: labelId, name: aiBlock.labelName, color: aiBlock.color};
    labels.push(found);
  }

  // convert px => frames
  const sFrame = pixelsToFrame(aiBlock.startPx);
  const eFrame = pixelsToFrame(aiBlock.startPx + aiBlock.widthPx);

  // remove any GT blocks that overlap
  removeOverlappingGTBlocks(sFrame, eFrame);

  // now create new block in GT
  const newBlock = addGTBlock(found.id, found.name, found.color, sFrame, eFrame);

  unselectAIBlock();
  console.log('Adopted block => new GT block', newBlock);
}
