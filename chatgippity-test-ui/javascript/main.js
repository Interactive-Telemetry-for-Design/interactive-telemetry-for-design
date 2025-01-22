// --- CONSTANTS / ELEMENTS ---
const video = document.getElementById('video');
const timeline = document.getElementById('timeline');
const timelineScale = document.getElementById('timelineScale');
const playhead = document.getElementById('playhead');
const timeBlock = document.getElementById('timeBlock');

let isDraggingPlayhead = false;
let timelineRect;
let isSelectingBlock = false;

// Set the total timeline width here (keep in sync with CSS or dynamic).
// This is how many pixels wide the timeline is, representing full video length.
const TIMELINE_WIDTH = 640;

// Create tick marks (e.g., every 10% or 5% of the video length, or every second).
function createTimelineTicks() {
  const duration = video.duration || 60; // fallback if no metadata loaded
  const numberOfTicks = 10;
  for (let i = 0; i <= numberOfTicks; i++) {
    const tick = document.createElement('div');
    tick.className = 'tick';

    // Position is (i / numberOfTicks) * TIMELINE_WIDTH
    let leftPos = (i / numberOfTicks) * TIMELINE_WIDTH;
    tick.style.left = leftPos + 'px';

    const label = document.createElement('div');
    label.className = 'tick-label';

    // Convert fraction to time, e.g. seconds
    const time = (i / numberOfTicks) * duration;
    label.textContent = formatTime(time);

    // Place the label near the tick
    label.style.left = leftPos + 'px';

    timelineScale.appendChild(tick);
    timelineScale.appendChild(label);
  }
}

// Convert seconds to mm:ss for display
function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60).toString().padStart(2, '0');
  return `${mins}:${secs}`;
}

// --- PLAYHEAD DRAGGING ---
function onPlayheadMouseDown(e) {
  e.preventDefault();
  isDraggingPlayhead = true;
  document.addEventListener('mousemove', onPlayheadMouseMove);
  document.addEventListener('mouseup', onPlayheadMouseUp);
}

function onPlayheadMouseMove(e) {
  if (!isDraggingPlayhead) return;
  movePlayhead(e.clientX);
}

function onPlayheadMouseUp(e) {
  isDraggingPlayhead = false;
  document.removeEventListener('mousemove', onPlayheadMouseMove);
  document.removeEventListener('mouseup', onPlayheadMouseUp);
}

// Move the playhead to a given X (within the timeline boundaries)
function movePlayhead(clientX) {
  const x = clamp(clientX - timelineRect.left, 0, TIMELINE_WIDTH);
  playhead.style.left = x + 'px';

  // Update video current time
  const duration = video.duration;
  const ratio = x / TIMELINE_WIDTH;
  video.currentTime = ratio * duration;
}

// --- TIME BLOCK SELECTION ---
let selectionStart = 0;
let selectionEnd = 0;

function onTimelineMouseDown(e) {
  // If we clicked directly on the playhead handle, do not create a block
  if (e.target === playhead || e.target === playhead.firstChild) {
    return;
  }

  isSelectingBlock = true;
  selectionStart = clamp(e.clientX - timelineRect.left, 0, TIMELINE_WIDTH);
  selectionEnd = selectionStart;

  timeBlock.style.display = 'block';
  timeBlock.style.left = selectionStart + 'px';
  timeBlock.style.width = '0px';

  document.addEventListener('mousemove', onTimelineMouseMove);
  document.addEventListener('mouseup', onTimelineMouseUp);
}

function onTimelineMouseMove(e) {
  if (!isSelectingBlock) return;

  selectionEnd = clamp(e.clientX - timelineRect.left, 0, TIMELINE_WIDTH);
  let left = Math.min(selectionStart, selectionEnd);
  let width = Math.abs(selectionEnd - selectionStart);

  timeBlock.style.left = left + 'px';
  timeBlock.style.width = width + 'px';
}

function onTimelineMouseUp(e) {
  isSelectingBlock = false;
  document.removeEventListener('mousemove', onTimelineMouseMove);
  document.removeEventListener('mouseup', onTimelineMouseUp);

  // If width is 0 or very small, maybe hide
  if (Math.abs(selectionEnd - selectionStart) < 2) {
    timeBlock.style.display = 'none';
  }
}

// --- UPDATE PLAYHEAD WHEN VIDEO PLAYS ---
function updatePlayheadPosition() {
  const duration = video.duration || 0;
  if (duration <= 0) return;
  
  const ratio = video.currentTime / duration;
  const x = ratio * TIMELINE_WIDTH;
  playhead.style.left = x + 'px';
}

// Utility: clamp a number between min and max
function clamp(num, min, max) {
  return Math.max(min, Math.min(num, max));
}

// Wait for the video's metadata to load so we have the actual duration
video.addEventListener('loadedmetadata', () => {
  // Clear any existing ticks if needed
  timelineScale.innerHTML = '';
  createTimelineTicks();
});

function init() {
    // Generate some example ticks on the timeline scale
    createTimelineTicks();
  
    // Update video time when playhead is dragged
    playhead.addEventListener('mousedown', onPlayheadMouseDown);
  
    // For creating selection blocks (time-block)
    timeline.addEventListener('mousedown', onTimelineMouseDown);
  
    // Keep track of timeline's bounding rect for offset calculations
    timelineRect = timeline.getBoundingClientRect();
  
    // When the video time updates (either by scrubbing or playing), update the playhead position
    video.addEventListener('timeupdate', updatePlayheadPosition);
  
    // If video is playing, we also want the playhead to move
    // The 'timeupdate' event is enough to keep it in sync, but you can
    // also add a requestAnimationFrame loop if you want it to be smoother.
  }

// Initialize after window load
window.addEventListener('load', init);
