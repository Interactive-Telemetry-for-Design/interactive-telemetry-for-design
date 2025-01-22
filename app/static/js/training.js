// We'll store label definitions here: { id, name, color }
let labels = [];

// We'll store the blocks here:
// Each block: { id, labelId, labelName, color, start, end, element, leftHandle, rightHandle }
let blocks = [];

// Current label index for naming new labels
let labelCounter = 1;

// For controlling which block is in 'edit' mode
let activeBlockId = null;

// Video element
const video = document.getElementById('video');

// Timeline element
const timelineContainer = document.getElementById('timelineContainer');
const playhead = document.getElementById('playhead');

// Some helpful constants
// If you'd like to consider frames, define a frames-per-second for your video:
const FPS = 30; // Example assumption

/********************************************************************
 * Helper Functions
 ********************************************************************/

// Convert current video time (in seconds) to 'frame' integer
function timeToFrame(time) {
    return Math.floor(time * FPS);
}

// Convert frame to timeline position in pixels
function frameToPixels(frame) {
    const totalFrames = video.duration * FPS;
    return (frame / totalFrames) * timelineContainer.clientWidth;
}

// Convert X position in timeline to a frame
function pixelsToFrame(x) {
    const totalFrames = video.duration * FPS;
    const clampedX = Math.min(Math.max(x, 0), timelineContainer.clientWidth);
    return Math.floor((clampedX / timelineContainer.clientWidth) * totalFrames);
}

// Dim all blocks except the one with id=blockId
function dimOtherBlocks(blockId) {
    blocks.forEach(b => {
    if (b.id === blockId) {
        b.element.classList.remove('dimmed');
    } else {
        b.element.classList.add('dimmed');
    }
    });
}

// Undim all blocks
function undimAllBlocks() {
    blocks.forEach(b => {
    b.element.classList.remove('dimmed');
    });
}

// Hide all handles
function hideAllHandles() {
    blocks.forEach(b => {
    b.leftHandle.classList.add('hidden');
    b.rightHandle.classList.add('hidden');
    });
}

// Check for overlap with existing blocks
function canMoveBlock(blockId, newStart, newEnd) {
    // Ensure newStart < newEnd, otherwise invalid
    if (newStart >= newEnd) return false;

    // Compare with other blocks
    for (let b of blocks) {
    if (b.id === blockId) continue;
    // If intervals overlap: (startA < endB) and (startB < endA)
    if (newStart < b.end && b.start < newEnd) {
        return false;
    }
    }
    return true;
}

/********************************************************************
 * Event Handlers
 ********************************************************************/

// Update playhead as video plays
video.addEventListener('timeupdate', () => {
    const currentFrame = timeToFrame(video.currentTime);
    const xPos = frameToPixels(currentFrame);
    playhead.style.left = xPos + 'px';
});

// When user clicks on the timeline (outside a block), go to that frame
timelineContainer.addEventListener('click', (e) => {
    // If the click is on a block or handle, we skip
    if (e.target.classList.contains('timeline-block') ||
        e.target.classList.contains('handle')) {
    return;
    }

    // 1. Undim everything
    undimAllBlocks();
    // 2. Hide handles
    hideAllHandles();
    // 3. Clear active block
    activeBlockId = null;

    // Also seek the video to the clicked position
    const rect = timelineContainer.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const frame = pixelsToFrame(clickX);
    video.currentTime = frame / FPS;
});

// Dragging the playhead to seek:
let isDraggingPlayhead = false;

timelineContainer.addEventListener('mousedown', (e) => {
    // Only start dragging if we didn't click on a block
    if (e.target === timelineContainer) {
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
    const rect = timelineContainer.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const frame = pixelsToFrame(x);
    const newTime = frame / FPS;
    video.currentTime = newTime;
}

// Handle block clicks (switch to 'edit' mode)
function onBlockClick(block) {
    return function(e) {
    e.stopPropagation(); // prevent timeline click
    // Make this block active
    activeBlockId = block.id;
    dimOtherBlocks(block.id);
    hideAllHandles();
    block.leftHandle.classList.remove('hidden');
    block.rightHandle.classList.remove('hidden');
    }
}

/********************************************************************
 * Creating/Editing Blocks
 ********************************************************************/

// When user clicks on a label, add a new block at the current video time
function createBlock(label) {
    const startFrame = timeToFrame(video.currentTime);
    const defaultBlockWidthFrames = 30; // e.g. ~1 second, if FPS=30
    const endFrame = startFrame + defaultBlockWidthFrames;

    // Check for overlap, if overlap, shift it a bit
    let actualStart = startFrame;
    let actualEnd = endFrame;

    // Quick fallback if default overlaps, just search for an open gap:
    // (Here we do a naive approach - you can do more robust logic.)
    let canPlace = canMoveBlock(null, actualStart, actualEnd);
    let shift = 0;
    while (!canPlace && shift < 3000) {
    shift += 1;
    actualStart = startFrame + shift;
    actualEnd = endFrame + shift;
    canPlace = canMoveBlock(null, actualStart, actualEnd);
    }
    if (!canPlace) {
    // If we really can't place, just return (or handle differently)
    alert('No space to create a new block!');
    return;
    }

    // Create DOM elements
    const blockEl = document.createElement('div');
    blockEl.classList.add('timeline-block');
    blockEl.style.backgroundColor = label.color;
    blockEl.style.left = frameToPixels(actualStart) + 'px';
    blockEl.style.width = (frameToPixels(actualEnd) - frameToPixels(actualStart)) + 'px';

    // Create handles
    const leftHandle = document.createElement('div');
    leftHandle.classList.add('handle', 'handle-left', 'hidden');
    const rightHandle = document.createElement('div');
    rightHandle.classList.add('handle', 'handle-right', 'hidden');

    blockEl.appendChild(leftHandle);
    blockEl.appendChild(rightHandle);
    timelineContainer.appendChild(blockEl);

    // Construct block object
    const blockObj = {
    id: Date.now() + '-' + Math.random(), // unique ID
    labelId: label.id,
    labelName: label.name,
    color: label.color,
    start: actualStart,
    end: actualEnd,
    element: blockEl,
    leftHandle: leftHandle,
    rightHandle: rightHandle
    };

    // Attach event for block click
    blockEl.addEventListener('click', onBlockClick(blockObj));

    // Add to global blocks
    blocks.push(blockObj);

    // Switch to editing mode for newly created block
    activeBlockId = blockObj.id;
    dimOtherBlocks(blockObj.id);
    blockObj.leftHandle.classList.remove('hidden');
    blockObj.rightHandle.classList.remove('hidden');

    // Handle dragging of left/right handles
    let isDraggingLeft = false;
    let isDraggingRight = false;

    leftHandle.addEventListener('mousedown', (e) => {
    e.stopPropagation();
    isDraggingLeft = true;
    });
    rightHandle.addEventListener('mousedown', (e) => {
    e.stopPropagation();
    isDraggingRight = true;
    });

    document.addEventListener('mousemove', (e) => {
    if (!isDraggingLeft && !isDraggingRight) return;
    const rect = timelineContainer.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const frame = pixelsToFrame(x);

    if (isDraggingLeft) {
        // Proposed new start
        let newStart = frame;
        // Keep it bounded
        if (newStart < 0) newStart = 0;
        if (newStart > blockObj.end) newStart = blockObj.end - 1;

        // Check overlap
        if (canMoveBlock(blockObj.id, newStart, blockObj.end)) {
        blockObj.start = newStart;
        // Update DOM
        blockEl.style.left = frameToPixels(blockObj.start) + 'px';
        blockEl.style.width = (frameToPixels(blockObj.end) - frameToPixels(blockObj.start)) + 'px';
        // Also update video time
        video.currentTime = blockObj.start / FPS;
        }
    }

    if (isDraggingRight) {
        // Proposed new end
        let newEnd = frame;
        // Keep it bounded
        const totalFrames = video.duration * FPS;
        if (newEnd > totalFrames) newEnd = totalFrames;
        if (newEnd < blockObj.start) newEnd = blockObj.start + 1;

        // Check overlap
        if (canMoveBlock(blockObj.id, blockObj.start, newEnd)) {
        blockObj.end = newEnd;
        // Update DOM
        blockEl.style.width = (frameToPixels(blockObj.end) - frameToPixels(blockObj.start)) + 'px';
        // Also update video time
        video.currentTime = blockObj.end / FPS;
        }
    }
    });

    document.addEventListener('mouseup', () => {
    isDraggingLeft = false;
    isDraggingRight = false;
    });
}

/********************************************************************
 * Label Management
 ********************************************************************/

const labelList = document.getElementById('labelList');
const addLabelButton = document.getElementById('addLabelButton');
const removeLabelButton = document.getElementById('removeLabelButton');

// Generate random color
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
    const labelObj = { id: labelId, name: labelName, color: color };

    labels.push(labelObj);

    const button = document.createElement('button');
    button.classList.add('label-button');
    button.style.backgroundColor = color;
    button.innerText = labelName;
    button.addEventListener('click', () => {
    createBlock(labelObj);
    });

    labelList.appendChild(button);
});

removeLabelButton.addEventListener('click', () => {
    if (labels.length === 0) return;
    // Remove the last label from the array
    const removedLabel = labels.pop();
    // Remove the last button from DOM
    labelList.removeChild(labelList.lastChild);

    // Also remove blocks with that label
    blocks = blocks.filter(b => {
    if (b.labelId === removedLabel.id) {
        // Remove from DOM
        b.element.remove();
        return false;
    }
    return true;
    });
});

/********************************************************************
 * Sending data to backend
 ********************************************************************/

document.getElementById('sendAnnotationsButton').addEventListener('click', () => {
    // Build the data in the desired format:
    // {label: string, frame_start: int, frame_end: int}[]
    const result = blocks.map(b => {
    return {
        label: b.labelName,
        frame_start: b.start,
        frame_end: b.end
    };
    });

    // Send to server via fetch
    fetch('/run', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify(result)
    })
    .then(response => response.json())
    .then(data => {
    console.log('Server response:', data);
    alert('Blocks sent to backend!');
    })
    .catch(err => {
    console.error('Error:', err);
    });
});