const video = document.getElementById('video');
const timeline = document.getElementById('timeline');
const playhead = document.getElementById('playhead');
const seeker = document.getElementById('seeker');
const labelsList = document.getElementById('labels-list');
const addLabelBtn = document.getElementById('add-label-btn');
const runBtn = document.getElementById('run-btn');

let activeBlock = null;
let blocks = [];
let isDragging = false;
let dragStartX = 0;
let dragOffset = 0;
let labelColors = [
    '#FF5733', '#33FF57', '#3357FF', '#FF33F5',
    '#33FFF5', '#F5FF33', '#FF3333', '#33FF33'
];
let nextColorIndex = 0;

// Convert timeline position to video time and vice versa
function posToTime(pos) {
    const rect = timeline.getBoundingClientRect();
    const percentage = pos / rect.width;
    return percentage * video.duration;
}

function timeToPos(time) {
    const rect = timeline.getBoundingClientRect();
    const percentage = time / video.duration;
    return percentage * rect.width;
}

// Update playhead position during video playback
video.addEventListener('timeupdate', () => {
    const pos = timeToPos(video.currentTime);
    playhead.style.left = `${pos}px`;
    seeker.style.left = `${pos}px`;
});

// Seeker drag functionality
seeker.addEventListener('mousedown', (e) => {
    isDragging = true;
    dragStartX = e.clientX;
    dragOffset = seeker.offsetLeft;
    document.addEventListener('mousemove', handleSeekerDrag);
    document.addEventListener('mouseup', () => {
        isDragging = false;
        document.removeEventListener('mousemove', handleSeekerDrag);
    });
});

function handleSeekerDrag(e) {
    if (!isDragging) return;
    const rect = timeline.getBoundingClientRect();
    const newX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    seeker.style.left = `${newX}px`;
    video.currentTime = posToTime(newX);
}

// Add new label
addLabelBtn.addEventListener('click', () => {
    const label = prompt('Enter label name:');
    if (!label) return;

    const color = labelColors[nextColorIndex];
    nextColorIndex = (nextColorIndex + 1) % labelColors.length;

    const labelBtn = document.createElement('button');
    labelBtn.textContent = label;
    labelBtn.className = 'label-btn';
    labelBtn.style.backgroundColor = color;
    labelBtn.addEventListener('click', () => createBlock(label, color));
    labelsList.appendChild(labelBtn);
});

// Create new block
function createBlock(label, color) {
    const block = document.createElement('div');
    block.className = 'block';
    block.style.backgroundColor = color;
    block.style.left = `${timeToPos(video.currentTime)}px`;
    block.style.width = '100px';

    const leftHandle = document.createElement('div');
    leftHandle.className = 'block-handle left';
    const rightHandle = document.createElement('div');
    rightHandle.className = 'block-handle right';

    block.appendChild(leftHandle);
    block.appendChild(rightHandle);
    timeline.appendChild(block);

    setActiveBlock(block);
    setupBlockHandlers(block, label);
}

// Set active block and dim others
function setActiveBlock(block) {
    activeBlock = block;
    document.querySelectorAll('.block').forEach(b => {
        if (b !== block) {
            b.classList.add('dimmed');
            b.querySelectorAll('.block-handle').forEach(h => h.style.display = 'none');
        }
    });
    if (block) {
        block.classList.remove('dimmed');
        block.querySelectorAll('.block-handle').forEach(h => h.style.display = 'block');
    }
}

// Setup block drag handlers
function setupBlockHandlers(block, label) {
    let isResizing = false;
    let activeHandle = null;

    block.addEventListener('mousedown', (e) => {
        if (e.target.classList.contains('block-handle')) {
            isResizing = true;
            activeHandle = e.target;
            document.addEventListener('mousemove', handleResize);
            document.addEventListener('mouseup', stopResize);
        }
    });

    block.addEventListener('click', (e) => {
        e.stopPropagation();
        setActiveBlock(block);
    });

    function handleResize(e) {
        if (!isResizing) return;
        const rect = timeline.getBoundingClientRect();
        const newX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));

        // Check for overlapping with other blocks
        const currentLeft = parseFloat(block.style.left);
        const currentWidth = parseFloat(block.style.width);
        const otherBlocks = Array.from(document.querySelectorAll('.block')).filter(b => b !== block);

        if (activeHandle.classList.contains('left')) {
            const minX = otherBlocks.reduce((min, b) => {
                if (parseFloat(b.style.left) + parseFloat(b.style.width) <= currentLeft + currentWidth) {
                    return Math.max(min, parseFloat(b.style.left) + parseFloat(b.style.width));
                }
                return min;
            }, 0);

            if (newX >= minX && newX < currentLeft + currentWidth) {
                block.style.width = `${currentLeft + currentWidth - newX}px`;
                block.style.left = `${newX}px`;
            }
        } else {
            const maxX = otherBlocks.reduce((max, b) => {
                if (parseFloat(b.style.left) >= currentLeft) {
                    return Math.min(max, parseFloat(b.style.left));
                }
                return max;
            }, rect.width);

            if (newX > currentLeft && newX <= maxX) {
                block.style.width = `${newX - currentLeft}px`;
            }
        }

        video.currentTime = posToTime(newX);
    }

    function stopResize() {
        isResizing = false;
        document.removeEventListener('mousemove', handleResize);
    }
}

// Reset active block when clicking outside
document.addEventListener('click', (e) => {
    if (!e.target.closest('.block')) {
        setActiveBlock(null);
    }
});

// Run button handler
runBtn.addEventListener('click', () => {
    const blocks = Array.from(document.querySelectorAll('.block')).map(block => {
        const label = block.dataset.label;
        const startTime = posToTime(parseFloat(block.style.left));
        const endTime = posToTime(parseFloat(block.style.left) + parseFloat(block.style.width));
        return {
            label,
            frame_start: Math.round(startTime * video.fps),
            frame_end: Math.round(endTime * video.fps)
        };
    });

    fetch('/process_blocks', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(blocks)
    })
    .then(response => response.json())
    .then(data => console.log('Server response:', data))
    .catch(error => console.error('Error:', error));
});