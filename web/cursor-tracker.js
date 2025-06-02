let isTracking = false;
let cursorData = [];
let startTime = null;
let lastRecordedPosition = null;
let lastMovementTime = null;
let trackingTimer = null;

const movementThreshold = 5; 
const idleThreshold = 3000; 
const trackingInterval = 100; 

const overlay = document.getElementById('overlay');
const startButton = document.getElementById('start-button');
const endButton = document.getElementById('end-button');
const statusDiv = document.getElementById('status');

startButton.addEventListener('click', () => {
    isTracking = true;
    cursorData = [];
    startTime = Date.now();
    lastRecordedPosition = null;
    lastMovementTime = Date.now();
    statusDiv.textContent = 'Status: Tracking';
    console.log('Tracking started.');

    trackingTimer = setInterval(() => {
        if (isTracking && lastRecordedPosition) {
            const timestamp = Date.now() - startTime;
            cursorData.push({
                x: lastRecordedPosition.x,
                y: lastRecordedPosition.y,
                timestamp
            });
        }
    }, trackingInterval);
});

endButton.addEventListener('click', () => {
    isTracking = false;
    clearInterval(trackingTimer);
    statusDiv.textContent = 'Status: Not Tracking';
    console.log('Tracking stopped.');
    downloadData();
});

overlay.addEventListener('mousemove', (event) => {
    if (isTracking) {
        const x = event.clientX;
        const y = event.clientY;

        if (!lastRecordedPosition) {
            lastRecordedPosition = { x, y };
        }

        const dx = x - lastRecordedPosition.x;
        const dy = y - lastRecordedPosition.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance > movementThreshold) {
            lastRecordedPosition = { x, y }; 
            lastMovementTime = Date.now(); 
        }
    }
});
function downloadData() {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(cursorData));
    
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    
    const sessionTime = new Date().toISOString().replace(/[:.]/g, '-');
    downloadAnchorNode.setAttribute("download", `cursor_data_${sessionTime}.json`);
    
    document.body.appendChild(downloadAnchorNode);
    
    downloadAnchorNode.click();
    
    downloadAnchorNode.remove();
}
