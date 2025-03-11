const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const socket = io();
let pointCloud;
let boundingBoxes = [];

// Adjust camera position for a better view
camera.position.set(0, 10, 50);
camera.lookAt(0, 0, 0);

// Mouse control variables
let isDragging = false;
let previousMousePosition = { x: 0, y: 0 };

document.addEventListener("mousedown", (event) => {
    isDragging = true;
    previousMousePosition = { x: event.clientX, y: event.clientY };
});

document.addEventListener("mouseup", () => {
    isDragging = false;
});

document.addEventListener("mousemove", (event) => {
    if (!isDragging) return;
    
    let deltaX = event.clientX - previousMousePosition.x;
    let deltaY = event.clientY - previousMousePosition.y;
    
    let rotationSpeed = 0.005;
    camera.rotation.y -= deltaX * rotationSpeed;
    camera.rotation.x -= deltaY * rotationSpeed;
    
    previousMousePosition = { x: event.clientX, y: event.clientY };
});

// Define colors for different object labels
const labelColors = {
    "Vehicle.Car": 0xff0000,      // Red
    "human.pedestrian.adult": 0x00ff00, // Green
    "vehicle.bicycle": 0x0000ff,   // Blue
    "Truck": 0xffff00,    // Yellow
    "vehicle.motorcycle": 0xff00ff // Magenta
};

// Create a function to visualize LiDAR points
function createPointCloud(points) {
    if (pointCloud) scene.remove(pointCloud);
    const geometry = new THREE.BufferGeometry();
    const vertices = new Float32Array(points.flat());
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    const material = new THREE.PointsMaterial({ color: 0xffffff, size: 0.1 });
    pointCloud = new THREE.Points(geometry, material);
    scene.add(pointCloud);
}

// Create bounding boxes with correct axis alignment
function createBoundingBoxes(boxes) {
    boundingBoxes.forEach(box => scene.remove(box));
    boundingBoxes = [];
    
    boxes.forEach(box => {
        const geometry = new THREE.BoxGeometry(box.l, box.h, box.w); // Swap width & length
        const color = labelColors[box.label] || 0xffffff; // Default to white if label not found
        const material = new THREE.MeshBasicMaterial({ color: color, wireframe: true });
        const mesh = new THREE.Mesh(geometry, material);
        
        // Adjust position (swap Y and Z for correct orientation)
        mesh.position.set(box.x, box.y + box.h / 2, box.z);
        
        // Apply yaw rotation (Three.js uses radians, LiDAR yaw may be in degrees)
        mesh.rotation.y = -box.yaw; 
        scene.add(mesh);
        boundingBoxes.push(mesh);
    });
}

// Listen for real-time data updates
socket.on('frame_data', data => {
    const parsed = JSON.parse(data);
    createPointCloud(parsed.points);
    createBoundingBoxes(parsed.boxes);
});

// Start animation
function startAnimation() {
    const start = document.getElementById('start').value;
    const end = document.getElementById('end').value;
    socket.emit('start_animation', { start, end });
}

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}
animate();