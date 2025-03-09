#!/bin/bash
# setup.sh - Environment setup script for LiDAR VRU detection on Jetson Nano

set -e  # Exit on error

# Print colored status messages
function echo_status() {
    echo -e "\e[1;34m[*] $1\e[0m"
}

function echo_success() {
    echo -e "\e[1;32m[+] $1\e[0m"
}

function echo_error() {
    echo -e "\e[1;31m[!] $1\e[0m"
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo_error "Please run this script as root (use sudo)"
    exit 1
fi

echo_status "Setting up environment for LiDAR VRU Detection on Jetson Nano"

# Update package lists
echo_status "Updating package lists..."
apt-get update

# Install system dependencies
echo_status "Installing system dependencies..."
apt-get install -y build-essential cmake unzip pkg-config
apt-get install -y libopenblas-dev liblapack-dev
apt-get install -y libhdf5-serial-dev hdf5-tools
apt-get install -y python3-dev python3-pip

# Install Python dependencies
echo_status "Installing Python dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install numpy==1.19.4
python3 -m pip install scipy==1.5.4
python3 -m pip install scikit-learn==0.24.2
python3 -m pip install numba==0.54.1
python3 -m pip install open3d==0.13.0
python3 -m pip install matplotlib

# Install PyTorch for Jetson
echo_status "Installing PyTorch for Jetson..."
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
rm torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# Set up project directory
PROJECT_DIR="/opt/lidar_vru_detection"
echo_status "Creating project directory at $PROJECT_DIR..."
mkdir -p $PROJECT_DIR

# Copy project files
echo_status "Copying project files..."
cp lidar_vru_detection.py $PROJECT_DIR/
cp visualization.py $PROJECT_DIR/
cp run.py $PROJECT_DIR/
chmod +x $PROJECT_DIR/run.py

# Create data and results directories
mkdir -p $PROJECT_DIR/data
mkdir -p $PROJECT_DIR/results

# Set permissions
echo_status "Setting permissions..."
chown -R $SUDO_USER:$SUDO_USER $PROJECT_DIR

echo_success "Setup complete!"
echo ""
echo "You can now run the detection system with:"
echo "cd $PROJECT_DIR"
echo "./run.py --input_dir /path/to/scans --output_dir /path/to/results"
echo ""
echo "For the challenge evaluation:"
echo "./run.py --input_dir /path/to/test_data/scans --output_dir ./results"
