#!/bin/bash
#
# solidworks_to_ros2.sh
#
# Takes a SolidWorks URDF export folder and:
#   1. Converts it to a ROS 2 package (fixes mesh paths, joint limits)
#   2. Builds the ROS 2 workspace
#   3. Launches RViz
#
# Usage:
#   ./solidworks_to_ros2.sh /path/to/solidworks_export [package_name]
#
# Example:
#   ./solidworks_to_ros2.sh ~/Downloads/KAKA kaka_description
#
# If package_name is not provided, it uses the folder name in lowercase.

set -e

# --- Parse arguments ---
if [ $# -lt 1 ]; then
    echo "Usage: $0 /path/to/solidworks_export [package_name]"
    echo ""
    echo "Example: $0 ~/Downloads/KAKA kaka_description"
    exit 1
fi

EXPORT_DIR="$(realpath "$1")"
FOLDER_NAME="$(basename "$EXPORT_DIR")"
PKG_NAME="${2:-$(echo "$FOLDER_NAME" | tr '[:upper:]' '[:lower:]')_description}"
WS_DIR="$HOME/repos/ros2_ws"
PKG_DIR="$WS_DIR/$PKG_NAME"

echo "============================================"
echo "SolidWorks → ROS 2 Pipeline"
echo "============================================"
echo "Input folder:  $EXPORT_DIR"
echo "Package name:  $PKG_NAME"
echo ""

# --- Validate input ---
URDF_FILE=$(find "$EXPORT_DIR" -name "*.urdf" -type f | head -1)
if [ -z "$URDF_FILE" ]; then
    echo "ERROR: No .urdf file found in $EXPORT_DIR"
    exit 1
fi
echo "Found URDF: $URDF_FILE"

MESH_DIR=$(find "$EXPORT_DIR" -name "meshes" -type d | head -1)

# --- Step 1: Create ROS 2 package structure ---
echo ""
echo "[Step 1] Creating ROS 2 package: $PKG_NAME"

mkdir -p "$PKG_DIR/urdf"
mkdir -p "$PKG_DIR/meshes"
mkdir -p "$PKG_DIR/launch"
mkdir -p "$PKG_DIR/rviz"

# --- Step 2: Copy and fix URDF ---
echo "[Step 2] Fixing URDF mesh paths and joint limits"

ROBOT_NAME=$(grep -oP 'name="[^"]*"' "$URDF_FILE" | head -1 | grep -oP '"[^"]*"' | tr -d '"')
echo "  Robot name: $ROBOT_NAME"

URDF_BASENAME=$(basename "$URDF_FILE")
cp "$URDF_FILE" "$PKG_DIR/urdf/$URDF_BASENAME"

# Replace any package:// references with the new package name
sed -i "s|package://[^/]*/|package://${PKG_NAME}/|g" "$PKG_DIR/urdf/$URDF_BASENAME"

# Fix SolidWorks joint limit artifact (lower=upper=0)
sed -i 's/lower="0" upper="0"/lower="-3.1416" upper="3.1416"/g' "$PKG_DIR/urdf/$URDF_BASENAME"
sed -i 's/lower="0.0" upper="0.0"/lower="-3.1416" upper="3.1416"/g' "$PKG_DIR/urdf/$URDF_BASENAME"

echo "  Fixed mesh paths: package://$PKG_NAME/"
echo "  Fixed zero joint limits → ±180°"

# --- Step 3: Copy meshes ---
if [ -n "$MESH_DIR" ]; then
    echo "[Step 3] Copying meshes"
    cp "$MESH_DIR"/* "$PKG_DIR/meshes/" 2>/dev/null || true
    MESH_COUNT=$(ls "$PKG_DIR/meshes/" | wc -l)
    echo "  Copied $MESH_COUNT mesh files"
else
    echo "[Step 3] No meshes to copy"
fi

# --- Step 4: Generate ROS 2 package files ---
echo "[Step 4] Generating ROS 2 package files"

cat > "$PKG_DIR/package.xml" << PKGEOF
<?xml version="1.0"?>
<package format="3">
  <name>$PKG_NAME</name>
  <version>1.0.0</version>
  <description>ROS 2 package for $ROBOT_NAME (converted from SolidWorks)</description>
  <maintainer email="user@example.com">user</maintainer>
  <license>MIT</license>

  <buildtool_depend>ament_cmake</buildtool_depend>
  <depend>robot_state_publisher</depend>
  <depend>joint_state_publisher_gui</depend>
  <depend>rviz2</depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
PKGEOF

cat > "$PKG_DIR/CMakeLists.txt" << CMEOF
cmake_minimum_required(VERSION 3.8)
project($PKG_NAME)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY urdf meshes launch rviz
  DESTINATION share/\${PROJECT_NAME}
)

ament_package()
CMEOF

cat > "$PKG_DIR/launch/display.launch.py" << LAUNCHEOF
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_dir = get_package_share_directory('$PKG_NAME')
    urdf_file = os.path.join(pkg_dir, 'urdf', '$URDF_BASENAME')
    rviz_file = os.path.join(pkg_dir, 'rviz', 'config.rviz')

    with open(urdf_file, 'r') as f:
        robot_desc = f.read()

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_desc}],
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_file],
        ),
    ])
LAUNCHEOF

cat > "$PKG_DIR/rviz/config.rviz" << RVIZEOF
Panels:
  - Class: rviz_common/Displays
Visualization Manager:
  Displays:
    - Class: rviz_default_plugins/RobotModel
      Name: RobotModel
      Enabled: true
      Description Topic:
        Value: /robot_description
    - Class: rviz_default_plugins/TF
      Name: TF
      Enabled: true
  Global Options:
    Fixed Frame: base_link
RVIZEOF

echo "  Generated: package.xml, CMakeLists.txt, launch file, rviz config"

# --- Step 5: Build and launch ---
echo "[Step 5] Building ROS 2 workspace"
cd "$WS_DIR"
source /opt/ros/jazzy/setup.bash
colcon build --packages-select "$PKG_NAME" 2>&1 | tail -3
source install/setup.bash
echo "  Build complete"

echo ""
echo "============================================"
echo "Launching RViz..."
echo "============================================"
ros2 launch "$PKG_NAME" display.launch.py
