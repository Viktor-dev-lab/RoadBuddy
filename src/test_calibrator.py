"""
Test Camera Calibration
Chạy: python test_calibration.py
"""

import sys
from pathlib import Path
import cv2

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils.camera_calibrator import CameraCalibrator


def main():
    """Main calibration workflow"""
    
    VIDEO_PATH = Path("data/videos/Dashcam2.mp4")
    CALIBRATION_OUTPUT = Path("config/camera_calibration.json")
    
    print("\n" + "=" * 80)
    print("🎯 CAMERA CALIBRATION FOR DASHCAM")
    print("=" * 80)
    print("\nThis will help you calculate accurate pixels_per_meter for your dashcam")
    print("\nAvailable methods:")
    print("  1. Lane Marking  - RECOMMENDED (most accurate)")
    print("  2. Known Object  - Good (use car, person, traffic sign)")
    print("  3. Perspective   - Advanced (depth measurement)")
    print("  4. Interactive   - Use multiple methods and average")
    print("=" * 80)
    
    if not VIDEO_PATH.exists():
        print(f"\n❌ ERROR: Video not found: {VIDEO_PATH}")
        print("\n💡 Please:")
        print("   1. Place your dashcam video in data/videos/")
        print("   2. Or update VIDEO_PATH in this script")
        return
    
    # Read middle frame
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Cannot read video")
        return
    
    # Create calibrator
    calibrator = CameraCalibrator(VIDEO_PATH)
    
    # Choose method
    print("\nSelect calibration method (1/2/3/4): ", end="")
    choice = input().strip()
    
    pixels_per_meter = None
    
    if choice == "1":
        # Lane marking
        print("\n📏 LANE MARKING CALIBRATION")
        print("   Vietnam standard: 3.0-3.5m lane width")
        print("   Highway: 3.5m")
        print("   City road: 3.0-3.2m")
        print("\nEnter lane width in meters (default 3.5): ", end="")
        
        lane_width = input().strip()
        lane_width = float(lane_width) if lane_width else 3.5
        
        pixels_per_meter = calibrator.calibrate_from_lane_marking(frame, lane_width)
    
    elif choice == "2":
        # Known object
        print("\n🚗 KNOWN OBJECT CALIBRATION")
        print("   Available objects:")
        print("   - car: 4.5m length (default)")
        print("   - bus: 12m length")
        print("   - person: 1.7m height")
        print("\nSelect object type (car/bus/person): ", end="")
        
        object_type = input().strip()
        object_type = object_type if object_type in ['car', 'bus', 'person'] else 'car'
        
        pixels_per_meter = calibrator.calibrate_from_known_object(frame, object_type)
    
    elif choice == "3":
        # Perspective
        print("\n📐 PERSPECTIVE CALIBRATION")
        print("   Measure known distance in depth direction")
        print("\nEnter reference distance in meters (default 10): ", end="")
        
        distance = input().strip()
        distance = float(distance) if distance else 10.0
        
        pixels_per_meter = calibrator.calibrate_with_perspective(frame, distance)
    
    elif choice == "4":
        # Interactive multi-method
        results = calibrator.calibrate_interactive(frame)
        pixels_per_meter = results.get('average')
    
    else:
        print("❌ Invalid choice")
        return
    
    if pixels_per_meter is None:
        print("\n❌ Calibration failed")
        return
    
    # Save calibration
    print("\n" + "=" * 80)
    print("Do you want to save this calibration? (y/n): ", end="")
    if input().strip().lower() == 'y':
        CALIBRATION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        calibrator.save_calibration(CALIBRATION_OUTPUT)
        
        print("\n✅ Calibration saved!")
        print(f"   File: {CALIBRATION_OUTPUT}")
        print(f"   Scale: {pixels_per_meter:.2f} pixels/meter")
        print("\n💡 Usage in code:")
        print(f"   pixels_per_meter = {pixels_per_meter:.2f}")
        print("   or")
        print(f"   from src.utils.camera_calibrator import CameraCalibrator")
        print(f"   ppm = CameraCalibrator.load_calibration('{CALIBRATION_OUTPUT}')")
    
    print("=" * 80)


if __name__ == "__main__":
    main()


# ============================================================
# QUICK CALIBRATION GUIDE
# ============================================================
"""
📋 CALIBRATION GUIDE:

1. LANE MARKING METHOD (RECOMMENDED):
   - Best for highway/city roads
   - Click left edge of lane → right edge
   - Use 3.5m for highway, 3.0m for city
   - Accuracy: ±5%

2. KNOWN OBJECT METHOD:
   - Use when lane marking not clear
   - Car: 4.5m (sedan), Bus: 12m
   - Click front → back of vehicle
   - Accuracy: ±10%

3. PERSPECTIVE METHOD:
   - Advanced, for experienced users
   - Measure depth (near → far)
   - Need known distance marker
   - Accuracy: ±15%

4. INTERACTIVE METHOD:
   - Use multiple methods
   - Take average for best result
   - Accuracy: ±3%

💡 TIPS:
- Choose frame with clear road markings
- Measure at middle distance (not too near/far)
- For dashcam, adaptive calibration is better
- Recalibrate for different camera positions

📊 TYPICAL VALUES:
- Highway dashcam: 20-30 pixels/meter
- City dashcam: 15-25 pixels/meter
- Varies with camera angle and resolution
"""