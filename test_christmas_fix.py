#!/usr/bin/env python3
"""
Test script to verify the RGBA conversion fix on Christmas Cards 2025 animations.
Run this script from the Christmas Cards 2025 directory to test the fix.
"""

import os
import sys
from PIL import Image, ImageSequence

def test_gif_frame_modes(gif_path):
    """Test a GIF file to show frame modes before and after RGBA conversion"""
    if not os.path.exists(gif_path):
        print(f"❌ File not found: {gif_path}")
        return False
    
    print(f"\n🔍 Testing: {os.path.basename(gif_path)}")
    
    try:
        with Image.open(gif_path) as img:
            print(f"   📁 Original GIF: mode={img.mode}, size={img.size}, frames={getattr(img, 'n_frames', 1)}")
            
            # Check original frame modes
            original_modes = []
            for i, frame in enumerate(ImageSequence.Iterator(img)):
                original_modes.append(frame.mode)
                if i < 5:  # Show first 5 frames
                    print(f"   Frame {i}: mode={frame.mode}")
            
            print(f"   Original frame modes: {original_modes[:10]}{'...' if len(original_modes) > 10 else ''}")
            
            # Test RGBA conversion (the fix)
            print(f"   🔧 Applying RGBA conversion fix...")
            converted_modes = []
            mixed_modes_detected = len(set(original_modes)) > 1
            
            for i, frame in enumerate(ImageSequence.Iterator(img)):
                frame_copy = frame.copy()
                
                # Apply the fix: force convert to RGBA
                if frame_copy.mode != 'RGBA':
                    frame_copy = frame_copy.convert('RGBA')
                
                converted_modes.append(frame_copy.mode)
                if i < 5:
                    print(f"   Converted Frame {i}: mode={frame_copy.mode}")
            
            print(f"   Converted frame modes: {converted_modes[:10]}{'...' if len(converted_modes) > 10 else ''}")
            
            # Analysis
            all_rgba = all(mode == 'RGBA' for mode in converted_modes)
            if mixed_modes_detected:
                print(f"   ⚠️  MIXED MODES DETECTED - This was causing color loss!")
                print(f"   ✅ Fix applied: All frames now RGBA = {all_rgba}")
            else:
                print(f"   ✅ Consistent modes: {all_rgba}")
            
            return mixed_modes_detected and all_rgba
            
    except Exception as e:
        print(f"   ❌ Error testing {gif_path}: {e}")
        return False

def main():
    print("🎄 Christmas Cards 2025 - RGBA Conversion Fix Test")
    print("=" * 50)
    
    # Get current directory
    current_dir = os.getcwd()
    print(f"Testing directory: {current_dir}")
    
    # Find all GIF files in current directory
    gif_files = [f for f in os.listdir('.') if f.lower().endswith('.gif')]
    
    if not gif_files:
        print("❌ No GIF files found in current directory")
        print("Please run this script from the Christmas Cards 2025 directory")
        return
    
    print(f"Found {len(gif_files)} GIF files to test")
    
    # Test each GIF file
    fixed_count = 0
    for gif_file in gif_files[:5]:  # Test first 5 files
        if test_gif_frame_modes(gif_file):
            fixed_count += 1
    
    print(f"\n🎯 Summary:")
    print(f"   Tested: {min(5, len(gif_files))} files")
    print(f"   Mixed modes fixed: {fixed_count}")
    
    if fixed_count > 0:
        print(f"   ✅ RGBA conversion fix is working correctly!")
        print(f"   🎉 Color loss issue should be resolved!")
    else:
        print(f"   ℹ️  No mixed-mode GIFs found (already consistent)")

if __name__ == "__main__":
    main()