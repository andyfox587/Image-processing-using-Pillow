#!/usr/bin/env python3
"""
Quick test to verify the frame reduction logic works
"""

def test_frame_selection():
    """Test the fixed frame selection logic"""
    # Simulate the fixed logic
    total_frames = 98
    target_frame_count = 20
    min_frames = max(6, total_frames // 10)  # Should be 9 for 98 frames
    
    actual_target = max(target_frame_count, min_frames, 6)  # Should be 20
    
    print(f"Total frames: {total_frames}")
    print(f"Target frames: {target_frame_count}")
    print(f"Min frames: {min_frames}")
    print(f"Actual target: {actual_target}")
    
    if actual_target >= total_frames:
        frames_to_keep = list(range(total_frames))
    else:
        step = total_frames / actual_target
        frames_to_keep = []
        
        # Always include first frame
        frames_to_keep.append(0)
        
        # Add evenly spaced frames
        for i in range(1, actual_target - 1):
            frame_idx = int(round(i * step))
            if frame_idx < total_frames and frame_idx not in frames_to_keep:
                frames_to_keep.append(frame_idx)
        
        # Always include last frame for proper loop
        if total_frames - 1 not in frames_to_keep:
            frames_to_keep.append(total_frames - 1)
            
        # Sort and ensure we don't exceed target
        frames_to_keep.sort()
        frames_to_keep = frames_to_keep[:actual_target]
    
    print(f"Selected {len(frames_to_keep)} frames: {frames_to_keep}")
    print(f"Step size: {total_frames / actual_target:.2f}")
    
    return len(frames_to_keep) == actual_target

if __name__ == "__main__":
    print("Testing frame selection logic...")
    success = test_frame_selection()
    print(f"Test {'PASSED' if success else 'FAILED'}")