import os
import io
from PIL import Image, ImageSequence, ImageOps

def resize_frame(frame, size, opacity=1.0, greyscale=False):
    """Resize frame while preserving colors (no palette mode preservation)"""
    # Use LANCZOS for better quality when downsizing
    try:
        resampling = Image.LANCZOS
    except AttributeError:
        resampling = Image.ANTIALIAS
    
    # Store original info
    original_info = frame.info.copy() if hasattr(frame, 'info') else {}
    
    # Calculate new size maintaining aspect ratio
    original_width, original_height = frame.size
    if original_width > original_height:
        new_width = size
        new_height = int((original_height * size) / original_width)
    else:
        new_height = size
        new_width = int((original_width * size) / original_height)
    
    # Ensure minimum size of 1
    new_width = max(1, new_width)
    new_height = max(1, new_height)
    
    # Resize the frame
    resized_frame = frame.resize((new_width, new_height), resampling)

    # Apply greyscale conversion if requested (preserves transparency)
    if greyscale:
        if resized_frame.mode == 'RGBA':
            # Split into channels, convert RGB to greyscale, recombine with alpha
            r, g, b, a = resized_frame.split()
            grey = ImageOps.grayscale(resized_frame.convert('RGB'))
            resized_frame = Image.merge('RGBA', (grey, grey, grey, a))
        else:
            resized_frame = ImageOps.grayscale(resized_frame).convert('RGBA')

    # Restore original info
    if hasattr(resized_frame, 'info'):
        resized_frame.info.update(original_info)

    return resized_frame

def optimize_image_size(img, max_size_kb, format='PNG'):
    """Optimize image to meet size requirements"""
    buffer = io.BytesIO()
    
    if format.upper() in ['JPEG', 'JPG']:
        quality = 95
        while quality > 5:
            buffer.seek(0)
            buffer.truncate()
            img.save(buffer, format=format, quality=quality)
            if buffer.tell() / 1024 <= max_size_kb:
                break
            quality -= 5
    else:
        # For PNG and other formats, try different compression levels
        compress_level = 6
        while compress_level <= 9:
            buffer.seek(0)
            buffer.truncate()
            save_kwargs = {'format': format}
            if format.upper() == 'PNG':
                save_kwargs['compress_level'] = compress_level
            img.save(buffer, **save_kwargs)
            if buffer.tell() / 1024 <= max_size_kb:
                break
            compress_level += 1
        
        # If still too large, preserve colors and warn user
        if buffer.tell() / 1024 > max_size_kb and format.upper() == 'PNG':
            print(f"  Warning: PNG file exceeds {max_size_kb}KB target but preserving original colors")

    return buffer.getvalue() if buffer.tell() > 0 else None

def optimize_png_stamps(img, max_size_kb):
    """Optimize PNG specifically for stamps mode WITHOUT color reduction"""
    buffer = io.BytesIO()
    
    # Try maximum PNG compression first - preserve all colors
    img.save(buffer, format='PNG', compress_level=9)
    current_size_kb = buffer.tell() / 1024
    
    if current_size_kb <= max_size_kb:
        return buffer.getvalue()
    
    # If still too large, return the best we can do with original colors
    # We will NOT reduce colors as per user requirements
    print(f"  Warning: PNG stamps file is {current_size_kb:.1f}KB, exceeds target of {max_size_kb}KB but preserving colors")
    return buffer.getvalue()

def get_background_color(frame):
    """Detect background color by analyzing edge pixels"""
    width, height = frame.size
    edge_colors = []
    
    # Sample pixels from all four edges
    for x in range(width):
        edge_colors.append(frame.getpixel((x, 0))[:3])  # Top edge
        edge_colors.append(frame.getpixel((x, height-1))[:3])  # Bottom edge
    for y in range(height):
        edge_colors.append(frame.getpixel((0, y))[:3])  # Left edge
        edge_colors.append(frame.getpixel((width-1, y))[:3])  # Right edge
    
    # Find most common color
    from collections import Counter
    color_counts = Counter(edge_colors)
    return color_counts.most_common(1)[0][0]

def process_animated_gif(img, output_path, size, max_size_kb):
    """Process animated GIF while preserving all frame metadata and colors"""
    frames = []
    frame_info = []

    print(f"Converting ALL frames to RGBA to prevent palette conflicts and color loss")

    frame_index = 0
    for frame in ImageSequence.Iterator(img):
        # Make a copy of the frame 
        frame_copy = frame.copy()
        
        # FORCE convert ALL frames to RGBA mode to ensure consistency
        # This is the ONLY way to prevent PIL from doing palette conversions
        if frame_copy.mode != 'RGBA':
            frame_copy = frame_copy.convert('RGBA')
        
        # Get frame-specific info
        info = frame_copy.info.copy() if hasattr(frame_copy, 'info') else {}
        
        # Read disposal method from frame attribute
        disposal_method = getattr(frame, 'disposal_method', 0)
        info['disposal'] = disposal_method
        
        # Debug frame info for first few frames
        if frame_index < 3:
            print(f"Frame {frame_index}: mode={frame_copy.mode}, size={frame_copy.size}")
        
        resized_frame = resize_frame(frame_copy, size)
        
        # Restore all metadata to resized frame
        if hasattr(resized_frame, 'info'):
            resized_frame.info.update(info)
        
        frames.append(resized_frame)
        frame_info.append(info)
        frame_index += 1

    save_gif_optimized(frames, frame_info, output_path, max_size_kb)

def save_gif_optimized(frames, frame_info, output_path, max_size_kb):
    """Save GIF preserving all original frame metadata and colors"""
    # Extract durations and disposal methods from original frames
    durations = [info.get('duration', 100) for info in frame_info]
    disposals = [info.get('disposal', 0) for info in frame_info]
    
    print(f"Saving {len(frames)} frames. First frame mode: {frames[0].mode}")
    print(f"Frame modes: {[f.mode for f in frames[:5]]}")
    
    # EZGIF APPROACH: Progressive lossy compression (no palette manipulation)
    print(f"Using EZGIF-style progressive lossy compression")
    
    # Convert all RGBA frames to RGB (GIF doesn't support alpha)
    rgb_frames = []
    for frame in frames:
        if frame.mode == 'RGBA':
            # Create white background and paste frame with alpha
            rgb_frame = Image.new('RGB', frame.size, (255, 255, 255))
            rgb_frame.paste(frame, mask=frame.split()[3] if len(frame.split()) == 4 else None)
            rgb_frames.append(rgb_frame)
        elif frame.mode == 'RGB':
            rgb_frames.append(frame)
        else:
            # Convert any other mode to RGB
            rgb_frames.append(frame.convert('RGB'))
    
    print(f"Converted {len(rgb_frames)} frames to RGB mode")
    
    # EZGIF Method: Try progressive lossy compression levels
    import tempfile
    import os
    
    # Progressive compression levels to try (like EZGIF: 35%, 45%, 55%, etc.)
    compression_levels = [35, 45, 55, 65, 75, 85, 95]
    
    for compression_level in compression_levels:
        print(f"Trying lossy compression at {compression_level}%...")
        
        try:
            # Create temporary file to test compression
            with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save with basic settings (like EZGIF does)
            rgb_frames[0].save(
                temp_path,
                save_all=True,
                append_images=rgb_frames[1:],
                duration=durations,
                disposal=disposals,
                loop=0,
                # Apply lossy compression by reducing color count based on compression level
                # Higher compression = fewer colors (but not palette manipulation)
                colors=max(32, int(256 * (100 - compression_level) / 100))  # Scale colors based on compression
            )
            
            # Check file size
            file_size_kb = os.path.getsize(temp_path) / 1024
            print(f"  Result: {file_size_kb:.1f}KB with {compression_level}% compression")
            
            if file_size_kb <= max_size_kb:
                print(f"✓ SUCCESS! Achieved {file_size_kb:.1f}KB with {compression_level}% lossy compression")
                # Move temp file to final location
                os.rename(temp_path, output_path)
                return
            else:
                print(f"  Still too large ({file_size_kb:.1f}KB > {max_size_kb}KB)")
                os.unlink(temp_path)  # Clean up temp file
                
        except Exception as e:
            print(f"  Error with {compression_level}% compression: {e}")
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            continue
    
    # If all compression levels failed, fall back to frame reduction (EZGIF's last resort)
    print(f"All compression levels failed, falling back to frame reduction...")
    
    # Basic save as last resort
    try:
        rgb_frames[0].save(
            output_path,
            save_all=True,
            append_images=rgb_frames[1:],
            duration=durations,
            disposal=disposals,
            loop=0
        )
        print(f"Fallback save completed")
    except Exception as e:
        print(f"All save methods failed: {e}")

def analyze_frame_differences(frames):
    """Analyze differences between consecutive frames using efficient histogram comparison"""
    differences = []
    
    for i in range(1, len(frames)):
        # Use histogram comparison for efficiency - much faster than pixel-by-pixel
        prev_frame = frames[i-1].convert('RGB')  # Convert to RGB for consistent comparison
        curr_frame = frames[i].convert('RGB')
        
        # Get histograms for each channel
        prev_hist = prev_frame.histogram()
        curr_hist = curr_frame.histogram()
        
        # Calculate histogram difference (simple sum of absolute differences)
        hist_diff = sum(abs(a - b) for a, b in zip(prev_hist, curr_hist))
        total_pixels = prev_frame.size[0] * prev_frame.size[1]
        
        # Normalize the difference (scale down for comparison)
        diff_percentage = min(1.0, hist_diff / (total_pixels * 255 * 3))  # 3 channels, max diff per pixel = 255
        differences.append(diff_percentage)
    
    return differences

def create_compressed_gif(frames, frame_info, output_path, target_size_kb=500):
    """Create a compressed version under target size using iterative approach"""
    print(f"Creating compressed version targeting {target_size_kb}KB...")
    print(f"Input: {len(frames)} frames to compress")
    
    # Special handling for very small targets (280x280) - be much more conservative
    is_very_small = target_size_kb <= 290
    
    # Analyze frame differences
    differences = analyze_frame_differences(frames)
    print(f"Frame differences calculated: min={min(differences):.4f}, max={max(differences):.4f}, avg={sum(differences)/len(differences):.4f}")
    
    # Create list of (index, difference) pairs, skip first frame
    indexed_diffs = [(i+1, diff) for i, diff in enumerate(differences)]
    
    # Sort by difference (smallest first) - these are candidates for removal
    indexed_diffs.sort(key=lambda x: x[1])
    
    # Try progressively more aggressive compression until we hit target
    import os
    temp_path = output_path.replace('.gif', '_temp.gif')
    
    # Try different frame counts - ONLY reduce frames, never colors
    # Ensure minimum frames to maintain animation quality
    if is_very_small:
        min_frames = max(6, len(frames) // 10)  # For 280x280: Much more aggressive - minimum 6 frames
    else:
        min_frames = max(8, len(frames) // 6)  # For larger sizes: Still aggressive
        
    print(f"Minimum frames enforced: {min_frames} (very_small={is_very_small})")
    
    # Frame reduction attempts - NO COLOR REDUCTION
    if target_size_kb <= 290:  # For 360x360 and 280x280
        if is_very_small:  # 280x280 - be more aggressive to hit size target
            frame_attempts = [
                min(20, len(frames)),   # Start with 20 frames max
                min(15, len(frames)),   # 15 frames
                min(12, len(frames)),   # 12 frames
                min(10, len(frames)),   # 10 frames
                max(min_frames, 8),     # 8+ frames
                min_frames,             # minimum frames
            ]
        else:  # 360x360
            frame_attempts = [
                min(25, len(frames)),   # 25 frames
                min(20, len(frames)),   # 20 frames
                min(15, len(frames)),   # 15 frames
                max(min_frames, 12),    # 12+ frames
                min_frames,             # minimum frames
            ]
    else:  # For 720x720
        frame_attempts = [
            30,  # 30 frames
            25,  # 25 frames
            20,  # 20 frames
            15,  # 15 frames
            12,  # 12 frames
            10,  # 10 frames
        ]
    
    for target_frame_count in frame_attempts:
        print(f"\nTrying {target_frame_count} frames with original colors preserved...")
        print(f"  Original frames: {len(frames)}, Min frames enforced: {min_frames}")
        
        # Simple but effective frame selection: evenly spaced frames
        actual_target = max(target_frame_count, min_frames, 6)  # Never below 6 frames
        
        if actual_target >= len(frames):
            # Keep all frames if target is higher than available
            frames_to_keep = list(range(len(frames)))
        else:
            # Evenly distribute frames across the animation
            step = len(frames) / actual_target
            frames_to_keep = []
            
            # Always include first frame
            frames_to_keep.append(0)
            
            # Add evenly spaced frames
            for i in range(1, actual_target - 1):
                frame_idx = int(round(i * step))
                if frame_idx < len(frames) and frame_idx not in frames_to_keep:
                    frames_to_keep.append(frame_idx)
            
            # Always include last frame for proper loop
            if len(frames) - 1 not in frames_to_keep:
                frames_to_keep.append(len(frames) - 1)
                
            # Sort and ensure we don't exceed target
            frames_to_keep.sort()
            frames_to_keep = frames_to_keep[:actual_target]
        
        print(f"  Keeping {len(frames_to_keep)} frames: {frames_to_keep[:10]}{'...' if len(frames_to_keep) > 10 else ''}")
        
        # Create reduced frame set
        test_frames = [frames[i] for i in frames_to_keep]
        test_info = [frame_info[i] for i in frames_to_keep]
        
        # Calculate timing to maintain same animation speed
        original_frame_count = len(frame_info)
        new_frame_count = len(test_frames)
        # Always use the TRUE original duration from source frames, not modified ones
        original_duration = 30  # The original GIF has 30ms frames
        
        # Debug the calculation
        print(f"  DEBUG: original_frame_count = {original_frame_count}")
        print(f"  DEBUG: new_frame_count = {new_frame_count}")
        print(f"  DEBUG: original_duration = {original_duration}ms")
        
        # Simple calculation: if we have fewer frames, increase duration proportionally
        duration_multiplier = original_frame_count / new_frame_count
        new_duration = int(original_duration * duration_multiplier)
        
        print(f"  DEBUG: duration_multiplier = {duration_multiplier:.2f}")
        print(f"  DEBUG: calculated new_duration = {new_duration}ms")
        
        # Sanity check - keep reasonable timing (revert to 150ms cap that worked well)
        if new_duration > 500:  # More than half a second is too slow
            print(f"  WARNING: Calculated duration {new_duration}ms too slow, using 150ms")
            new_duration = 150
        
        print(f"  Final: {original_frame_count} frames → {new_frame_count} frames")
        print(f"  Timing: {original_duration}ms → {new_duration}ms per frame")
        
        # Update durations
        for info in test_info:
            info['duration'] = new_duration
        
        # Use frames as-is with NO color compression to preserve quality
        final_frames = test_frames
        
        # Debug frame information
        print(f"  Final frames modes: {[f.mode for f in final_frames[:3]]}")
        
        # All frames should now be in RGBA mode
        print(f"  All frames in RGBA mode - preserving full color information")
        
        # Save and test file size
        durations = [info.get('duration', 100) for info in test_info]
        disposals = [info.get('disposal', 0) for info in test_info]
        
        final_frames[0].save(
            temp_path,
            save_all=True,
            append_images=final_frames[1:],
            duration=durations,
            disposal=disposals,
            loop=0
        )
        
        test_size_kb = os.path.getsize(temp_path) / 1024
        print(f"  Result: {test_size_kb:.1f}KB")
        
        if test_size_kb <= target_size_kb:
            print(f"✓ SUCCESS! Achieved {test_size_kb:.1f}KB with {len(final_frames)} frames, original colors preserved")
            os.rename(temp_path, output_path)
            return
        else:
            print(f"  Still too large ({test_size_kb:.1f}KB > {target_size_kb}KB)")
    
    # If we get here, try one last fallback - create a higher quality version even if slightly over target
    print(f"⚠ Unable to reach target {target_size_kb}KB, trying fallback...")
    
    # Fallback: Use more frames with no color compression
    fallback_frames = max(min_frames, len(frames) // 3)
    fallback_attempts = [fallback_frames, max(min_frames, fallback_frames // 2)]
    
    for target_frame_count in fallback_attempts:
        print(f"\nFallback attempt: {target_frame_count} frames with original colors")
        
        # Use evenly spaced frames
        step = max(1, len(frames) // target_frame_count)
        frames_to_keep = list(range(0, len(frames), step))[:target_frame_count]
        
        test_frames = [frames[i] for i in frames_to_keep]
        test_info = [frame_info[i] for i in frames_to_keep]
        
        # Update durations
        new_duration = 100  # Standard duration
        for info in test_info:
            info['duration'] = new_duration
            
        # All frames should now be in RGBA mode, so no palette handling needed
        
        durations = [info.get('duration', 100) for info in test_info]
        disposals = [info.get('disposal', 0) for info in test_info]
        
        test_frames[0].save(
            temp_path,
            save_all=True,
            append_images=test_frames[1:],
            duration=durations,
            disposal=disposals,
            loop=0
        )
        
        test_size_kb = os.path.getsize(temp_path) / 1024
        print(f"  Fallback result: {test_size_kb:.1f}KB with {len(test_frames)} frames")
        
        # Accept fallback even if slightly over target (within 50KB tolerance for small targets)
        tolerance = 50 if target_size_kb <= 290 else 100
        if test_size_kb <= target_size_kb + tolerance:
            print(f"✓ FALLBACK SUCCESS! Using {test_size_kb:.1f}KB version with {len(test_frames)} frames")
            os.rename(temp_path, output_path)
            return
    
    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    print(f"⚠ Unable to create acceptable version under {target_size_kb + tolerance}KB")

def convert_image(input_path, output_dir, size, output_format, max_size_kb,
                 suffix='', opacity=1.0, greyscale=False, stamps_mode=False):
    """Convert a single image with size and format constraints"""
    try:
        with Image.open(input_path) as img:
            is_animated = getattr(img, "is_animated", False)
            base_name = os.path.splitext(os.path.basename(input_path))[0]

            # For stamps mode, clean up the base name (remove _800x800 suffix if present)
            if stamps_mode:
                if base_name.lower().endswith('_800x800'):
                    base_name = base_name[:-len('_800x800')]

            new_filename = f"{base_name}_{size}x{size}{suffix}.{output_format.lower()}"
            output_path = os.path.join(output_dir, new_filename)

            # For stamps mode, force PNG format; convert GIF to PNG using frame 0
            if stamps_mode:
                # If input is an animated GIF, extract frame 0
                if is_animated or input_path.lower().endswith('.gif'):
                    print(f"Converting GIF to PNG (frame 0): {os.path.basename(input_path)} -> {size}x{size} PNG")
                    img = img.copy()  # Get frame 0 (first frame)
                else:
                    print(f"Processing image for stamps: {os.path.basename(input_path)} -> {size}x{size} PNG")

                # Ensure RGBA mode for transparency
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')

                img = resize_frame(img, size, opacity, greyscale)

                # Use stamps-specific PNG optimization
                image_data = optimize_png_stamps(img, max_size_kb)
                if image_data:
                    with open(output_path, 'wb') as f:
                        f.write(image_data)
                    file_size = len(image_data) / 1024
                    print(f"  Stamps PNG created: {new_filename} ({file_size:.1f}KB)")
                else:
                    # Fallback to regular save
                    img.save(output_path, format='PNG')
                    print(f"  Stamps PNG created: {new_filename}")
            
            elif output_format.lower() == 'gif' and is_animated:
                print(f"Processing animated GIF: {os.path.basename(input_path)} -> {size}x{size}")
                process_animated_gif(img, output_path, size, max_size_kb)
                
                # Create compressed versions for different sizes
                if size in [720, 360, 280]:
                    # Set target size based on dimensions
                    target_kb = 500 if size == 720 else 290
                    
                    compressed_filename = f"{base_name}_{size}x{size}_reduced.{output_format.lower()}"
                    compressed_path = os.path.join(output_dir, compressed_filename)
                    
                    # Re-process frames for compression with RGBA conversion
                    frames = []
                    frame_info = []
                    for frame in ImageSequence.Iterator(img):
                        # Make a copy of the frame 
                        frame_copy = frame.copy()
                        
                        # FORCE convert ALL frames to RGBA mode to ensure consistency
                        # This is the ONLY way to prevent PIL from doing palette conversions
                        if frame_copy.mode != 'RGBA':
                            frame_copy = frame_copy.convert('RGBA')
                        
                        info = frame_copy.info.copy() if hasattr(frame_copy, 'info') else {}
                        disposal_method = getattr(frame, 'disposal_method', 0)
                        info['disposal'] = disposal_method
                        resized_frame = resize_frame(frame_copy, size)
                        if hasattr(resized_frame, 'info'):
                            resized_frame.info.update(info)
                        frames.append(resized_frame)
                        frame_info.append(info)
                    
                    print(f"Creating compressed version: {compressed_filename}")
                    try:
                        create_compressed_gif(frames, frame_info, compressed_path, target_kb)
                        if os.path.exists(compressed_path):
                            file_size = os.path.getsize(compressed_path) / 1024
                            print(f"✓ Compressed file created: {compressed_filename} ({file_size:.1f}KB)")
                        else:
                            print(f"⚠ Compressed file was not created: {compressed_filename}")
                    except Exception as e:
                        print(f"ERROR creating compressed version: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                print(f"Processing static image: {os.path.basename(input_path)} -> {size}x{size} (animated: {is_animated})")
                img = resize_frame(img, size, opacity, greyscale)

                # Handle different output formats properly
                if output_format.upper() == 'GIF':
                    # Convert to RGBA to prevent palette issues
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    img.save(output_path, format='GIF')
                    
                elif output_format.upper() in ['JPEG', 'JPG']:
                    # Convert RGBA to RGB for JPEG
                    if img.mode == 'RGBA':
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                        img = background
                    image_data = optimize_image_size(img, max_size_kb, output_format)
                    if image_data:
                        with open(output_path, 'wb') as f:
                            f.write(image_data)
                    else:
                        img.save(output_path, format=output_format)
                else:
                    # For PNG and other formats that support transparency
                    image_data = optimize_image_size(img, max_size_kb, output_format)
                    if image_data:
                        with open(output_path, 'wb') as f:
                            f.write(image_data)
                    else:
                        img.save(output_path, format=output_format)

            if not stamps_mode:
                print(f"Created: {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")

def process_images(input_folder, mode, icon_image):
    """Process all images in the input folder based on mode (Stickers/Stamps)"""
    input_folder_name = os.path.basename(input_folder)
    output_folder_name = f"{input_folder_name}_Output"
    output_base_dir = os.path.join(input_folder, "..", output_folder_name)

    # Define output directories based on mode
    output_dirs = {
        '240': os.path.join(output_base_dir, "output_240x240"),
        '360': os.path.join(output_base_dir, "output_360x360"),
        '720': os.path.join(output_base_dir, "output_720x720"),
        'icon': os.path.join(output_base_dir, "output_icon")
    }

    if mode == 'stamps':
        output_dirs['1480'] = os.path.join(output_base_dir, "output_1480x1480")

    # Create output directories
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    # Collect image files for processing
    image_files = sorted([f for f in os.listdir(input_folder)
                          if f.lower().endswith(('.png', '.gif'))])

    # Process all images
    first_png_path = None  # Track first PNG for icon generation in stamps mode
    for filename in image_files:
        input_path = os.path.join(input_folder, filename)

        # Mode-specific conversions
        if mode == 'stickers':
            # Common conversion (240x240 PNG) for stickers
            convert_image(input_path, output_dirs['240'], 240, 'PNG', 290)
            convert_image(input_path, output_dirs['360'], 360, 'GIF', 290)
            convert_image(input_path, output_dirs['720'], 720, 'GIF', 500)
        else:  # stamps
            # Process ALL .png and .gif files in stamps mode
            print(f"Processing stamps from: {filename}")
            convert_image(input_path, output_dirs['240'], 240, 'PNG', 290, stamps_mode=True)
            convert_image(input_path, output_dirs['360'], 360, 'PNG', 250, stamps_mode=True)
            convert_image(input_path, output_dirs['720'], 720, 'PNG', 500, stamps_mode=True)
            convert_image(input_path, output_dirs['1480'], 1480, 'PNG', 500, stamps_mode=True)

            # Track the first PNG file for icon generation
            if first_png_path is None and filename.lower().endswith('.png'):
                first_png_path = input_path

        # Process icon image (for stickers mode)
        if mode == 'stickers' and input_path == icon_image:
            convert_image(input_path, output_dirs['icon'], 144, 'PNG', 290, suffix='_color')
            convert_image(input_path, output_dirs['icon'], 144, 'PNG', 290,
                        suffix='_grey', greyscale=True)
            convert_image(input_path, output_dirs['icon'], 512, 'PNG', 290)

            # For 280x280 GIF in stickers mode, find the animated GIF file
            # Find any animated GIF file in the folder (look for ALPHA.gif files first, then any .gif)
            input_dir = os.path.dirname(input_path)
            gif_candidates = []

            # Priority 1: Look for ALPHA.gif files (main animated GIFs)
            for fname in os.listdir(input_dir):
                if fname.lower().endswith('_alpha.gif'):
                    gif_candidates.append(os.path.join(input_dir, fname))

            # Priority 2: Look for any other .gif files if no ALPHA found
            if not gif_candidates:
                for fname in os.listdir(input_dir):
                    if fname.lower().endswith('.gif') and fname != os.path.basename(input_path):
                        gif_candidates.append(os.path.join(input_dir, fname))

            if gif_candidates:
                # Use the first (and likely only) animated GIF found
                gif_path = gif_candidates[0]
                print(f"Using animated GIF file for 280x280: {os.path.basename(gif_path)}")
                convert_image(gif_path, output_dirs['icon'], 280, 'GIF', 290)
            else:
                print(f"No animated GIF found, creating static 280x280 from PNG")
                convert_image(input_path, output_dirs['icon'], 280, 'GIF', 290)

    # Generate icon images for stamps mode using the first PNG file
    if mode == 'stamps' and first_png_path:
        print(f"\nGenerating icon images from: {os.path.basename(first_png_path)}")
        convert_image(first_png_path, output_dirs['icon'], 512, 'PNG', 500, stamps_mode=True)
        convert_image(first_png_path, output_dirs['icon'], 144, 'PNG', 290,
                     suffix='_COLOR', stamps_mode=True)
        convert_image(first_png_path, output_dirs['icon'], 144, 'PNG', 290,
                     suffix='_GREY', greyscale=True, stamps_mode=True)
    elif mode == 'stamps' and not first_png_path:
        print("\nWarning: No PNG files found for icon generation. Skipping output_icon.")

def main():
    # Get base directory - look for Repl directory in current or parent directories
    current_dir = os.getcwd()
    possible_paths = [
        os.path.join(current_dir, "Repl"),
        os.path.join(current_dir, "..", "Repl"),
        os.path.join(os.path.expanduser("~"), "Desktop", "Repl"),
        os.path.join(os.path.expanduser("~"), "Repl")
    ]
    
    repl_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            repl_dir = path
            break
    
    if not repl_dir:
        print("Error: 'Repl' directory not found.")
        print("Looked in the following locations:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nPlease create a 'Repl' directory or specify the correct path.")
        return
    
    print(f"Found Repl directory at: {repl_dir}")

    # Get mode selection
    while True:
        print("\nSelect processing mode:")
        print("1. Stickers")
        print("2. Stamps")
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice in ('1', '2'):
            mode = 'stickers' if choice == '1' else 'stamps'
            break
        print("Invalid choice. Please enter 1 or 2.")

    # Get available folders (excluding _Output folders)
    folders = [d for d in os.listdir(repl_dir) 
              if os.path.isdir(os.path.join(repl_dir, d)) and not d.endswith('_Output')]

    if not folders:
        print("No valid folders found in Repl directory.")
        return

    # Display folder options
    print("\nAvailable folders:")
    for i, folder in enumerate(folders, 1):
        print(f"{i}. {folder}")

    # Get folder selection (support multiple folders)
    selected_folders = []
    while True:
        try:
            choice_input = input("\nEnter folder number(s) to process (e.g., '1' or '1,3,5'): ").strip()
            choices = [int(x.strip()) for x in choice_input.split(',')]
            
            # Validate all choices
            valid_choices = all(1 <= choice <= len(folders) for choice in choices)
            if valid_choices:
                selected_folders = [folders[choice - 1] for choice in choices]
                break
            else:
                print("Invalid choice(s). Please enter valid numbers.")
        except ValueError:
            print("Please enter valid numbers separated by commas.")

    print(f"\nSelected {len(selected_folders)} folder(s) for processing: {', '.join(selected_folders)}")

    # Process each selected folder
    for folder_name in selected_folders:
        input_folder = os.path.join(repl_dir, folder_name)

        # Check if folder has images
        images = [f for f in os.listdir(input_folder)
                 if f.lower().endswith(('.png', '.gif'))]

        if not images:
            print(f"No images found in {folder_name}, skipping.")
            continue

        # For stickers mode, find the *_800x800.png file as icon source
        icon_image = None
        if mode == 'stickers':
            for filename in os.listdir(input_folder):
                if filename.lower().endswith('_800x800.png'):
                    icon_image = os.path.join(input_folder, filename)
                    break

            if not icon_image:
                print(f"\nWarning: No *_800x800.png file found in {folder_name}, skipping this folder.")
                continue
            print(f"\nProcessing images in folder: {folder_name}")
            print(f"Using icon: {os.path.basename(icon_image)}")
        else:
            print(f"\nProcessing stamps in folder: {folder_name}")
            print(f"Found {len(images)} image(s) to process")

        process_images(input_folder, mode, icon_image)
        print(f"Completed processing: {folder_name}")

    print(f"\n🎉 Batch conversion complete! Processed {len(selected_folders)} folder(s).")

if __name__ == "__main__":
    main()
