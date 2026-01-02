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
    
    # Convert to RGBA to ensure transparency is preserved
    if frame.mode != 'RGBA':
        frame = frame.convert('RGBA')
    
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
    """Save GIF and optimize with gifsicle for better quality compression"""
    import subprocess
    import shutil
    
    # Extract durations from original frames
    durations = [info.get('duration', 100) for info in frame_info]
    
    print(f"Saving {len(frames)} frames with gifsicle optimization")
    
    # Convert frames to palette mode with transparency
    p_frames = []
    for frame in frames:
        if frame.mode == 'RGBA':
            # Get alpha channel before conversion
            alpha = frame.split()[3]
            
            # Convert to palette without dithering
            frame_p = frame.convert('P', palette=Image.ADAPTIVE, colors=255, dither=Image.Dither.NONE)
            
            # Create mask where alpha == 0 (transparent pixels)
            mask = Image.eval(alpha, lambda a: 255 if a == 0 else 0)
            
            # Paste transparency index where pixels are transparent
            frame_p.paste(255, mask)
            
            p_frames.append(frame_p)
        elif frame.mode == 'P':
            p_frames.append(frame)
        else:
            frame_p = frame.convert('P', palette=Image.ADAPTIVE, colors=255, dither=Image.Dither.NONE)
            p_frames.append(frame_p)
    
    # Save initial GIF
    temp_unoptimized = output_path.replace('.gif', '_temp_unopt.gif')
    
    p_frames[0].save(
        temp_unoptimized,
        save_all=True,
        append_images=p_frames[1:],
        duration=durations,
        disposal=2,
        loop=0,
        transparency=255
    )
    
    # Check if gifsicle is available
    gifsicle_path = shutil.which('gifsicle')
    
    if gifsicle_path:
        # Try progressively more aggressive lossy compression
        lossy_levels = [30, 60, 80, 100, 120, 150, 200]
        
        for lossy in lossy_levels:
            try:
                # Run gifsicle with lossy compression
                result = subprocess.run([
                    'gifsicle',
                    '--optimize=3',
                    '--careful',
                    f'--lossy={lossy}',
                    '--colors=256',
                    temp_unoptimized,
                    '-o', output_path
                ], capture_output=True, text=True)
                
                if result.returncode == 0 and os.path.exists(output_path):
                    file_size_kb = os.path.getsize(output_path) / 1024
                    print(f"  gifsicle lossy={lossy}: {file_size_kb:.1f}KB")
                    
                    if file_size_kb <= max_size_kb:
                        print(f"✓ SUCCESS with gifsicle lossy={lossy}: {file_size_kb:.1f}KB")
                        # Clean up temp file
                        if os.path.exists(temp_unoptimized):
                            os.remove(temp_unoptimized)
                        return
                else:
                    print(f"  gifsicle error: {result.stderr}")
                    
            except Exception as e:
                print(f"  gifsicle exception: {e}")
        
        # If all lossy levels failed to meet target, use the last result
        if os.path.exists(output_path):
            file_size_kb = os.path.getsize(output_path) / 1024
            print(f"⚠ Best gifsicle result: {file_size_kb:.1f}KB (target was {max_size_kb}KB)")
        else:
            # Fallback to unoptimized
            shutil.copy(temp_unoptimized, output_path)
            print(f"⚠ Falling back to unoptimized GIF")
    else:
        # No gifsicle, just use the unoptimized version
        shutil.copy(temp_unoptimized, output_path)
        print(f"⚠ gifsicle not available, using Pillow output")
    
    # Clean up temp file
    if os.path.exists(temp_unoptimized):
        os.remove(temp_unoptimized)

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
    """Create a compressed GIF using gifsicle for better quality"""
    import subprocess
    import shutil
    
    print(f"Creating compressed version targeting {target_size_kb}KB...")
    print(f"Input: {len(frames)} frames")
    
    # Convert frames to palette mode with transparency
    p_frames = []
    for frame in frames:
        if frame.mode == 'RGBA':
            # Get alpha channel before conversion
            alpha = frame.split()[3]
            
            # Convert to palette without dithering
            frame_p = frame.convert('P', palette=Image.ADAPTIVE, colors=255, dither=Image.Dither.NONE)
            
            # Create mask where alpha == 0 (transparent pixels)
            mask = Image.eval(alpha, lambda a: 255 if a == 0 else 0)
            
            # Paste transparency index where pixels are transparent
            frame_p.paste(255, mask)
            
            p_frames.append(frame_p)
        elif frame.mode == 'P':
            p_frames.append(frame)
        else:
            frame_p = frame.convert('P', palette=Image.ADAPTIVE, colors=255, dither=Image.Dither.NONE)
            p_frames.append(frame_p)
    
    # Get durations
    durations = [info.get('duration', 100) for info in frame_info]
    
    # Save initial GIF with all frames
    temp_path = output_path.replace('.gif', '_temp.gif')
    
    p_frames[0].save(
        temp_path,
        save_all=True,
        append_images=p_frames[1:],
        duration=durations,
        disposal=2,
        loop=0,
        transparency=255
    )
    
    # Check if gifsicle is available
    gifsicle_path = shutil.which('gifsicle')
    
    if gifsicle_path:
        # Try progressively more aggressive lossy compression
        lossy_levels = [30, 60, 80, 100, 120, 150, 200]
        
        for lossy in lossy_levels:
            try:
                result = subprocess.run([
                    'gifsicle',
                    '--optimize=3',
                    '--careful',
                    f'--lossy={lossy}',
                    '--colors=256',
                    temp_path,
                    '-o', output_path
                ], capture_output=True, text=True)
                
                if result.returncode == 0 and os.path.exists(output_path):
                    file_size_kb = os.path.getsize(output_path) / 1024
                    print(f"  gifsicle lossy={lossy}: {file_size_kb:.1f}KB")
                    
                    if file_size_kb <= target_size_kb:
                        print(f"✓ SUCCESS with gifsicle lossy={lossy}: {file_size_kb:.1f}KB")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        return
                        
            except Exception as e:
                print(f"  gifsicle exception: {e}")
        
        # If lossy compression alone isn't enough, try frame reduction + lossy
        if os.path.exists(output_path):
            file_size_kb = os.path.getsize(output_path) / 1024
            if file_size_kb > target_size_kb:
                print(f"  Lossy compression not enough, trying frame reduction...")
                
                # Try reducing frames
                frame_counts = [len(frames) // 2, len(frames) // 3, len(frames) // 4]
                
                for target_frames in frame_counts:
                    if target_frames < 6:
                        target_frames = 6
                    
                    # Evenly space frames
                    step = len(p_frames) / target_frames
                    reduced_indices = [int(i * step) for i in range(target_frames)]
                    reduced_frames = [p_frames[i] for i in reduced_indices]
                    
                    # Adjust timing
                    original_duration = durations[0] if durations else 100
                    new_duration = int(original_duration * len(frames) / target_frames)
                    new_duration = min(new_duration, 150)  # Cap at 150ms
                    
                    # Save reduced version
                    temp_reduced = output_path.replace('.gif', '_temp_reduced.gif')
                    reduced_frames[0].save(
                        temp_reduced,
                        save_all=True,
                        append_images=reduced_frames[1:],
                        duration=new_duration,
                        disposal=2,
                        loop=0,
                        transparency=255
                    )
                    
                    # Apply gifsicle
                    for lossy in [80, 120, 200]:
                        result = subprocess.run([
                            'gifsicle',
                            '--optimize=3',
                            '--careful',
                            f'--lossy={lossy}',
                            '--colors=256',
                            temp_reduced,
                            '-o', output_path
                        ], capture_output=True, text=True)
                        
                        if result.returncode == 0 and os.path.exists(output_path):
                            file_size_kb = os.path.getsize(output_path) / 1024
                            print(f"  {target_frames} frames + lossy={lossy}: {file_size_kb:.1f}KB")
                            
                            if file_size_kb <= target_size_kb:
                                print(f"✓ SUCCESS with {target_frames} frames + lossy={lossy}")
                                if os.path.exists(temp_reduced):
                                    os.remove(temp_reduced)
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                                return
                    
                    if os.path.exists(temp_reduced):
                        os.remove(temp_reduced)
        
        # Use best result we got
        if os.path.exists(output_path):
            file_size_kb = os.path.getsize(output_path) / 1024
            print(f"⚠ Best result: {file_size_kb:.1f}KB (target was {target_size_kb}KB)")
        else:
            shutil.copy(temp_path, output_path)
            
    else:
        # No gifsicle - fall back to just the Pillow version
        shutil.copy(temp_path, output_path)
        print(f"⚠ gifsicle not available")
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)

def convert_image(input_path, output_dir, size, output_format, max_size_kb, 
                 suffix='', opacity=1.0, greyscale=False, stamps_mode=False):
    """Convert a single image with size and format constraints"""
    try:
        with Image.open(input_path) as img:
            is_animated = getattr(img, "is_animated", False)
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            new_filename = f"{base_name}_{size}x{size}{suffix}.{output_format.lower()}"
            output_path = os.path.join(output_dir, new_filename)

            # For stamps mode, force PNG format and ignore animated GIFs
            if stamps_mode:
                print(f"Processing image for stamps: {os.path.basename(input_path)} -> {size}x{size} PNG")
                img = resize_frame(img, size, opacity, greyscale)
                
                # Use stamps-specific PNG optimization
                image_data = optimize_png_stamps(img, max_size_kb)
                if image_data:
                    with open(output_path, 'wb') as f:
                        f.write(image_data)
                    file_size = len(image_data) / 1024
                    print(f"✓ Stamps PNG created: {new_filename} ({file_size:.1f}KB)")
                else:
                    # Fallback to regular save
                    img.save(output_path, format='PNG')
                    print(f"✓ Stamps PNG created: {new_filename}")
            
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
                    # Convert to RGBA first
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    
                    # Flood fill white background from corners with transparency
                    from PIL import ImageDraw
                    
                    # Get image data as pixels
                    pixels = img.load()
                    width, height = img.size
                    
                    # Define white (with tolerance)
                    def is_white(pixel, tolerance=30):
                        if len(pixel) >= 3:
                            return all(c >= 255 - tolerance for c in pixel[:3])
                        return False
                    
                    # Flood fill from corners using a queue-based approach
                    from collections import deque
                    visited = set()
                    to_make_transparent = set()
                    
                    # Start from all four corners
                    corners = [(0, 0), (width-1, 0), (0, height-1), (width-1, height-1)]
                    queue = deque()
                    
                    for corner in corners:
                        if is_white(pixels[corner]):
                            queue.append(corner)
                            visited.add(corner)
                    
                    # Flood fill
                    while queue:
                        x, y = queue.popleft()
                        to_make_transparent.add((x, y))
                        
                        # Check neighbors (4-connected)
                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                if (nx, ny) not in visited:
                                    visited.add((nx, ny))
                                    if is_white(pixels[nx, ny]):
                                        queue.append((nx, ny))
                    
                    # Make background pixels transparent
                    for x, y in to_make_transparent:
                        r, g, b, a = pixels[x, y]
                        pixels[x, y] = (r, g, b, 0)
                    
                    # Now convert to palette mode with transparency
                    # Get the alpha channel
                    alpha = img.split()[3]
                    
                    # Convert to palette mode
                    img_p = img.convert('P', palette=Image.ADAPTIVE, colors=255)
                    
                    # Create a mask where alpha == 0 (transparent pixels)
                    mask = Image.eval(alpha, lambda a: 255 if a == 0 else 0)
                    
                    # Paste transparency index (255) where pixels are transparent
                    img_p.paste(255, mask)
                    
                    # Save with transparency
                    img_p.save(output_path, format='GIF', transparency=255)
                    
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

    # Process all images
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.gif')):
            input_path = os.path.join(input_folder, filename)

            # Mode-specific conversions
            if mode == 'stickers':
                # Common conversion (240x240 PNG) for stickers
                convert_image(input_path, output_dirs['240'], 240, 'PNG', 290)
                convert_image(input_path, output_dirs['360'], 360, 'GIF', 290)
                convert_image(input_path, output_dirs['720'], 720, 'GIF', 500)
            else:  # stamps
                # Only process the 800x800 PNG file for stamps mode
                if filename.lower().endswith('_800x800.png'):
                    print(f"Processing stamps from: {filename}")
                    convert_image(input_path, output_dirs['360'], 360, 'PNG', 250, stamps_mode=True)
                    convert_image(input_path, output_dirs['720'], 720, 'PNG', 500, stamps_mode=True)
                    convert_image(input_path, output_dirs['1480'], 1480, 'PNG', 500, stamps_mode=True)
                else:
                    print(f"Skipping {filename} - stamps mode only processes *_800x800.png files")

            # Process icon image (only for stickers mode)
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
                for filename in os.listdir(input_dir):
                    if filename.lower().endswith('_alpha.gif'):
                        gif_candidates.append(os.path.join(input_dir, filename))
                
                # Priority 2: Look for any other .gif files if no ALPHA found
                if not gif_candidates:
                    for filename in os.listdir(input_dir):
                        if filename.lower().endswith('.gif') and filename != os.path.basename(input_path):
                            gif_candidates.append(os.path.join(input_dir, filename))
                
                if gif_candidates:
                    # Use the first (and likely only) animated GIF found
                    gif_path = gif_candidates[0]
                    print(f"Using animated GIF file for 280x280: {os.path.basename(gif_path)}")
                    convert_image(gif_path, output_dirs['icon'], 280, 'GIF', 290)
                else:
                    print(f"No animated GIF found, creating static 280x280 from PNG")
                    convert_image(input_path, output_dirs['icon'], 280, 'GIF', 290)

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
        
        # Automatically find the *_800x800.png file as icon
        icon_image = None
        for filename in os.listdir(input_folder):
            if filename.lower().endswith('_800x800.png'):
                icon_image = os.path.join(input_folder, filename)
                break
        
        if not icon_image:
            print(f"\nWarning: No *_800x800.png file found in {folder_name}, skipping this folder.")
            continue
        
        print(f"\nProcessing images in folder: {folder_name}")
        print(f"Using icon: {os.path.basename(icon_image)}")
        
        # Check if folder has images
        images = [f for f in os.listdir(input_folder) 
                 if f.lower().endswith(('.png', '.gif'))]
        
        if not images:
            print(f"No images found in {folder_name}, skipping.")
            continue
            
        process_images(input_folder, mode, icon_image)
        print(f"✓ Completed processing: {folder_name}")

    print(f"\n🎉 Batch conversion complete! Processed {len(selected_folders)} folder(s).")

if __name__ == "__main__":
    main()
