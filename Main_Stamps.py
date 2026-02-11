import os
import io
from PIL import Image, ImageSequence

def resize_frame(frame, size, opacity):
    """Resize, center, and pad to square with transparent background"""
    # Use LANCZOS for better quality when downsizing
    resampling = Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS
    
    # Convert to RGBA first to ensure proper alpha handling
    frame = frame.convert('RGBA')
    
    # Preserve aspect ratio - fit within size x size bounds
    frame.thumbnail((size, size), resampling)
    
    # Create transparent square canvas
    square_frame = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    
    # Center the resized frame on the transparent canvas
    frame_width, frame_height = frame.size
    x_offset = (size - frame_width) // 2
    y_offset = (size - frame_height) // 2
    
    # Ensure offsets are not negative
    x_offset = max(0, x_offset)
    y_offset = max(0, y_offset)
    
    # Paste with alpha compositing
    if frame.mode == 'RGBA':
        square_frame.paste(frame, (x_offset, y_offset), frame)
    else:
        square_frame.paste(frame, (x_offset, y_offset))
    
    if opacity < 1.0:
        alpha = square_frame.split()[3]
        alpha = alpha.point(lambda p: int(p * opacity))
        square_frame.putalpha(alpha)
    
    return square_frame

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

def process_animated_gif(img, output_path, size):
    frames = []
    durations = []

    for frame in ImageSequence.Iterator(img):
        duration = frame.info.get('duration', 100)
        frame = resize_frame(frame, size, 1.0)
        
        # Convert frame to palette mode and maintain transparency
        if frame.mode == 'RGBA':
            # Use simpler approach: convert RGBA to P with transparency
            frame_p = frame.convert('P', palette=Image.ADAPTIVE, colors=255)
            frame_p.info['transparency'] = 255
        else:
            # If no alpha channel, convert directly
            frame_p = frame.convert('P', colors=255)

        frames.append(frame_p)
        durations.append(duration)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        format='GIF',
        optimize=True,
        duration=durations,
        loop=0,
        transparency=255,
        disposal=2  # Restore to background (transparent)
    )

def convert_image(input_path, output_dir, size, output_format, suffix='', opacity=1.0):
    try:
        with Image.open(input_path) as img:
            is_animated = getattr(img, "is_animated", False)
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            new_filename = f"{base_name}_{size}x{size}{suffix}.{output_format.lower()}"
            output_path = os.path.join(output_dir, new_filename)

            if output_format.lower() == 'gif' and is_animated:
                process_animated_gif(img, output_path, size)
            elif output_format.lower() == 'gif':
                # Static image to GIF with transparency
                img = resize_frame(img, size, opacity)
                
                if img.mode == 'RGBA':
                    # Use simpler approach for static images
                    img_p = img.convert('P', palette=Image.ADAPTIVE, colors=255)
                    img_p.info['transparency'] = 255
                    img_p.save(output_path, format='GIF', optimize=True, transparency=255)
                else:
                    # No transparency, convert directly
                    img_p = img.convert('P', colors=255)
                    img_p.save(output_path, format='GIF', optimize=True)
            else:
                # PNG or other formats
                img = resize_frame(img, size, opacity)
                img.save(output_path, format=output_format.upper(), optimize=True)

            print(f"Created: {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")

def process_images(input_folder):
    input_folder_name = os.path.basename(input_folder)
    output_folder_name = f"{input_folder_name}_Output"
    output_base_dir = os.path.join("/home/runner/Image-Processing-using-Pillow/Repl", output_folder_name)

    output_dirs = {
        240: os.path.join(output_base_dir, "output_240x240_PNG"),
        360: os.path.join(output_base_dir, "output_360x360_GIF"),
        720: os.path.join(output_base_dir, "output_720x720_GIF"),
        1480: os.path.join(output_base_dir, "output_1480x1480_GIF"),
        'misc': os.path.join(output_base_dir, "output_misc")
    }

    for dir_name in output_dirs.values():
        os.makedirs(dir_name, exist_ok=True)
        print(f"Created output directory: {dir_name}")

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.gif')):
            input_path = os.path.join(input_folder, filename)

            # Standard conversions for all files
            convert_image(input_path, output_dirs[240], 240, 'PNG')
            convert_image(input_path, output_dirs[360], 360, 'GIF')
            convert_image(input_path, output_dirs[720], 720, 'GIF')
            convert_image(input_path, output_dirs[1480], 1480, 'GIF')

            # Special handling for files starting with "2THUMBZ_17_MISC_"
            if filename.startswith("2THUMBZ_10_LOGO_"):
                convert_image(input_path, output_dirs['misc'], 280, 'GIF')
                convert_image(input_path, output_dirs['misc'], 512, 'PNG')
                convert_image(input_path, output_dirs['misc'], 144, 'PNG', suffix='_On')
                convert_image(input_path, output_dirs['misc'], 144, 'PNG', suffix='_Off', opacity=0.7)

def main():
    repl_dir = "/home/runner/Image-Processing-using-Pillow/Repl"
    # List subfolders in the Repl directory
    subfolders = [d for d in os.listdir(repl_dir) if os.path.isdir(os.path.join(repl_dir, d))]

    if not subfolders:
        print("No subfolders found in the Repl directory.")
        return

    print("Stamps - Available folders in Repl directory:")
    for i, folder in enumerate(subfolders, 1):
        print(f"{i}. {folder}")

    while True:
        choice_input = input("\nEnter folder number(s) to process (e.g., '1', '1,3,5', or '58-61'), or 'q' to quit: ").strip()
        if choice_input.lower() == 'q':
            break

        try:
            # Parse input supporting both comma-separated values and ranges
            choices = []
            for part in choice_input.split(','):
                part = part.strip()
                if '-' in part:
                    range_parts = part.split('-')
                    if len(range_parts) == 2:
                        start = int(range_parts[0].strip())
                        end = int(range_parts[1].strip())
                        if start <= end:
                            choices.extend(range(start, end + 1))
                        else:
                            choices.extend(range(start, end - 1, -1))
                    else:
                        raise ValueError("Invalid range format")
                else:
                    choices.append(int(part))

            # Remove duplicates while preserving order
            seen = set()
            unique_choices = []
            for c in choices:
                if c not in seen:
                    seen.add(c)
                    unique_choices.append(c)
            choices = unique_choices

            # Validate all choices
            invalid = [c for c in choices if not (1 <= c <= len(subfolders))]
            if invalid:
                print(f"Invalid folder number(s): {invalid}. Please enter numbers between 1 and {len(subfolders)}.")
                continue

            selected_folders = [subfolders[c - 1] for c in choices]
            total = len(selected_folders)
            print(f"\nSelected {total} folder(s) for processing: {', '.join(selected_folders)}")

            completed = 0
            skipped = 0
            for i, selected_folder in enumerate(selected_folders, 1):
                folder_path = os.path.join(repl_dir, selected_folder)

                print(f"\n{'='*60}")
                print(f"  [{i} of {total}] Starting: {selected_folder}")
                print(f"{'='*60}")

                process_images(folder_path)

                completed += 1
                print(f"\n  DONE: {selected_folder} ({i} of {total} complete)")
                if i < total:
                    print(f"  Proceeding to next folder...")

            print(f"\n{'='*60}")
            print(f"  Batch processing complete!")
            print(f"  Processed: {completed} folder(s)")
            if skipped > 0:
                print(f"  Skipped: {skipped} folder(s)")
            print(f"{'='*60}")

        except ValueError:
            print("Please enter valid numbers separated by commas, or ranges like '58-61'.")

if __name__ == "__main__":
    main()
