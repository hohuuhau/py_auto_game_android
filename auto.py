import uiautomator2 as u2
import cv2
import numpy as np
import time
from typing import Optional, Tuple, Dict, List
import os
import subprocess
import sys
from datetime import datetime
import logging

def setup_logging():
    """
    Setup logging configuration
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    # Setup logging configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/auto_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

def log_message(message: str, level: str = 'info'):
    """
    Log message with timestamp
    Args:
        message: Message to log
        level: Log level (info, warning, error, debug)
    """
    if level == 'info':
        logging.info(message)
    elif level == 'warning':
        logging.warning(message)
    elif level == 'error':
        logging.error(message)
    elif level == 'debug':
        logging.debug(message)

def print_statistics(stats: Dict[str, int]) -> None:
    """
    Print formatted statistics
    Args:
        stats: Dictionary containing statistics
    """
    stats_info = f"""
{'='*50}
FINAL STATISTICS
{'='*50}
Total matches: {stats['total']}
Wins: {stats['win']}
Losses: {stats['lose']}
"""
    if stats['total'] > 0:
        win_rate = round(stats['win'] / stats['total'] * 100, 2)
        stats_info += f"Win rate: {win_rate}%\n"
    stats_info += f"{'='*50}"
    
    log_message(stats_info)
    
    # Save statistics to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"stats_{timestamp}.txt"
    try:
        with open(filename, 'w') as f:
            f.write(stats_info)
        log_message(f"Statistics saved to {filename}")
    except Exception as e:
        log_message(f"Error saving statistics: {e}", 'error')

def cleanup(device: Optional[u2.Device] = None) -> None:
    """
    Cleanup resources before exit
    Args:
        device: Connected device (optional)
    """
    try:
        # Clean up temporary files
        if os.path.exists("src/screen.png"):
            os.remove("src/screen.png")
    except Exception as e:
        log_message(f"Error during cleanup: {e}", 'error')

def get_connected_devices() -> List[str]:
    """
    Get list of connected devices
    Returns:
        List of device IDs
    """
    try:
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
        devices = []
        for line in result.stdout.split('\n')[1:]:  # Skip first line
            if line.strip() and 'device' in line:
                device_id = line.split()[0]
                devices.append(device_id)
        return devices
    except Exception as e:
        log_message(f"Error getting device list: {e}", 'error')
        return []

def select_device() -> Optional[str]:
    """
    Allow user to select device from list
    Returns:
        Selected device ID or None if no devices
    """
    try:
        devices = get_connected_devices()
        
        if not devices:
            log_message("No devices found, retrying...")
            time.sleep(1)
            return select_device()
            
        print("\nConnected devices:")
        for i, device in enumerate(devices, 1):
            print(f"{i}. {device}")
            
        while True:
            try:
                choice = input("\nSelect device number (or press Enter for first device): ").strip()
                if not choice:  # If user presses Enter
                    return devices[0]
                    
                index = int(choice) - 1
                if 0 <= index < len(devices):
                    return devices[index]
                else:
                    log_message("Invalid selection!")
            except ValueError:
                log_message("Please enter a number!")
    except KeyboardInterrupt:
        log_message("Device selection cancelled by user")
        return None

def resize_template(template: np.ndarray, scale: float) -> np.ndarray:
    """
    Resize template by scale factor
    Args:
        template: Template to resize
        scale: Scale factor
    Returns:
        Resized template
    """
    width = int(template.shape[1] * scale)
    height = int(template.shape[0] * scale)
    return cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)

def get_screen_scale(screen: np.ndarray, reference_width: int = 1920) -> float:
    """
    Calculate scale ratio based on screen size
    Args:
        screen: Screen image
        reference_width: Reference width (default 1920px)
    Returns:
        Scale ratio
    """
    screen_width = screen.shape[1]
    return screen_width / reference_width

def debug_save_image(image: np.ndarray, name: str) -> None:
    """
    Save image for debugging purposes
    Args:
        image: Image to save
        name: Name of the image
    """
    debug_dir = "debug"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(debug_dir, f"{name}_{timestamp}.png")
    cv2.imwrite(filename, image)
    log_message(f"Debug image saved: {filename}", 'debug')

def find_template_position(screen: np.ndarray, template: np.ndarray, threshold: float = 0.8, y_offset: int = 0, debug: bool = False) -> Optional[Tuple[int, int]]:
    """
    Find template position in screen image using multiple methods
    Args:
        screen: Screen image
        template: Template to find
        threshold: Similarity threshold
        y_offset: Y-axis offset
        debug: Enable debug mode
    Returns:
        Tuple (x, y) of center position or None if not found
    """
    # Convert images to grayscale
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Try different template matching methods
    methods = [
        ('TM_CCOEFF_NORMED', cv2.TM_CCOEFF_NORMED),
        ('TM_CCORR_NORMED', cv2.TM_CCORR_NORMED),
        ('TM_SQDIFF_NORMED', cv2.TM_SQDIFF_NORMED)
    ]
    
    best_match = None
    best_val = -1
    best_method = None
    
    for method_name, method in methods:
        res = cv2.matchTemplate(screen_gray, template_gray, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # For TM_SQDIFF_NORMED, the best match is the minimum value
        if method == cv2.TM_SQDIFF_NORMED:
            match_val = 1 - min_val  # Convert to similarity score
            match_loc = min_loc
        else:
            match_val = max_val
            match_loc = max_loc
            
        if match_val > best_val:
            best_val = match_val
            best_match = match_loc
            best_method = method_name
    
    if debug:
        debug_info = f"""
Debug information:
Screen size: {screen.shape}
Template size: {template.shape}
Best match method: {best_method}
Best match value: {best_val}
Threshold: {threshold}
"""
        log_message(debug_info, 'debug')
        
        # Save debug images
        debug_save_image(screen, "screen")
        debug_save_image(template, "template")
        
        # Draw rectangle on match
        if best_val >= threshold:
            h, w = template_gray.shape
            debug_image = screen.copy()
            cv2.rectangle(debug_image, best_match, (best_match[0] + w, best_match[1] + h), (0, 255, 0), 2)
            debug_save_image(debug_image, "match")
    
    if best_val < threshold:
        if debug:
            log_message("No match found above threshold", 'debug')
        return None
        
    h, w = template_gray.shape
    center_x = best_match[0] + w // 2
    center_y = best_match[1] + h // 2 + int(y_offset)
    
    if debug:
        log_message(f"Match position: ({center_x}, {center_y})", 'debug')
    
    return (center_x, center_y)

def load_templates(template_dir: str = 'src') -> Dict[str, np.ndarray]:
    """
    Load all templates from directory
    Args:
        template_dir: Template directory path
    Returns:
        Dictionary containing templates
    """
    templates = {}
    for filename in os.listdir(template_dir):
        if filename.endswith('.png'):
            name = os.path.splitext(filename)[0]
            path = os.path.join(template_dir, filename)
            templates[name] = cv2.imread(path)
    return templates

def main():
    # Setup logging
    log_file = setup_logging()
    log_message(f"Log file: {log_file}")
    
    # Initialize statistics
    stats = {'total': 0, 'win': 0, 'lose': 0}
    
    try:
        # Select device
        device_id = select_device()
        if not device_id:
            log_message("Device selection cancelled")
            cleanup()
            sys.exit(0)
            
        log_message(f"Connecting to device: {device_id}")
        d = u2.connect(device_id)
        
        # Load templates
        templates = load_templates()
        
        log_message("Press Ctrl+C to stop and view statistics")
        log_message("Press 'd' to toggle debug mode")
        debug_mode = False
        
        while True:
            # Take screenshot
            d.screenshot("src/screen.png")
            screen = cv2.imread('src/screen.png')
            
            # Check for debug toggle
            if cv2.waitKey(1) & 0xFF == ord('d'):
                debug_mode = not debug_mode
                log_message(f"Debug mode: {'enabled' if debug_mode else 'disabled'}")
            
            # Close
            if pos := find_template_position(screen, templates['close'], debug=debug_mode):
                log_message("Found close")
                d.click(*pos)
                time.sleep(1)
                continue
            
            # Skip
            if pos := find_template_position(screen, templates['skip'], debug=debug_mode):
                log_message("Found skip")
                d.click(*pos)
                time.sleep(1)
                continue
            
            # Check templates
            if pos := find_template_position(screen, templates['challenge'], debug=debug_mode):
                log_message("Found challenge")
                stats['total'] += 1
                d.click(*pos)
                time.sleep(1)
                continue
            
            # Win
            if pos := find_template_position(screen, templates['win'], y_offset=1200, debug=debug_mode):
                log_message("Found win")
                stats['win'] += 1
                d.click(*pos)
                time.sleep(1)
                continue
            
            # Lose
            if pos := find_template_position(screen, templates['lose'], y_offset=1200, debug=debug_mode):
                log_message("Found lose")
                stats['lose'] += 1
                d.click(*pos)
                time.sleep(1)
                continue
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        log_message("Stopping...")
        print_statistics(stats)
        cleanup(d if 'd' in locals() else None)
        sys.exit(0)
    except Exception as e:
        log_message(f"Error: {e}", 'error')
        print_statistics(stats)
        cleanup(d if 'd' in locals() else None)
        sys.exit(1)

if __name__ == "__main__":
    main()
    