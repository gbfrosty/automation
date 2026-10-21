# Standard Libraries
import sys
import traceback
import random
import time
import math
import keyboard
import functools
import inspect
from collections import namedtuple

# Third Party Libraries
import cv2
import pydirectinput
import pyautogui
import pytweening
import pygetwindow as gw
import numpy as np
#import tesseract
from pynput import mouse
from PIL import Image

class Automate:
    def __init__(self, window_title="ABC", attempts=0, failure_limit=2, confidence=0.9, travel_duration=1.2, error_rate=0.03):
        self.paused = False
        self.window_title = window_title
        self.attempts = attempts
        self.failure_limit = failure_limit
        self.confidence = confidence
        self.travel_duration = travel_duration
        self.error_rate = error_rate
        
        # Typing Settings
        self.key_min, self.key_max = 0.05, 0.15
        self.adj_keys = {
            'q': 'wa', 'w': 'qes', 'e': 'wrsd', 'r': 'etdf', 't': 'ryfg', 'y': 'tugh',
            'a': 'qswz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc', 'g': 'ftyhbv',
            '1': '2q', '2': '13qw', '3': '24we', '4': '35er', '5': '46rt',
            '6': '57ty', '7': '68yu', '8': '79io', '9': '80op', '0': '9p'
        }

        self.confidence_config = {
            'default': 0.8,
            'very_high': 0.95
        }

        self.timing_config = {
            'nano': (0.01, 0.02),
            'micro': (0.05, 0.075),
            'consecutive': (0.09, 0.11),
            'click': (0.2, 0.3),
            'default': (0.4, 0.6),
            'medium_pause': (0.6, 0.75),
            'long_pause': (0.75, 1.0),
            'page_load': (2.0, 4.0),
            'safety': (8.0, 10.0)
        }

        # These have property getters
        self._reader = None
        self._regions = None
        self._window = None

        # Internal state
        self.success_count = 0
        self.consecutive_failures = 0
        
        pydirectinput.PAUSE = 0
        pyautogui.PAUSE = 0
        pydirectinput.FAILSAFE = True


        # Add a keypress handler to pause execution
        keyboard.add_hotkey('p', self._toggle_pause)

         # This stops PyAutoGUI from raising exceptions when an image is missing
        pyautogui.useImageNotFoundException(False)

    # --- Getters ---

    @property
    def window(self):
        if self._window is None:
             self._window = gw.getWindowsWithTitle(self.window_title)
        return self._window

    @property
    def reader(self):
        """Only imports and loads the model the first time it is accessed."""
        if self._reader is None:
          print("Loading OCR engine (this may take a few seconds)...")
          import easyocr  # Lazy import
          # gpu=False prevents the CUDA/MPS search delay
          self._reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        return self._reader

    @property
    def regions(self):
        if self._regions is None:
            if self.window is None:
                print('Cannot set regions, window is not defined')
                return None
            windowObj = self.window
            win = windowObj[0]
            w, h = win.width, win.height
            l, t = win.left, win.top

            # Use integer division // to ensure PyAutoGUI compatibility
            self._regions = {
              'window': (l, t, w, h),
              'top':    (l, t, w, h // 2),
              'right':  (l + int(w * 0.5), t, w // 2, h), # Changed height to full h
              'bottom': (l, t + int(h * 0.5), w, h // 2),
              'left':   (l, t, w // 2, h),
              'centre': (l + int(w * 0.5), t + int(h * 0.5), 1, 1)
            }

        return self._regions

    def get_confidence(self, key='default'):
        return self.confidence_config[key]

    def get_timing(self, key='default'):
        t1, t2 = self.timing_config[key] 
        return random.uniform(t1, t2)
    
    # --- Private Methods ---

    def _toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            print("\n[PAUSED] Execution halted. Press 'p' to resume...")
        else:
            print("\n[RESUMED] Continuing execution...")

    def _get_pixel_offset(self, lbound=-3, ubound=3):
        return random.randint(lbound, ubound)

    # --- Human-Like Behaviour --- #

    def _get_bezier_point(self, p0, p1, p2, t):
        # Standard quadratic Bezier formula
        return (1 - t)**2 * np.array(p0) + 2 * (1 - t) * t * np.array(p1) + t**2 * np.array(p2)

    def _human_curved_move(self, target_x, target_y, slowdown=None):
        if slowdown is None:
            slowdown = self.get_timing('micro')
      
        start_x, start_y = pyautogui.position()
        end_x, end_y = target_x, target_y

        dist = np.linalg.norm(np.array([end_x - start_x, end_y - start_y]))
        if dist < 10:
            return

        # Randomize control point for human arc, keeping start/end fixed
        offset = dist * 0.3
        mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
        control_x = mid_x + random.uniform(-offset, offset)
        control_y = mid_y + self.get_timing('micro')

        # Pre-calculate points to avoid loop-timing drift
        steps = int(120 * (self.travel_duration * slowdown))
        path = []
        for i in range(steps + 1):
            t = pytweening.easeInOutSine(i / steps)
            pos = self._get_bezier_point((start_x, start_y), (control_x, control_y), (end_x, end_y), t)
            path.append((int(pos[0]), int(pos[1])))

        # Execute movement
        for pos in path:
            pyautogui.moveTo(pos[0], pos[1])

    def _human_drag_rel(self, x_dist, y_dist, steps=10):
        pyautogui.mouseDown()
        time.sleep(self.get_timing('consecutive'))

        # Track "ideal" path to prevent remainder accumulation
        last_x, last_y = 0, 0

        for i in range(1, steps + 1):
            # Calculate absolute progress (prevents remainder drift)
            current_target_x = int((x_dist / steps) * i)
            current_target_y = int((y_dist / steps) * i)

            # Determine delta for this specific step
            delta_x = current_target_x - last_x
            delta_y = current_target_y - last_y

            if delta_x != 0 or delta_y != 0:
                pyautogui.moveRel(delta_x, delta_y)

            last_x, last_y = current_target_x, current_target_y
            time.sleep(self.get_timing('nano'))

        time.sleep(self.get_timing('nano'))
        pyautogui.mouseUp()

    def _jitter(self, lbound=None, ubound=None, chance=1):
        """ Simulates a tiny mouse move """
        if lbound is None and ubound is None:
            x_offset = self._get_pixel_offset()
            y_offset = self._get_pixel_offset()
        elif lbound is None:
            x_offset = self._get_pixel_offset(ubound=ubound)
            y_offset = self._get_pixel_offset(ubound=ubound)
        else:
            x_offset = self._get_pixel_offset(lbound=lbound)
            y_offset = self._get_pixel_offset(lbound=lbound)
        
        if random.random() < chance:
            pydirectinput.moveRel(x_offset, y_offset, relative=True, _pause=False)

    def _human_click(self):
        self._jitter()
        pydirectinput.mouseDown()
        start, hold = time.time(), self.get_timing('micro')
        while time.time() - start < hold:
            self._jitter(-1, 1, 0.2)
            time.sleep(self.get_timing('nano'))
        pydirectinput.mouseUp()

    def _human_double_click(self):
        self._jitter()
        pydirectinput.double_click()
        self._jitter(-2, 2, 0.4)

    def _human_typing(self, text, press_enter=False):
        for i in range(1, 5):
            pydirectinput.press('backspace')
            time.sleep(self.get_timing('click'))
        for char in str(text).lower():
            if char in self.adj_keys and random.random() < self.error_rate:
                wrong_char = random.choice(self.adj_keys[char])
                pydirectinput.write(wrong_char)
                time.sleep(self.get_timing('click'))
                pydirectinput.press('backspace')
                time.sleep(self.get_timing('click'))

            pydirectinput.write(char)
            time.sleep(random.uniform(self.key_min, self.key_max))
            if random.random() < 0.04:
                time.sleep(self.get_timing('medium_pause'))
        
        if press_enter:
            time.sleep(random.uniform(self.get_timing('click')))
            pydirectinput.press('enter')

    # Used by the client when needing to add an offset to a dimension
    def _parse_offset(self, val, ref_tuple, index):
        if isinstance(val, str):
            val = val.replace(" ", "")

            if ref_tuple:
                if '+' in val:
                    _, offset = val.split('+')
                    return ref_tuple[index] + int(offset)
                elif '-' in val:
                    _, offset = val.split('-')
                    return ref_tuple[index] - int(offset)
                else:
                    return ref_tuple[index]
            
            # Return as int if just a numerical string was passed
            return int(val)
        return val

    def _prepare_interaction(self, loc, area, conf, timeout, relative, optional, grayscale):
        """Internal Helper to find a location and move the mouse"""
        if timeout is None: timeout = self.get_timing('page_load')
        if conf is None: conf = self.get_confidence()
        target = None

        l, t, w, h = self.regions[area]
  
        self.check_pause()

        if isinstance(loc, str):
            target = self.wait(loc, area=area, timeout=timeout, 
                conf=conf, grayscale=grayscale)
        elif isinstance(loc, (tuple, list)):
            Point = namedtuple('Point', ['x', 'y'])
            if relative:
                target = Point(loc[0], loc[1])
            else:
                target = Point(loc[0] + l, loc[1] + t)
        if not target:
            if not optional:
                caller = inspect.stack()[1].function
                print(f"Problem with click on {loc} from function {caller}")
            return target
        self.window[0].activate(); 
        self._human_curved_move(target.x, target.y)
        time.sleep(self.get_timing('micro'))

        return target

    # --- Public API ---

    def add_region(self, name, dimensions, reference='window'):
        # Type check
        if not isinstance(dimensions, tuple):
            print(f"Error: Dimensions for '{name}' must be a tuple. Received: {type(dimensions)}")
            return

        # Length check (ensure it has x, y, w, l)
        if len(dimensions) != 4:
            print(f"Error: Dimensions for '{name}' must contain 4 values (x, y, w, l).")
            return

        l, t, w, h = dimensions
        ref_data = self.regions.get(reference)
        
        resolved_l = self._parse_offset(l, ref_data, 0) if isinstance(l, str) else l
        resolved_t = self._parse_offset(t, ref_data, 1) if isinstance(t, str) else t
        resolved_w = self._parse_offset(w, ref_data, 2) if isinstance(w, str) else w
        resolved_h = self._parse_offset(h, ref_data, 3) if isinstance(h, str) else h

        self._regions[name] = (resolved_l, resolved_t, resolved_w, 
            resolved_h)

    def get_text(self, img, detail=0):
        img_np = np.array(img)
        return self.reader.readtext(img_np, detail=0) 

    def check_pause(self):
        # Call this inside any loop to handle the pausing logic.
        while self.paused:
            time.sleep(self.get_timing('consecutive')) # Sleep prevents high CPU usage while waiting

    def click_listener(self, area='window'):
        l, t, w, h = self.regions[area]
        clicked_coords = {'x': 0, 'y': 0}

        print(f"Waiting for click in the {area} region...")

        def on_click(x, y, button, pressed):
            if pressed:
                # Calculate the offset
                clicked_coords['x'] = x - l + 78
                clicked_coords['y'] = y  - t
                # Stop listener
                return False 

        # Collect events until released
        with mouse.Listener(on_click=on_click) as listener:
            listener.join()

        return (clicked_coords['x'], clicked_coords['y'])

    def type_text(self, text, press_enter=True, focus_window=True):
        """Types text with realistic human errors and delays."""
        try:
            if focus_window:
                if self.window: self.window[0].activate(); time.sleep(self.get_timing())
                else: return False

            self._human_typing(text, press_enter)
            return True
        except Exception: traceback.print_exc(); return False

    def click(self, loc, area="window", conf=None, pause=None, 
        timeout=None, relative=True, optional=False, grayscale=True):

        if pause is None: pause = self.get_timing()

        target = self._prepare_interaction(loc, area, conf, timeout, relative, optional, grayscale)
        
        if target:
            self._human_click()
            time.sleep(pause)
        return target

    def double_click(self, loc, area="window", conf=None, 
        pause=None, timeout=None, relative=True, optional=False, grayscale=True):

        if pause is None: pause = self.get_timing()

        target = self._prepare_interaction(loc, area, conf, timeout, relative, optional, grayscale)
        
        if target:
            self._human_double_click()
            time.sleep(pause)
        return target
    
    def hold_until(self, loc=None, target_image=None, area="window", conf=None, 
        pause=None, timeout=None, relative=True, optional=False, grayscale=True):
        
        if target_image is None:
            print("Hold until function needs a target image to work with, exiting ...")
            return None

        if self.find(target_image, area, conf, grayscale):
            # If the target image is found, return early
            return

        if pause is None: pause = self.get_timing()
        if timeout is None: timeout = self.get_timing('safety')

        # Use the shared logic to move to the 'button'
        target = self._prepare_interaction(loc, area, conf, timeout, relative, optional, grayscale)
        if not target: return

        try:
            start_time = time.perf_counter()

            pydirectinput.mouseDown()
            # Loop until the new target_image appears or until timeout
            while not self.find(target_image, area, conf, grayscale):
                time.sleep(self.get_timing())
                if (time.perf_counter() - start_time) >= timeout: break

        finally:
            pydirectinput.mouseUp()
            time.sleep(pause)

    def hold_while(self, loc=None, target_image=None, area="window", conf=None, 
        pause=None, timeout=None, relative=True, optional=False, grayscale=True):
        
        if target_image is None:
            print("Hold until function needs a target image to work with, exiting ...")
            return None

        if not self.find(target_image, area, conf, grayscale):
            # If the target image is not found, exit early
            return

        if pause is None: pause = self.get_timing()
        if timeout is None: timeout = self.get_timing('safety')

        # Use the shared logic to move to the 'button'
        target = self._prepare_interaction(loc, area, conf, timeout, relative, optional, grayscale)
        if not target: return

        try:
            start_time = time.perf_counter()

            pydirectinput.mouseDown()
            # Loop until the new target_image appears or until timeout
            while self.find(target_image, area, conf, grayscale):
                time.sleep(self.get_timing())
                if (time.perf_counter() - start_time) >= timeout: break

        finally:
            pydirectinput.mouseUp()
            time.sleep(pause)

    def drag(self, x, y, startX=None, startY=None, slowdown=None, pause=None):
        if slowdown is None:
            slowdown = self.get_timing('micro')
        if pause is None:
            pause = self.get_timing('medium_pause')

        # Determine start position
        # Only use random jitter if no specific startX/startY is provided
        if startX is None or startY is None:
            l, t, w, h = self.regions['window']
            startX = l + (w // 2) + random.randint(-2, 2)
            startY = t + (h // 2) + random.randint(-2, 2)
        else:
            # If explicit start coordinates are provided, apply minimal jitter to Y 
            # only if desired, otherwise keep them strict for viewport navigation
            startY += random.randint(-2, 2)

        self._human_curved_move(startX, startY, slowdown=slowdown)

        time.sleep(self.get_timing('micro'))
        self._human_drag_rel(x, y)
        time.sleep(pause)

    def moveTo(self, loc, area="window"):
        l, t, w, h = self.regions[area]
        pyautogui.moveTo(loc[0] + l, loc[1] + t)

    def find(self, image_path, area="window", conf=None, grayscale=True):
        if conf is None: conf = self.get_confidence()
        return pyautogui.locateCenterOnScreen(image_path, region=self.regions[area], confidence=conf, grayscale=grayscale)

    def findAll(self, image_path, area="window", min_dist=10, conf=None, grayscale=True):
        if conf is None: conf = self.get_confidence()
        matches = pyautogui.locateAllOnScreen(image_path, region=self.regions[area], confidence=conf, grayscale=grayscale)
        
        locations = []

        for box in matches:
            self.check_pause()
            is_duplicate = False
            centre = pyautogui.center(box)
            for loc in locations:
                dist = ((centre.x - loc.x)**2 + (centre.y - loc.y)**2)**0.5
                if dist < min_dist:
                    is_duplicate = True
                    break

            if not is_duplicate:
                locations.append(pyautogui.center(box))

        return locations

    def findWithMask(self, target_path, area='window', conf=None, min_dist=20):
        if conf is None: conf = self.get_confidence('very_high')
        l, t, w, h = self.regions[area]

        source_img = cv2.imread(target_path, cv2.IMREAD_UNCHANGED)
        template_bgr = source_img[:, :, 0:3]
        mask = source_img[:, :, 3]
        t_h, t_w = template_bgr.shape[:2]

        screen_pil = pyautogui.screenshot(region=(l, t, w, h))
        screen_bgr = cv2.cvtColor(np.array(screen_pil), cv2.COLOR_RGB2BGR)

        # Match Template returns a 2D array of confidence scores
        result = cv2.matchTemplate(screen_bgr, template_bgr, cv2.TM_CCORR_NORMED, mask=mask)

        # Get all coordinates that pass the threshold
        locs = np.where(result >= conf)

        # Create a list of (x, y, score)
        candidates = []
        for pt in zip(*locs[::-1]):
            candidates.append({
              'x': pt[0],
              'y': pt[1],
              'score': result[pt[1], pt[0]]
            })

        # Sort candidates by score (highest first)
        candidates.sort(key=lambda x: x['score'], reverse=True)

        unique_matches = []
        for cand in candidates:
            self.check_pause()
            # Check if this candidate is too close to any already-accepted match
            is_duplicate = False
            for accepted in unique_matches:
                dist = math.hypot(cand['x'] - accepted['loc'][0], cand['y'] - accepted['loc'][1])
                if dist < min_dist:
                    is_duplicate = True
                    break

            if not is_duplicate:
              # Crop the image once here
                crop = screen_bgr[cand['y']:cand['y']+t_h, cand['x']:cand['x']+t_w]
                unique_matches.append({
                    'loc': (cand['x'], cand['y']),
                    'crop': crop,
                    'score': cand['score']
                })

        return unique_matches


    def getText(self, crop):
        """Now accepts an image crop directly, avoiding the screenshot error."""
        results = self.reader.readtext(crop, detail=0)
        return results[0] if results else ""


    def exists(self, image_path, area="window", conf=None, grayscale=True):
        if conf is None: conf = self.get_confidence()
        return self.find(image_path, area=area, conf=conf, grayscale=grayscale)

    def move(self, loc):
        pyautogui.moveRel()

    def wait(self, image_path, area="window", timeout=None, 
        conf=None, grayscale=True):
        if timeout is None: timeout = self.get_timing('safety')
        if conf is None: conf = self.get_confidence()

        start_time = time.perf_counter()
  
        while (time.perf_counter() - start_time) < timeout:
            self.check_pause()
            loc = self.find(image_path, area, conf)
            if loc:
                return loc
            time.sleep(self.get_timing('consecutive'))
        return None

    def is_finished(self):
        """Returns True if the bot has reached success or failure limits."""
        return (self.attempts > 0 and self.success_count >= self.attempts) or (self.consecutive_failures >= self.failure_limit)

    def windowSize(self):
        print(self.regions['window'])

    def screenshot(self, box, output_path=None, area='window'):
        rl, rt, rw, rh = self.regions[area]
        l, t, w, h = box
        region = (
            int(l + rl),
            int(t + rt),
            int(w),
            int(h)
        )
        screenshot = pyautogui.screenshot(region=region)
        if output_path:
            new_img = Image.new('RGB', (w, h))
            new_img.paste(screenshot, (0, 0))
            self.save_image(new_img, output_path)
        return screenshot

    def combine_screenshots(self, images, output_path=None,
        orientation="vertical"):
        max_w = 0
        max_h = 0
        total_w = 0
        total_h = 0
        
        for img in images:
            total_w += img.width
            total_h += img.height
            max_w = max(max_w, img.width)
            max_h = max(max_h, img.height)

        
        if orientation == "horizontal":
            canvas_size = (total_w, max_h)
        else: 
            canvas_size = (max_w, total_h)

        new_img = Image.new('RGB', canvas_size)

        offset = 0

        for img in images:
            if orientation == "horizontal":
                new_img.paste(img, (offset, 0))
                offset += img.width
            else: 
                new_img.paste(img, (0, offset))
                offset += img.height
        
        if output_path:
            self.save_image(new_img, output_path)

        return new_img

    def crop_screenshot(self, img, region, output_path=None):
        crop = img.crop(region)
        if output_path:
            self.save_image(crop, output_path)
        return crop

    def save_image(self, img, output_path):
        img.save(output_path)