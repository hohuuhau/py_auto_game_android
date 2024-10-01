import uiautomator2 as u2
import cv2
import numpy as np
import pyautogui as pg
import time

#function to process the image and return x, y center of the best match position to click
#return none if not found

def click_on_best_match(screen, template, threshold=0.8):
    # Convert image to grayscale
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Find template in image
    res = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    
    # Get the best match position
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # Check if the best match exceeds the threshold
    if max_val < threshold:
        return None
    
    # Get the best match position
    top_left = max_loc
    h, w = template_gray.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    # Draw rectangle around the best match
    #cv2.rectangle(screen, top_left, bottom_right, 255, 2)
    
    # Draw the point center of the best match
    #cv2.circle(screen, (top_left[0] + w // 2, top_left[1] + h // 2), 5, (0, 0, 255), -1)
    
    # Save the result
    #cv2.imwrite('src/result.png', screen)
    
    # Click on the best match
    return (top_left[0] + w // 2, top_left[1] + h // 2)

def getPositionByAdded(screen, template, threshold=0.8):
    # Convert image to grayscale
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Find template in image
    res = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    
    # Get the best match position
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # Check if the best match exceeds the threshold
    if max_val < threshold:
        return None
    
    # Get the best match position
    top_left = max_loc
    h, w = template_gray.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    # Draw rectangle around the best match
    # cv2.rectangle(screen, top_left, bottom_right, 255, 2)
    
    # Draw the point center of the best match
    # cv2.circle(screen, (top_left[0] + w // 2, top_left[1] + h + 1200 // 2), 5, (0, 0, 255), -1)
    
    # Save the result
    # cv2.imwrite('src/win_result.png', screen)
    
    # Click on the best match
    return (top_left[0] + w // 2, top_left[1] + h + 1200 // 2)

def debug():
    d = u2.connect("emulator-5554") # connect to device
    d.screenshot("src/screen.png") # take screenshot
    screen = cv2.imread('src/screen.png')
    lose = cv2.imread('src/lose.png')
    res = getPositionByAdded(screen, lose, 0.8)
    if res is not None:
        print("Found lose")
        d.click(res[0], res[1])


d = u2.connect("emulator-5554") # connect to device
# Load image from file using numpy
boqua = cv2.imread('src/boqua.png')
khieuchien = cv2.imread('src/khieuchien.png')
win = cv2.imread('src/win.png')
lose = cv2.imread('src/lose.png')

while True:
    try:
        
        d.screenshot("src/screen.png") # take screenshot
        screen = cv2.imread('src/screen.png')
        res = click_on_best_match(screen, boqua, 0.8)
        
        if res is not None:
            print("Found bo qua")
            d.click(res[0], res[1])
            time.sleep(1)
            continue
        
        res = click_on_best_match(screen, khieuchien, 0.8)
        if res is not None:
            print("Found khieu chien")
            d.click(res[0], res[1])
            time.sleep(1)
            continue
        
        res = getPositionByAdded(screen, win, 0.8)
        if res is not None:
            print("Found win")
            d.click(res[0], res[1])
            time.sleep(1)
            continue
        
        res = getPositionByAdded(screen, lose, 0.8)
        if res is not None:
            print("Found lose")
            d.click(res[0], res[1])
            time.sleep(1)
            continue
        
        time.sleep(1)
        # print("Not found")
    except KeyboardInterrupt:
        print("Exit")
        break
    except Exception as e:
        print(e)
        time.sleep(1)
        continue
    