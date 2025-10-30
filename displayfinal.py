import pygame
import sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

pygame.init()

WINDOWSIZEX = 640
WINDOWSIZEY = 480

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

BOUNDARYINT = 10
MODEL = load_model("digitrecognition.h5")

LABELS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"
}

FONT = pygame.font.Font("freesansbold.ttf", 24)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("Digit Recognition Model")

iswriting = False
number_xcord = []
number_ycord = []

DISPLAYSURF.fill(BLACK)

def preprocess_image(surface, x_min, x_max, y_min, y_max):
    """Crop, resize, normalize, and prepare image for prediction"""
    # Capture the region
    image_array = pygame.surfarray.array3d(surface)
    
    # Transpose to get correct orientation (pygame uses (width, height, channels))
    image_array = np.transpose(image_array, (1, 0, 2))
    
    # Crop the region
    cropped = image_array[y_min:y_max, x_min:x_max]
    
    # Convert to grayscale
    cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)

    # Resize to 28x28 (MNIST format)
    image = cv2.resize(cropped, (28, 28))

    # Normalize to [0, 1]
    image = image / 255.0
    
    # Reshape for model input (batch_size, height, width, channels)
    image = image.reshape(1, 28, 28, 1)

    return image


while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 8, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            if number_xcord and number_ycord:
                rect_min_x = max(min(number_xcord) - BOUNDARYINT, 0)
                rect_max_x = min(max(number_xcord) + BOUNDARYINT, WINDOWSIZEX)
                rect_min_y = max(min(number_ycord) - BOUNDARYINT, 0)
                rect_max_y = min(max(number_ycord) + BOUNDARYINT, WINDOWSIZEY)

                # Preprocess for model
                image = preprocess_image(DISPLAYSURF, rect_min_x, rect_max_x, rect_min_y, rect_max_y)
                
                # Predict with verbose=0 to suppress output
                prediction = MODEL.predict(image, verbose=0)
                label = LABELS[np.argmax(prediction)]

                # Display result
                textsurface = FONT.render(label, True, RED, WHITE)
                textrect = textsurface.get_rect()
                textrect.center = (rect_min_x + 60, rect_min_y - 20)
                DISPLAYSURF.blit(textsurface, textrect)

            number_xcord = []
            number_ycord = []

        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)

    pygame.display.update()