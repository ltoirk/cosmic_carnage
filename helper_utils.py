import pygame
def fill(surface, color):
    """Fill all pixels of the surface with color, preserve transparency."""
    w, h = surface.get_size()
    r, g, b, _ = color
    for x in range(w):
        for y in range(h):
            a = surface.get_at((x, y))[3]
            surface.set_at((x, y), pygame.Color(r, g, b, a))


import colorsys

def generate_rgb_values(n):
    """Generate n RGB values that are equally spaced and contrastive."""
    rgb_values = []
    for i in range(n):
        # Generate colors in HSV space and convert to RGB
        hue = i / n  # Evenly spaced hue values
        saturation = 1.0  # Full saturation
        value = 1.0  # Full brightness
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        # Convert RGB values to 0-255 range
        rgb_values.append((int(r * 255), int(g * 255), int(b * 255)))
    return rgb_values
