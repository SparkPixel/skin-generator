def get_box_faces():
    """
    Three.js BoxGeometry default faces:
    0: +X (Right, mathematically, but char left if char faces +Z)
    1: -X (Left, mathematically, but char right)
    2: +Y (Top)
    3: -Y (Bottom)
    4: +Z (Front)
    5: -Z (Back)
    
    Vertices order for each face:
    0: Top-Left
    1: Top-Right
    2: Bottom-Left
    3: Bottom-Right
    """
    pass

# We will create a small node script that uses puppeteer to load the page, 
# then evaluates what the UV arrays are, but more importantly, 
# testing exactly which faces are flipped.
