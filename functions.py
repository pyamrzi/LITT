from shapely.geometry import Polygon, shape, Point
import math
import numpy as np
import random
import cv2
import rasterio.features
from PIL import Image

# returns an array of classified roi objects and polygon points (fast)
def getRois(conn, image):
    output = []
    roi_service = conn.getRoiService()
    rois = roi_service.findByImage(image.getId(), None).rois
    for i in range(0, len(rois)):
        shape = rois[i].getShape(0)

        # skip if ROI is a transfer
        comment = shape.getTextValue()
        if comment is not None:
            comment = comment._val

        if comment == "Transferred Annotation":
            continue

        roi = {
            "id": rois[i].getId().getValue()
        }
        if(str(type(shape)) == "<class 'omero.model.PolygonI'>"):
            points = shape.getPoints().getValue()
            color = shape.getStrokeColor()._val
            gleason = "None"
            if(color == 822031103):
                gleason = "Grade 3"
            elif(color == -387841):
                gleason = "Grade 4 FG"
            elif(color == -32047105):
                gleason = "Grade 4 CG"
            elif(color == 570425343):
                gleason = "Grade 5"

            roi["points"] = points
            roi["gleason"] = gleason

            # only want to process tumor annotations
            if(roi["gleason"] != "None"):
                output.append(roi)

    return output

# create a shapely polygon from the string returned by Omero
def polygonFromPoints(pointsStr):
    # split into ordered pairs
    pairsStr = pointsStr.split()

    # loop through each ordered pair and split into arrays of points 
    pairs = []
    for i in range(0, len(pairsStr)):
        pair = pairsStr[i].split(',')
        pairs.append(list(filter(None, pair)))

    # create polygon
    try:
        return Polygon(pairs)
    except:
        print("Error")
        pass

    return None

# get tile bounds of mxn dimensions from a polygon
def tileBoundsFromPolygon(polygon, m, n, remove=True):
    # get a bounding box around the polygon
    boundary = polygon.bounds # (minx, miny, maxx, maxy)
    minx = boundary[0]
    miny = boundary[1]
    maxx = boundary[2]
    maxy = boundary[3]

    box = Polygon([[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]])

    numXTiles = math.floor((maxx-minx)/m)
    numYTiles = math.floor((maxy-miny)/n)

    tiles = []

    # split box into 500x500 tiles (roughly)
    xPoint = minx
    yPoint = miny
    for i in range(0, numXTiles+1):
        for j in range(0, numYTiles+1):
            tiles.append(Polygon([[xPoint, yPoint], [xPoint+m, yPoint], [xPoint+m, yPoint+n], [xPoint, yPoint+n]]))
            yPoint = yPoint + n

        xPoint = xPoint + m
        yPoint = miny

    # remove tiles that are not >50% overlapped
    threshold = 0
    if remove == True:
        threshold = 50
    
    i = 0
    while(i < len(tiles)):
        box2 = tiles[i]
        percentOverlap = 0
        if(polygon.intersects(box2)):
            # check for self-intersection
            if(polygon.is_valid):
                percentOverlap = polygon.intersection(box2).area / box2.area
            else:
                try:
                    fixedPolygon = polygon.buffer(0)
                    percentOverlap = fixedPolygon.intersection(box2).area / box2.area
                except:
                    print("intersection could not be fixed, discarding")
                    percentOverlap = 0

        if(percentOverlap <= threshold):
            tiles.remove(box2)
            # want to be on correct index after removal
            i = i - 1 
        i = i + 1

    return tiles

# get tiles given bounding boxes
def findTiles(pixels, boxes):
    input = []
    for i in range(0, len(boxes)):
        box = boxes[i]
        # get smallest x and y values to originate from
        minx = math.floor(box.bounds[0])
        miny = math.floor(box.bounds[1])
        width = math.floor(box.bounds[2] - minx)
        height = math.floor(box.bounds[3] - miny)

        # get all three channels
        input.append((0, 0, 0, (minx, miny, width, height)))
        input.append((0, 1, 0, (minx, miny, width, height)))
        input.append((0, 2, 0, (minx, miny, width, height)))

    tiles = pixels.getTiles(input)

    output = []
    # combine channels
    i = 0
    rg = []
    for tile in tiles:
        if(i == 2):
            rgb = np.dstack((rg[0], rg[1], tile))
            output.append(rgb)
            i = 0
            rg = []
        else:
            rg.append(tile)
            i = i+1

    return output

# get one random 500x500 tile from image (and what it should be labeled as)
def getRandomTileFromImage(imgId, conn):
    image = conn.getObject("Image", imgId)
    sizex = image.getSizeX()
    sizey = image.getSizeY()

    # generate a random box
    randx = random.choice(range(sizex-500))
    randy = random.choice(range(sizey-500))
    box = Polygon([[randx, randy], [randx+500, randy], [randx+500, randy+500], [randx, randy+500]])

    # get image tile
    pixels = image.getPrimaryPixels()
    tile = findTiles(pixels, [box])[0]

    # determine label by checking if box intersects with any ROIs
    rois = getRois(conn, image)
    for roi in rois:
        # get polygon's points
        currRoi = roi["points"]
        
        # get the bounds of each tile
        poly = polygonFromPoints(currRoi)

        # classify the image as that gleason grade if within ROI
        if box.intersects(poly):
            return tile, roi["gleason"]

    if isBackground(tile):
        return tile, "Background"
    else:
        return tile, "Unlabeled"

# does the same thing as the above method but won't give a Background image
def getRandomTileFromImageNoBackground(imgId, conn):
    # code will continue to loop while Background images are found
    while(True):
        image = conn.getObject("Image", imgId)
        sizex = image.getSizeX()
        sizey = image.getSizeY()
    
        # generate a random box
        randx = random.choice(range(sizex-500))
        randy = random.choice(range(sizey-500))
        box = Polygon([[randx, randy], [randx+500, randy], [randx+500, randy+500], [randx, randy+500]])
    
        # get image tile
        pixels = image.getPrimaryPixels()
        tile = findTiles(pixels, [box])[0]
    
        # determine label by checking if box intersects with any ROIs
        rois = getRois(conn, image)
        for roi in rois:
            # get polygon's points
            currRoi = roi["points"]
            
            # get the bounds of each tile
            poly = polygonFromPoints(currRoi)
    
            # classify the image as that gleason grade if within ROI
            if box.intersects(poly):
                return tile, roi["gleason"]
    
        if isBackground(tile) == False:
            return tile, "Unlabeled"

# only get cancerous tile approach 2 (believe it or not this is faster)
def getRandomCancerousTile(imgId, conn):
    # put loop cap to prevent infinite loops
    count = 0
    while(True):
        image = conn.getObject("Image", imgId)
        sizex = image.getSizeX()
        sizey = image.getSizeY()
    
        # generate a random box
        randx = random.choice(range(sizex-500))
        randy = random.choice(range(sizey-500))
        box = Polygon([[randx, randy], [randx+500, randy], [randx+500, randy+500], [randx, randy+500]])
    
        # get image tile
        pixels = image.getPrimaryPixels()
        tile = findTiles(pixels, [box])[0]
    
        # determine label by checking if box intersects with any ROIs
        rois = getRois(conn, image)
        for roi in rois:
            # get polygon's points
            currRoi = roi["points"]
            
            # get the bounds of each tile
            poly = polygonFromPoints(currRoi)
    
            # classify the image as that gleason grade if within ROI
            if box.intersects(poly):
                return tile, roi["gleason"]

        # infinite loop prevention
        count += 1
        if count >= 1e12:
            print("Could not find a cancerous tile")
            return None
    
# get all roi tiles from an image
def getTilesFromImage(image, conn, roiObjs, UNet=False, size=512):
    finalTiles = []

    allPolygons = []

    for roi in roiObjs:
        # get polygon's points
        currRoi = roi["points"]

        # get the bounds of each tile
        poly = polygonFromPoints(currRoi)
        # tileList is a list of BOXES
        tileList = tileBoundsFromPolygon(poly, size, size, remove=not UNet)
        allPolygons.append(poly)

        # get image tiles
        pixels = image.getPrimaryPixels()
        # tiles should be in the same order as the boxes
        tiles = findTiles(pixels, tileList)

        for i in range(len(tiles)):
            tile = tiles[i]
            if(isBackground(tile) == False):
                finalTiles.append({
                    "tile": tile,
                    "roiId": roi["id"],
                    "imgId": image.getId(),
                    "gleason": roi["gleason"],
                    "box": tileList[i]
                })

    # image tiles, polygons
    return finalTiles, allPolygons

# get mxn boxes along the edge of an image
def getEdgeBoxes(sizex, sizey, m, n):
    boxes = []

    #skip some images because we don't need so many
    skipFactor = 60

    # top edge
    for i in range(0, sizex-500, m*skipFactor):
        box = Polygon([[i, 0], [i+500, 0], [i+500, 500], [i, 500]])
        boxes.append(box)
    # bottom edge
    for i in range(0, sizex-500, m*skipFactor):
        box = Polygon([[i, sizey-500], [i+500, sizey-500], [i+500, sizey], [i, sizey]])
        boxes.append(box)
    # left edge
    for i in range(0, sizey-500, n*skipFactor):
        box = Polygon([[0, i], [500, i], [500, i+500], [0, i+500]])
        boxes.append(box)
    # right edge
    for i in range(0, sizey-500, n*skipFactor):
        box = Polygon([[sizex-500, i], [sizex, i], [sizex, i+500], [sizex-500, i+500]])
        boxes.append(box)

    return boxes

def getEdgeTilesFromImage(imgId, conn, m, n):
    image = conn.getObject("Image", imgId)
    sizex = image.getSizeX()
    sizey = image.getSizeY()

    finalTiles = []

    # get edge boxes
    boxes = getEdgeBoxes(sizex, sizey, m, n)

    # get image tiles
    pixels = image.getPrimaryPixels()
    tiles = findTiles(pixels, boxes)

    for tile in tiles:
        if(isBackground(tile) == True):
            finalTiles.append({
                "tile": tile,
                "roiId": "none",
                "imgId": image.getId(),
                "gleason": "background"
            })

    return finalTiles, boxes

# generates nxn boxes to cover an image
def getAllBoxes(sizex, sizey, n=500):
    boxes = []
    # go up/down
    for j in range(0, sizey-n, n):
        # go left/right
        for i in range(0, sizex-n, n):
            box = Polygon([[i, j], [i+n, j], [i+n, j+n], [i, j+n]])
            boxes.append(box)

    return boxes

# gets all nxn tiles in an image
def getAllTiles(conn, imgId, n):
    image = conn.getObject("Image", imgId)
    sizex = image.getSizeX()
    sizey = image.getSizeY()

    # get edge boxes
    boxes = getAllBoxes(sizex, sizey, n)

    # get image tiles
    pixels = image.getPrimaryPixels()
    tiles = findTiles(pixels, boxes)

    # will need boxes again to get masks
    return tiles, boxes

# [IN PROGRESS] generate image-sized masks for each gleason grade
def generateMasks(conn, image):
    mask_dict = {}
    rois = getRois(conn, image)

    # create a list of ROIS of each gleason
    grade3 = []
    grade4fg = []
    grade4cg = []
    grade5 = []
    for roi in rois:
        if roi["gleason"] == "Grade 3":
            grade3.append(polygonFromPoints(roi["points"]))
        elif roi["gleason"] == "Grade 4 FG":
            grade4fg.append(polygonFromPoints(roi["points"]))
        elif roi["gleason"] == "Grade 4 CG":
            grade4cg.append(polygonFromPoints(roi["points"]))
        elif roi["gleason"] == "Grade 5":
            grade5.append(polygonFromPoints(roi["points"]))

    sizex = image.getSizeX()
    sizey = image.getSizeY()

    if len(grade3) > 0:
        grade3_mask = rasterio.features.rasterize(grade3, out_shape=(sizex, sizey))
        mask_dict["grade3"] = grade3_mask
    if len(grade4fg) > 0:
        grade4fg_mask = rasterio.features.rasterize(grade4fg, out_shape=(sizex, sizey))
        mask_dict["grade4fg"] = grade4fg_mask
    if len(grade4cg) > 0:
        grade4cg_mask = rasterio.features.rasterize(grade4cg, out_shape=(sizex, sizey))
        mask_dict["grade4cg"] = grade4cg_mask
    if len(grade5) < 0:
        grade5_mask = rasterio.features.rasterize(grade5, out_shape=(sizex, sizey))
        mask_dict["grade5"] = grade5_mask

    return mask_dict

def getUNetData(conn, imgId):
    image = conn.getObject("Image", imgId)

    # get all tiles and their boxes
    tiles, boxes = getAllTiles(conn, imgId, 512)

    # get all masks
    mask_dict = generateMasks(conn, image)

    # stores the tile and mask
    tile_mask_dict = {}

    # use those boxes segment the masks


def getUnlabeledTilesFromImage(image, conn, allBoxes, cap=800, backgroundAllowed=False, size=512):
    sizex = image.getSizeX()
    sizey = image.getSizeY()

    finalTiles = []

    # find 800 (random) unlabeled boxes
    validBoxes = 0
    boxes = []
    while(validBoxes < cap):
        randX = random.random()*(sizex - 1000)
        randY = random.random()*(sizey - 1000)
        box = Polygon([[randX, randY], [randX+size, randY], [randX+size, randY+size], [randX, randY+size]])

        intersects = False
        for polygon in allBoxes:
            if(box.intersects(polygon)):
                intersects = True
                break

        if(intersects == False):
            validBoxes += 1
            boxes.append(box)

    # get image tiles
    pixels = image.getPrimaryPixels()
    tiles = findTiles(pixels, boxes)

    for tile in tiles:
        if(backgroundAllowed or isBackground(tile) == False):
            finalTiles.append({
                "tile": tile,
                "roiId": "none",
                "imgId": image.getId(),
                "gleason": "unlabeled",
                "box": box
            })

    return finalTiles

def isBackground(tile):
    hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)

    lower = np.array([0,10,0])
    upper = np.array([255,255,255])

    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(tile, tile, mask = mask)

    # is background if less than 75000 non-white pixels
    if (np.count_nonzero(res) < 75000):
        return True

    return False

### UNET stuff ###

# input a list of all rois within the box, get masks sorted by gleason grade as output
def get_masks(rois, box, resolution=100):
    # create a list of ROIS of each gleason
    grade3 = []
    grade4fg = []
    grade4cg = []
    grade5 = []

    # gather polygons by grade
    for tup in rois:
        poly = tup[0]
        roi = tup[1]
        if roi["gleason"] == "Grade 3":
            grade3.append(poly)
        elif roi["gleason"] == "Grade 4 FG":
            grade4fg.append(poly)
        elif roi["gleason"] == "Grade 4 CG":
            grade4cg.append(poly)
        elif roi["gleason"] == "Grade 5":
            grade5.append(poly)

    # dictionary to store all masks
    mask_dict = {}
    mask_dict["Grade 3"] = create_binary_mask(grade3, box, resolution)
    mask_dict["Grade 4 FG"] = create_binary_mask(grade4fg, box, resolution)
    mask_dict["Grade 4 CG"] = create_binary_mask(grade4cg, box, resolution)
    mask_dict["Grade 5"] = create_binary_mask(grade5, box, resolution)

    return mask_dict

# get_masks helper method. Generates the mask (slow)
def create_binary_mask(rois, box, resolution=100):
    # Get bounding box limits
    x_min, y_min, x_max, y_max = box.bounds
    
    # Create a grid of points
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    xv, yv = np.meshgrid(x, y)
    
    mask = np.zeros((resolution, resolution), dtype=np.uint8)

    # Check if each point is within any ROI polygon
    for i in range(resolution):
        for j in range(resolution):
            point = Point(xv[j, i], yv[j, i])  # Corrected indexing here
            if any(polygon.contains(point) for polygon in rois):
                mask[j, i] = 1

    return mask

# input all rois, returns a list of all ROIs within a "box" (fast)
def roisWithin(all_rois, box):
    # stores all rois within boundary
    rois = []
    for roi in all_rois:
        points = roi["points"]
        
        # get the shapely polygon
        poly = polygonFromPoints(points)

        if box.intersects(poly):
            # also include polygon in rois returned
            rois.append([poly, roi])

    return rois

# returns a dictionary of T/F depending on if a certain type of ROI is contained within the image
# rois inputted as [poly, roi]
def containsWhich(rois):
    grade3 = False
    grade4fg = False
    grade4cg = False
    grade5 = False
    for tup in rois:
        roi = tup[1]
        if roi["gleason"] == "Grade 3":
            grade3 = True
        elif roi["gleason"] == "Grade 4 FG":
            grade4fg = True
        elif roi["gleason"] == "Grade 4 CG":
            grade4cg = True
        elif roi["gleason"] == "Grade 5":
            grade5 = True

    return grade3, grade4fg, grade4cg, grade5

def lowerResolution(tile, outputSize=512):
    resized = np.array(Image.fromarray(tile).resize((outputSize,outputSize)))
    return resized

# similar to getAllTiles but only gets a list of the boxes
def getRoiBoxes(image, conn, roiObjs, UNet=True, size=512):
    allBoxes = []
    allPolygons = []

    for roi in roiObjs:
        # get polygon's points
        currRoi = roi["points"]

        # get the bounds of each tile
        poly = polygonFromPoints(currRoi)
        allPolygons.append(poly)

        # tileList is a list of BOXES
        boxes = tileBoundsFromPolygon(poly, size, size, remove=not UNet)

        for i in range(len(boxes)):
            box = boxes[i]
            allBoxes.append({
                "roiId": roi["id"],
                "imgId": image.getId(),
                "gleason": roi["gleason"],
                "box": box
            })


    return allBoxes, allPolygons

# input the box around the tile and recieve one tile
def getTileFromBox(image, box):
    # get image tiles
    pixels = image.getPrimaryPixels()
    # tiles should be in the same order as the boxes
    tile = findTiles(pixels, [box])[0]

    return tile

# getUnlabeledTilesFromImage but returns boxes instead of tiles
def getUnlabeledBoxes(image, conn, allBoxes, cap=800, backgroundAllowed=True, size=512):
    sizex = image.getSizeX()
    sizey = image.getSizeY()

    finalBoxes = []

    # find 800 (random) unlabeled boxes
    validBoxes = 0
    boxes = []
    while(validBoxes < cap):
        randX = random.random()*(sizex - size)
        randY = random.random()*(sizey - size)
        box = Polygon([[randX, randY], [randX+size, randY], [randX+size, randY+size], [randX, randY+size]])

        intersects = False
        for polygon in allBoxes:
            if(box.intersects(polygon)):
                intersects = True
                break

        if(intersects == False):
            validBoxes += 1
            boxes.append(box)

    for box in boxes:
        if(backgroundAllowed or isBackground(tile) == False):
            finalBoxes.append({
                "roiId": "none",
                "imgId": image.getId(),
                "gleason": "unlabeled",
                "box": box
            })

    return finalBoxes