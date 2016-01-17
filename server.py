from flask import Flask, request, redirect, url_for, jsonify
import cv2
import numpy as np
import scipy.spatial as spatial
import scipy.cluster as clstr
from time import time
from collections import defaultdict
from functools import partial
import glob
import caffe
import skimage
import pickle

""" CONSTANTS """

CAFFENET_DEPLOY_TXT = '/Users/daylenyang/caffe/models/finetune_chess/deploy.prototxt'
CAFFENET_MODEL_FILE = '/Users/daylenyang/caffe/models/finetune_chess/finetune_chess_iter_5554.caffemodel'

categories = ['bb', 'bk', 'bn', 'bp', 'bq', 'br', 'empty', 'wb', 'wk', 'wn', 'wp', 'wq', 'wr']
BATCH_SIZE = 64

assert 64 % BATCH_SIZE == 0

""" LOGIC """

def auto_canny(image, sigma=0.33):
    """
    Canny edge detection with automatic thresholds.
    """
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged

def hor_vert_lines(lines):
    """
    A line is given by rho and theta. Given a list of lines, returns a list of
    horizontal lines (theta=90 deg) and a list of vertical lines (theta=0 deg).
    """
    h = []
    v = []
    for distance, angle in lines:
        if angle < np.pi / 4 or angle > np.pi - np.pi / 4:
            v.append([distance, angle])
        else:
            h.append([distance, angle])
    return h, v

def intersections(h, v):
    """
    Given lists of horizontal and vertical lines in (rho, theta) form, returns list
    of (x, y) intersection points.
    """
    points = []
    for d1, a1 in h:
        for d2, a2 in v:
            A = np.array([[np.cos(a1), np.sin(a1)], [np.cos(a2), np.sin(a2)]])
            b = np.array([d1, d2])
            point = np.linalg.solve(A, b)
            points.append(point)
    return np.array(points)

def cluster(points, max_dist=50):
    """
    Given a list of points, returns a list of cluster centers.
    """
    Y = spatial.distance.pdist(points)
    Z = clstr.hierarchy.single(Y)
    T = clstr.hierarchy.fcluster(Z, max_dist, 'distance')
    clusters = defaultdict(list)
    for i in range(len(T)):
        clusters[T[i]].append(points[i])
    clusters = clusters.values()
    clusters = map(lambda arr: (np.mean(np.array(arr)[:,0]), np.mean(np.array(arr)[:,1])), clusters)
    return clusters

def closest_point(points, loc):
    """
    Returns the list of points, sorted by distance from loc.
    """
    dists = np.array(map(partial(spatial.distance.euclidean, loc), points))
    return points[dists.argmin()]

def find_corners(points, img_dim):
    """
    Given a list of points, returns a list containing the four corner points.
    """
    center_point = closest_point(points, (img_dim[0] / 2, img_dim[1] / 2))
    points.remove(center_point)
    center_adjacent_point = closest_point(points, center_point)
    points.append(center_point)
    grid_dist = spatial.distance.euclidean(np.array(center_point), np.array(center_adjacent_point))
    
    img_corners = [(0, 0), (0, img_dim[1]), img_dim, (img_dim[0], 0)]
    board_corners = []
    tolerance = 0.25 # bigger = more tolerance
    for img_corner in img_corners:
        while True:
            cand_board_corner = closest_point(points, img_corner)
            points.remove(cand_board_corner)
            cand_board_corner_adjacent = closest_point(points, cand_board_corner)
            corner_grid_dist = spatial.distance.euclidean(np.array(cand_board_corner), np.array(cand_board_corner_adjacent))
            if corner_grid_dist > (1 - tolerance) * grid_dist and corner_grid_dist < (1 + tolerance) * grid_dist:
                points.append(cand_board_corner)
                board_corners.append(cand_board_corner)
                break
    return board_corners

def four_point_transform(img, points, square_length=1816):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [0, square_length], [square_length, square_length], [square_length, 0]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (square_length, square_length))

def find_board(fname):
    """
    Given a filename, returns the board image.
    """
    start = time()
    img = cv2.imdecode(fname, 1)
    if img is None:
        print 'no image'
        return None
    print img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3, 3))
    
    # Canny edge detection
    edges = auto_canny(gray)
    if np.count_nonzero(edges) / float(gray.shape[0] * gray.shape[1]) > 0.015:
        print 'too many edges'
        return None

    # Hough line detection
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    if lines is None:
        print 'no lines'
        return None

    lines = np.reshape(lines, (-1, 2))
    
    # Compute intersection points
    h, v = hor_vert_lines(lines)
    if len(h) < 9 or len(v) < 9:
        print 'too few lines'
        return None
    points = intersections(h, v)
    
    # Cluster intersection points
    points = cluster(points)

    # Find corners
    img_shape = np.shape(img)
    points = find_corners(points, (img_shape[1], img_shape[0]))
    
    # Perspective transform
    new_img = four_point_transform(img, points)

    return new_img

def split_board(img):
    """
    Given a board image, returns an array of 64 smaller images.
    """
    arr = []
    sq_len = img.shape[0] / 8
    for i in range(8):
        for j in range(8):
            arr.append(img[i * sq_len : (i + 1) * sq_len, j * sq_len : (j + 1) * sq_len])
    return arr

def shrink_blanks(fen):
    if '_' not in fen:
        return fen
    new_fen = ''
    blanks = 0
    for char in fen:
        if char == '_':
            blanks += 1
        else:
            if blanks != 0:
                new_fen += str(blanks)
                blanks = 0
            new_fen += char
    if blanks != 0:
        new_fen += str(blanks)
    return new_fen

def get_fen(arr):
    fen = ''
    for sq in arr:
        if sq == 'empty':
            fen += '_'
        elif sq[0] == 'b':
            fen += sq[1]
        else:
            fen += str(sq[1]).upper()
    fens = [fen[i:i+8] for i in range(0, 64, 8)]
    fens = map(shrink_blanks, fens)
    fen = '/'.join(fens)
    return fen

def get_square_to_pieces_dict(prob_matrix):
    d = {}
    for i in range(len(prob_matrix)):
        d[i] = map(lambda x: categories[x], np.argsort(-prob_matrix[i]))
    return d

""" LOADING """

net = caffe.Net(CAFFENET_DEPLOY_TXT, CAFFENET_MODEL_FILE, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.array([104.00698793, 116.66876762, 122.67891434]));
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

caffe.set_device(0)
caffe.set_mode_gpu()
net.blobs['data'].reshape(BATCH_SIZE, 3, 227, 227)

""" ROUTES """

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Chess ID. usage: /upload'

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'JPG', 'JPEG'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            start = time()
            img = np.asarray(bytearray(file.read()))
            board = find_board(img)
            if board is None:
                return jsonify({'error': 'could not find board'})
            squares = split_board(board)
            print 'finished board rec', time() - start
            input_images = [transformer.preprocess('data', skimage.img_as_float(square).astype(np.float32)) for square in squares]
            print 'finished preprocess', time() - start
            predictions = None
            print 'using batch size', BATCH_SIZE
            for i in range(0, 64, BATCH_SIZE):
                net.blobs['data'].data[...] = np.array(input_images[i:i+BATCH_SIZE])
                out = net.forward()['prob']
                if predictions == None:
                    predictions = out
                else:
                    predictions = np.vstack((predictions, out))

            print 'finished nn', time() - start
            fen = get_fen(map(lambda x: categories[x], np.argmax(predictions, axis=1)))
            json = {'fen': fen, 'time': time() - start}
            return jsonify(json)
    return '''
    <!doctype html>
    <title>Chess ID</title>
    <h1>Upload board picture</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)