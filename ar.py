
import cv2
import numpy as np
import math

def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """
    return np.linalg.norm(np.array(p0)-np.array(p1))

def get_corners_list(image):
    """Returns a list of image corner coordinates used in warping.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    if len(image.shape) == 2:
        H,W = image.shape
    else:        
        H,W,C = image.shape
    corners = [(0,0),(0, H-1),(W-1,0),(W-1,H-1)]
#     print(corners)
    return corners
    

def find_markers_hough(image, template=None):
    """Finds four corner markers.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
#     cv2.imwrite("test.png", image)

    threshold = 40
    col_var_threshold = 50
    if len(image.shape) == 2:
        H,W = image.shape
    else:        
        H,W,C = image.shape
    
    t_image = image.copy()
    kernel_x = cv2.getGaussianKernel(W,500)
    kernel_y = cv2.getGaussianKernel(H,500)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    for i in range(3): t_image[:,:,i] = t_image[:,:,i] * mask
        
#     image = cv2.GaussianBlur(image,(3,3),5)
#     image = cv2.medianBlur(image,7)    
    t_image = cv2.GaussianBlur(t_image,(5,5),5)
    t_image = cv2.medianBlur(t_image,9)    
    gray = cv2.cvtColor(t_image, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite("test0.png", gray)
    edges = cv2.Canny(t_image,10,100)
#     cv2.imwrite("test1.png", edges)

    markers = []
#     circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=10,minRadius=5,maxRadius=0)[0]
    circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=10,minRadius=5,maxRadius=0)
    if circles is not None:
        circles = circles[0]
        img = image.copy()
        for c in circles: cv2.circle(img, (int(c[0]),int(c[1])), c[2], (0,255,0), thickness=1)
#         cv2.imwrite("test2.png", img)
        
        for c in circles:
            d = c[2]/3
            if c[1]-d > 0 and c[0]-d > 0 and c[1]+d < H and c[0]+d < W:
                r1 = get_avg_neighborhood(gray,int(c[1]-d),int(c[0]-d))
                r2 = get_avg_neighborhood(gray,int(c[1]-d),int(c[0]+d))
                r3 = get_avg_neighborhood(gray,int(c[1]+d),int(c[0]+d))
                r4 = get_avg_neighborhood(gray,int(c[1]+d),int(c[0]-d))
                c1 = np.abs(r1 - r2) > threshold
                c2 = np.abs(r3 - r4) > threshold
                c3 = np.abs(r1 - r3) < threshold
                c4 = np.abs(r2 - r4) < threshold
                c5 = (r1 - r2) * (r4 - r3) < 0
                criteria = [c1,c2,c3,c4,c5]
#                 print(criteria,r1,r2,r3,r4)
                if np.prod(criteria):
                    c6 = True
                    if len(markers) != 0: 
                        for m in markers: 
                            c6 *= euclidean_distance((c[0],c[1]),m) > threshold
                    if c6: markers.append((int(c[0]),int(c[1])))       
                    
                    
    if len(markers) != 4 and template is not None:
        markers = []
        max_score = -float('inf')
        matched_list = None
        for a in range(0,180,15):
            for s in range(2):
                scale = (s+1)/1
                h,w = template.shape[0],template.shape[1]
                R = cv2.getRotationMatrix2D((int(h/2),int(w/2)), a, 1)
                resized_template = cv2.resize(template, (int(template.shape[0]*scale),int(template.shape[1])), interpolation = cv2.INTER_AREA) 
                template_rotated = rotate_ang(resized_template, R)
                gray_template = cv2.cvtColor(template_rotated, cv2.COLOR_BGR2GRAY)
                norm_image = cv2.normalize(image,None,-127,128,norm_type=cv2.NORM_MINMAX)
                norm_template = cv2.normalize(template_rotated,None,-127,128,norm_type=cv2.NORM_MINMAX)
#                 matched = cv2.matchTemplate(image,template_rotated,method=cv2.TM_CCOEFF_NORMED)
                matched = cv2.matchTemplate(norm_image,norm_template,method=cv2.TM_CCOEFF_NORMED)
#                 matched = cv2.matchTemplate(gray,gray_template,method=cv2.TM_CCOEFF_NORMED)
                score = np.max(matched)
                if score > max_score:
                    max_score = score
                    matched_list = matched
        
        while True:
            _, _, _, coord = cv2.minMaxLoc(matched_list)
            matched_list[coord[1],coord[0]] = 0
            c6 = True
            true_coord = (int(coord[0]+h/2),int(coord[1]+w/2))
            if len(markers) != 0: 
                for m in markers: 
                    c6 *= euclidean_distance(true_coord,m) > threshold
            c6 *= np.std(image[true_coord[1],true_coord[0],:]) < col_var_threshold
            if len(markers) == 3:
                c6 *= not in_triangle(markers[0],markers[1],markers[2],true_coord)
            if c6: markers.append(true_coord)
            if len(markers) >= 4: 
                break

    if len(markers) == 0: return None
    markers = order_markers(markers,H,W)
#     print(markers)                
    
    return markers


def find_markers_fast(image, template=None):
    # Initialize Lists
    tempImage = np.copy(image)
    rows, columns, _ = tempImage.shape
    
    kernel_x = cv2.getGaussianKernel(columns,500)
    kernel_y = cv2.getGaussianKernel(rows,500)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    for i in range(3): tempImage[:,:,i] = tempImage[:,:,i] * mask

    # Implement HarrisCorner Detection
#     tempImage = cv2.medianBlur(tempImage, 3)
#     tempImage = cv2.GaussianBlur(tempImage, (5, 5), 4)
    gray = cv2.cvtColor(tempImage, cv2.COLOR_BGR2GRAY)

#     gray = cv2.GaussianBlur(gray, (5, 5), 0)
#     res = cv2.cornerHarris(gray, 7, 5, 0.05)
#     res = cv2.cornerHarris(gray, 7, 7, 0.001)
#     gray = cv2.dilate(gray, None)

#     res = cv2.cornerHarris(gray, 9, 9, 0.001)
    res = cv2.cornerHarris(gray, 11, 11, 0.0005)
    res = cv2.normalize(res,                # src
                        None,               # dst
                        0,                  # alpha
                        255,                # beta
                        cv2.NORM_MINMAX,    # norm type
                        cv2.CV_32F,         # dtype
                        None                # mask
                        )

    pts = np.where(res >= 0.12 * res.max())
    pts = list(zip(pts[1], pts[0]))
    pts = np.asarray(pts)
    pts = np.float32(pts)

    # Points Grouping and filtering - Using K-Mean Cluster with 4 groups
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(pts, 4, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # Remove outlier
    thres = 20;
    pts2 = list()

    for i in pts:
        if (abs(i[0] - center[0][0]) + abs(i[1] - center[0][1]) < thres) or \
           (abs(i[0] - center[1][0]) + abs(i[1] - center[1][1]) < thres) or \
           (abs(i[0] - center[2][0]) + abs(i[1] - center[2][1]) < thres) or \
           (abs(i[0] - center[3][0]) + abs(i[1] - center[3][1]) < thres):

            pts2.append(i)

    pts2 = np.asarray(pts2)
    
    distance = [(x[0], x[1], x[0] ** 2 + x[1] ** 2, x[0] ** 2 + (x[1] - rows) ** 2,
                (x[0] - columns) ** 2 + x[1] ** 2, (x[0] - columns) ** 2 + (x[1] - rows) ** 2) for x in center]

    d1 = distance[:]
    d1.sort(key=lambda var: var[2])
    d2 = [pts for pts in d1 if pts == d1[1] or pts == d1[2] or pts == d1[3]]
    d2.sort(key=lambda var: var[3])
    d3 = [pts for pts in d2 if pts == d2[1] or pts == d2[2]]
    d3.sort(key=lambda var: var[4])

    returnList = [d1[0], d2[0], d3[0], d3[1]]
    returnList = [(int(pts[0]), int(pts[1])) for pts in returnList]

    return returnList

find_markers = find_markers_fast
# find_markers = find_markers_hough

def area_of_triangle(p1,p2,p3): 
  
    return abs((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2.0) 
  
  
def in_triangle(p1,p2,p3,px): 
  
    established_A = area_of_triangle(p1,p2,p3) 
    return established_A == area_of_triangle(px,p2,p3) + area_of_triangle(p1,px,p3) + area_of_triangle(p1,p2,px)


def rotate_ang(image,R):
    out = image.copy()
    h,w = out.shape[0],out.shape[1]
    for x in range(h):
        for y in range(w):
            x1 = R[0,0]*x + R[0,1]*y + R[0,2]
            y1 = R[1,0]*x + R[1,1]*y + R[1,2]
            if int(x1) < h and int(y1) < w:
                out[x,y] = image[int(x1),int(y1)]
    return out
    

def order_markers(markers, H, W):
    final_list = []
    markers_array = np.array(markers)
#     print(markers)
    left_seq = np.argsort(markers_array[:,0])        
    top_seq = np.argsort(markers_array[:,1])     
    top_left_seq = np.argsort(markers_array[left_seq[:2],1])
    top_right_seq = np.argsort(markers_array[left_seq[-2:],1])     
#     print(left_seq, top_seq)   
    if left_seq[0] == top_left_seq[0]:
        final_list.append((markers[left_seq[0]][0],markers[left_seq[0]][1]))
        final_list.append((markers[left_seq[1]][0],markers[left_seq[1]][1]))
    else:
        final_list.append((markers[left_seq[1]][0],markers[left_seq[1]][1]))
        final_list.append((markers[left_seq[0]][0],markers[left_seq[0]][1]))
    if left_seq[-1] == top_right_seq[-1]:
        final_list.append((markers[left_seq[-1]][0],markers[left_seq[-1]][1]))
        final_list.append((markers[left_seq[-2]][0],markers[left_seq[-2]][1]))
    else:
        final_list.append((markers[left_seq[-2]][0],markers[left_seq[-2]][1]))
        final_list.append((markers[left_seq[-1]][0],markers[left_seq[-1]][1]))
    
    left_sort = sorted(markers, key=lambda t: (t[0]*H + t[1]))
    left_ones = sorted(left_sort[0:2], key=lambda t: (t[1]*W + t[0]))
    right_ones = sorted(left_sort[2:4], key=lambda t: (t[1]*W + t[0]))
    final_list = [left_ones[0],left_ones[1],right_ones[0],right_ones[1]]

    return final_list


def get_avg_neighborhood(gray,x,y):
    return float(gray[x,y])
#     return np.mean([gray[x,y],gray[x+1,y],gray[x-1,y],gray[x,y+1],gray[x,y-1],gray[x+1,y+1],gray[x+1,y-1],gray[x-1,y-1],gray[x-1,y+1]])


def draw_box(image, markers, thickness=1,color=(0,0,0)):
    """Draws lines connecting box markers.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """
#     cv2.line(image, markers[0], markers[1], (255,1,255), thickness)
#     cv2.line(image, markers[0], markers[2], (255,1,255), thickness)
#     cv2.line(image, markers[1], markers[3], (255,1,255), thickness)
#     cv2.line(image, markers[2], markers[3], (255,1,255), thickness)
    cv2.line(image, markers[0], markers[1], color, thickness=thickness)
    cv2.line(image, markers[0], markers[2], color, thickness=thickness)
    cv2.line(image, markers[1], markers[3], color, thickness=thickness)
    cv2.line(image, markers[2], markers[3], color, thickness=thickness)
    return image


def project_mirror_onto_imageB(imageA, imageB, homography,m_homography):
    """Projects mirror into the marked area in imageB.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """
    if len(imageB.shape) == 2:
        H,W = imageB.shape
        C = 1      
    else:        
        H,W,C = imageB.shape
    
    image = imageB.copy()
    imageM = imageA.copy()
    imageM = cv2.flip(imageM, 1)

    
    map_x, map_y = get_mapping(m_homography,imageA.shape[0],imageA.shape[1],center=True)
    imageM = cv2.remap(imageM,map1=np.float32(map_x),map2=np.float32(map_y),interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_TRANSPARENT)
#     cv2.imwrite("test3.png", imageM)
    
#     map_x[abs(map_x)<10] = 100000
#     map_x[abs(map_x-imageA.shape[1]+1)<10] = 100000
#     map_y[abs(map_y)<10] = 100000
#     map_y[abs(map_y-imageA.shape[0]+1)<10] = 100000
#     map_x = map_x-10000
#     map_y = map_y-10000
#     map = map_x + map_y
#     map = cv2.normalize(map, None, 0, 255, cv2.NORM_MINMAX)
# #     print(np.where(map == 255)[0].shape,np.where(map == 255)[1].shape)
#     cv2.imwrite("test5.png", map)
    
    gray = cv2.cvtColor(imageM, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),3)
#     gray = cv2.medianBlur(gray,11)    
    gray[gray > 0.1] = 10000
#     cv2.imwrite("test6.png", gray)
    res = cv2.cornerHarris(gray, 11, 11, 0.005)
    res = cv2.normalize(res,None, 0,255,cv2.NORM_MINMAX, cv2.CV_32F,None)

    pts = np.where(res >= 0.1 * res.max())
    pts = list(zip(pts[1], pts[0]))
    pts = np.asarray(pts)
    pts = np.float32(pts)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(pts, 4, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    t_center = []
    for e in center: t_center.append(tuple(e))
    map = imageM.copy()
    map = draw_box(map, order_markers(t_center, imageA.shape[0], imageA.shape[1]),3,255)
    centroid = np.mean(center,0)
    cv2.circle(map, tuple(centroid), 10, (0, 50, 255), -1)
    x_l = int(centroid[1]-imageA.shape[0]/2)
    x_u = int(centroid[1]+imageA.shape[0]/2)
    y_l = int(centroid[0]-imageA.shape[1]/2)
    y_u = int(centroid[0]+imageA.shape[1]/2)
    map = draw_box(map, order_markers([(y_l,x_l),(y_l,x_u),(y_u,x_l),(y_u,x_u)], imageA.shape[0], imageA.shape[1]),3,(0, 50, 255))

#     cv2.imwrite("test5.png", map)
    
    imageM = imageM[x_l:x_u,y_l:y_u] 
    
#     imageM = cv2.flip(imageM, 1)
#     cv2.imwrite("test4.png", imageM)

    map_x, map_y = get_mapping(homography,H,W)
    try:
        cv2.remap(imageM,dst=image,map1=np.float32(map_x),map2=np.float32(map_y),interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_TRANSPARENT)
    except:
        print('skipped')
    
    return image              

def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """
    if len(imageB.shape) == 2:
        H,W = imageB.shape
        C = 1      
    else:        
        H,W,C = imageB.shape
    
    image = imageB.copy()
    
    map_x, map_y = get_mapping(homography,H,W)
    cv2.remap(imageA,dst=image,map1=np.float32(map_x),map2=np.float32(map_y),interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_TRANSPARENT)
    

    return image          

def get_mapping(H,h,w,center=False):
    
    if center:
        magnify = 6
        r,c = np.indices((h*magnify, w*magnify))
        r = r - h*2
        c = c - w*2
        stacked_indices = np.array([c.ravel(),r.ravel(),np.ones_like(c.ravel())])
        backwrap = np.dot(np.linalg.inv(H), stacked_indices)
        map_x,map_y = backwrap[:-1] / backwrap[-1]
        map_x,map_y = map_x.reshape((h*magnify, w*magnify)), map_y.reshape((h*magnify, w*magnify))
        find_thrs = 0.005
#         print(np.where(abs(map_x) == np.min(abs(map_x))))
#         print(np.where(abs(map_x-w+1) == np.min(abs(map_x-w+1))))
#         print(np.where(abs(map_y) == np.min(abs(map_y))))
#         print(np.where(abs(map_y-h+1) == np.min(abs(map_y-h+1))))
#         ,np.where(abs(map_x-w+1)<0.5).shape,np.where(abs(map_y)<0.5).shape,np.where(abs(map_y-h+1)<0.5).shape)
    else:
        r,c = np.indices((h, w))
        stacked_indices = np.array([c.ravel(),r.ravel(),np.ones_like(c.ravel())])
        backwrap = np.dot(np.linalg.inv(H), stacked_indices)
        map_x,map_y = backwrap[:-1] / backwrap[-1]
        map_x,map_y = map_x.reshape((h, w)), map_y.reshape((h, w))
    return map_x,map_y
    
def interpolate(image,x,y,C):
    h,w = image.shape[0],image.shape[1]
#     x,y = y,x
    x0,x1,y0,y1 = int(np.floor(x)),int(np.ceil(x)),int(np.floor(y)),int(np.ceil(y))
    if x1 >= h: x1 = h-1
    if y1 >= w: y1 = w-1
    if x0 >= h or y0 >= w or x0 < 0 or y0 < 0: return None 
#     print(x,y,x0,x1,y0,y1,image.shape)
    if C == 1:
        c0 = image[int(x),int(y)]
        c1 = (1-x+x0)*(1-y+y0)*image[x0,y0]
        c2 = (1-x+x0)*(y-y0)*image[x0,y1]
        c3 = (x-x0)*(1-y+y0)*image[x1,y0]
        c4 = (x-x0)*(y-y0)*image[x1,y1]
        col = int(c1 + c2 + c3 + c4)
    else: 
        c0 = image[int(x),int(y)]
        c1 = (1-x+x0)*(1-y+y0)*image[x0,y0,:]
        c2 = (1-x+x0)*(y-y0)*image[x0,y1,:]
        c3 = (x-x0)*(1-y+y0)*image[x1,y0,:]
        c4 = (x-x0)*(y-y0)*image[x1,y1,:]
        col = c1 + c2 + c3 + c4
        col = (int(col[0]),int(col[1]),int(col[2]))
#     return c0
    return col
        
        
def backwrap(x2,y2,h):
    
    x1 = (h[0,0]*x2 + h[0,1]*y2 + h[0,2]) / (h[2,0]*x2 + h[2,1]*y2 + h[2,2]) 
    y1 = (h[1,0]*x2 + h[1,1]*y2 + h[1,2]) / (h[2,0]*x2 + h[2,1]*y2 + h[2,2])  
    
#     x1 = (h[0,0]*x2 + h[1,0]*y2 + h[2,0]) / (h[0,2]*x2 + h[1,2]*y2 + h[2,2])
#     y1 = (h[0,1]*x2 + h[1,1]*y2 + h[2,1]) / (h[0,2]*x2 + h[1,2]*y2 + h[2,2]) 
    
    return x1,y1


def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """
    to_solve = []
    
    for (x,y), (u,v) in zip(src_points, dst_points):
        to_solve.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        to_solve.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
        
    u,s,v = np.linalg.svd(np.array(to_solve))
    homograph = v[v.shape[0]-1,:] / v[v.shape[0]-1,v.shape[1]-1]
    homograph = homograph.reshape(3,3)
    
    return homograph


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    video = cv2.VideoCapture(filename)

    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break
        
        c = cv2.waitKey(1)
        if c == 27: break

    video.release()
    yield None
    