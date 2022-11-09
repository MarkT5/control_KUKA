import math

import cv2
import matplotlib.pyplot as pyplot
import numpy as np
import scipy
import scipy.spatial


def open_log():
    f = open("lidar_odom_log/lidar_odom_log_8.txt", "r")
    txt_log_data = f.read().split('\n')
    log_data = []
    for i in txt_log_data:
        sp_log_data = i.split(';')
        if sp_log_data[-1] == '':
            break
        odom = sp_log_data[0].split(',')
        lidar = sp_log_data[1].split(',')
        odom = list(map(float, odom))
        lidar = list(map(float, lidar))
        log_data.append([odom, lidar])

    # print(len(log_data), "samples in log file")
    return log_data[80]


# inp2 = log_data[7]

map_size = 100
discrete = 20
all_detected_corners = []


def scle_to_arr(x, y):
    # return [int(map_size / 2 + discrete * x), int(map_size / 2 - discrete * y)]
    return [discrete * x, -discrete * y]


def fit_line(points):
    r = np.polyfit([p[0] for p in points], [p[1] for p in points], 10)
    return r


def conv_cil_to_dec(input, ind=None):
    out_points = []
    odom, lidar = input
    x, y, ang = odom
    cent_y, cent_x = y, x
    cent_y = cent_y - 0.3 * math.cos(ang + math.pi / 2)
    cent_x = cent_x + 0.3 * math.sin(ang + math.pi / 2)
    if isinstance(ind, int):
        lid_ang = ind * math.radians(240) / len(lidar) - ang - math.radians(30)
        lid_dist = lidar
        if lid_dist > 5.5:
            return None, None
        ox = cent_x + lid_dist * math.sin(lid_ang)
        oy = cent_y + lid_dist * math.cos(lid_ang)
        ox, oy = scle_to_arr(ox, oy)
        return [ox, oy]
    elif isinstance(ind, list):
        for i in ind:
            lid_ang = i * math.radians(240) / len(lidar) - ang - math.radians(30)
            lid_dist = lidar[i]
            if lid_dist > 5.5:
                continue
            ox = cent_x + lid_dist * math.sin(lid_ang)
            oy = cent_y + lid_dist * math.cos(lid_ang)
            out_points.append(scle_to_arr(ox, oy))
    else:
        for i in range(len(lidar)):
            lid_ang = i * math.radians(240) / len(lidar) - ang - math.radians(30)
            lid_dist = lidar[i]
            if lid_dist > 5.5:
                continue
            ox = cent_x + lid_dist * math.sin(lid_ang)
            oy = cent_y + lid_dist * math.cos(lid_ang)
            out_points.append(scle_to_arr(ox, oy))
    return out_points


def best_fit_transform(A, B, cornerA_ind, cornerB_ind):
    # assert A.shape == B.shape

    # get number of dimensions
    n, m = A.shape
    # translate points to their centroids
    weights = np.concatenate((np.array(range(0, cornerA_ind))**4, np.array(range(n, cornerA_ind, -1))**4))
    weights_sum = np.sum(weights)
    weights = np.diag(weights)
    centroid_A = np.sum(np.dot(weights, A), axis=0) / weights_sum
    centroid_B = np.sum(np.dot(weights, B), axis=0) / weights_sum
    #print(centroid_A, "new")
    #centroid_A = np.mean(A, axis=0)
    #centroid_B = np.mean(B, axis=0)
    #print(centroid_A, "old")
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    # assert src.shape == dst.shape

    A_tree = scipy.spatial.KDTree(dst)
    dist, indexes = A_tree.query(src)
    return dist.ravel(), indexes.ravel()


def icp(A, B, cornerA_ind, cornerB_ind, init_pose=None, max_iterations=20, tolerance=0.01):
    print(A.shape, B.shape)
    # assert A.shape == B.shape
    cornerA = A[cornerA_ind]
    cornerB = B[cornerB_ind]
    # get number of dimensions
    m = A.shape[1]


    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0
    distances = []
    i = 0
    error = None
    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T, cornerA_ind, cornerB_ind)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            error = np.abs(prev_error - mean_error)
            print('smol')
            break

        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:m, :].T, cornerA_ind, cornerB_ind)

    return T, distances, i, error


def split_objects(data):
    odom, lidar = data
    min_len = 30
    ox, oy = 0, 0
    all_objects = [[]]
    out_objects = []
    for i in range(1, len(lidar)):
        if lidar[i] > 5.5 or lidar[i - 1] > 5.5:
            continue
        if abs(lidar[i - 1] - lidar[i]) < 0.4:
            all_objects[-1].append(i - 1)
        else:
            all_objects.append([])
    for i in all_objects:
        if len(i) > min_len:
            out_objects.append(conv_cil_to_dec([odom, lidar], i))
    out_objects = sorted(out_objects, key=lambda x: len(x), reverse=True)
    return out_objects


def hough_transform_dec(input):
    resolution = int(map_size * 2.5)
    approx_reg = int(map_size * 0.2)
    min_detect_val = 0.08
    approx_reg_mid = int(approx_reg / 2)
    max_point_val = len(input)
    detected_peaks_params = []

    hough_graph = np.zeros((resolution, resolution))
    for i in range(len(input)):
        for j in range(1, resolution):
            point = int(input[i][1] * math.cos(math.pi / resolution * j) - input[i][0] * math.sin(
                math.pi / resolution * j) + resolution / 2)
            if abs(point) < resolution:
                hough_graph[point][j] += 1 / max_point_val

    resized = cv2.resize(hough_graph, (500, 500), interpolation=cv2.INTER_AREA)
    cv2.imshow("hough_graph", resized)

    # define an 8-connected neighborhood
    n_mask = scipy.ndimage.generate_binary_structure(2, 1)
    neighborhood = np.zeros((approx_reg, approx_reg))
    neighborhood[approx_reg_mid][approx_reg_mid] = 1
    neighborhood = scipy.ndimage.binary_dilation(neighborhood, structure=n_mask).astype(n_mask.dtype)
    for i in range(int(approx_reg_mid / 3)):
        neighborhood = scipy.ndimage.binary_dilation(neighborhood, structure=neighborhood).astype(n_mask.dtype)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = scipy.ndimage.maximum_filter(hough_graph, size=20) == hough_graph

    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (hough_graph < min_detect_val)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = scipy.ndimage.binary_erosion(background, structure=neighborhood, border_value=1)
    background_inv = (eroded_background == False)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (and operation)
    detected_peaks = local_max & background_inv

    for i in range(len(detected_peaks)):
        for j in range(len(detected_peaks)):
            if detected_peaks[i][j]:
                f, p, = math.pi / resolution * j, i - resolution / 2
                detected_peaks_params.append([f, p, hough_graph[i][j]])
    detected_peaks_params = sorted(detected_peaks_params, key=lambda x: x[-1], reverse=True)
    return detected_peaks_params


def find_farthest(points, threshold=1):
    if not points:
        return None
    x1, y1 = points[0]
    x2, y2 = points[-1]
    main_vert = False
    perp_vert = False
    a_main = 0
    a_perp = 0
    max_dist = -1
    max_i = None
    if x1 == x2:
        main_vert = True
    else:
        a_main = (y1 - y2) / (x1 - x2)
    b_main = y1 - a_main * x1
    if a_main != 0:
        a_perp = -1 / a_main
    else:
        perp_vert = True
    for i in range(1, len(points) - 2):
        xp, yp = points[i]
        if main_vert:
            dist = xp ** 2
        elif perp_vert:
            dist = yp ** 2
        else:
            b_perp = yp - a_perp * xp
            xc = (b_main - b_perp) / (a_perp - a_main)
            yc = xc * a_main + b_main
            dist = (xp - xc) ** 2 + (yp - yc) ** 2
        if dist > max_dist:
            max_dist = dist
            max_i = i
    if max_dist < threshold:
        return None
    if max_dist == -1:
        return None
    # print(max_dist)
    # draw_line(a_main, b_main)
    # draw_line(a_perp, b_perp)
    return max_i


def binary_search(arr, x):
    low, high = 0, len(arr) - 1
    mid = 0
    mid_old = 0
    while high >= low:
        mid = (high + low) // 2
        if arr[mid] == x:
            return None
        elif arr[mid] > x:
            high = mid
        else:
            low = mid
        if mid_old == high + low:
            break
        mid_old = high + low
    return mid + 1


def douglas_peucker(points):
    iters = 20
    all_points = len(points)
    min_weight = iters * len(points) // 15
    point_list = [0, len(points) - 1]
    weights = [0]
    new_point_list = [0, len(points) - 1]
    for a in range(iters):
        for i in range(1, len(point_list)):
            point_ind = (point_list[i - 1] + point_list[i]) // 2
            # print(3**(3-8*2*(abs(point_ind-all_points//2))/all_points))
            add_point = find_farthest(list(points[point_list[i - 1]:point_list[i]]), 2)
            if not add_point:
                weights[i - 1] += point_list[i] - point_list[i - 1]
                continue
            add_point += point_list[i - 1]
            adress = binary_search(new_point_list, add_point)
            if not adress:
                continue
            weights.insert(adress, 0)
            new_point_list.insert(adress, add_point)
        point_list = new_point_list[:]
    out_points = []
    for i in range(len(weights)):
        if weights[i] > min_weight:
            if not out_points:
                out_points.append(point_list[i])
            out_points.append(point_list[i + 1])
    return out_points, weights


def find_corners_from_hough(object):
    lines = hough_transform_dec(object)
    f1, p1, val1 = lines[0]
    try:
        a1 = math.tan(f1)
        b1 = (p1 / math.sin(math.pi / 2 - f1))
        f2, p2, val2 = lines[1]
        a2 = math.tan(f2)
        b2 = (p2 / math.sin(math.pi / 2 - f2))
        x = (b2 - b1) / (a1 - a2)
        y = x * a1 + b1
        x1 = np.linspace(-10, 10, 100)
        y1 = a1 * x1 + b1
        # pyplot.plot(x1, y1, label='line1')

        x2 = np.linspace(-10, 10, 100)
        y2 = a2 * x2 + b2
        # pyplot.plot(x2, y2, label='line2')
        return [x, y]
    except:
        return None

def find_corners(object, approx_points_ind):
    connection_coords = [[object[p][0], object[p][1]] for p in approx_points_ind]
    if not connection_coords:
        return None
    b_prev = None
    b_curr = 0
    a_prev = None
    ang_curr = None
    ang_prev = None
    offset = 0.1
    corner = []
    corner_lines = []
    for i in range(1, len(connection_coords)):

        x1, y1 = connection_coords[i-1]
        x2, y2 = connection_coords[i]
        a_curr = 0
        if x1 == x2 or y1 == y2:
            #print("vert")
            pass
        else:
            a_curr = (y1 - y2) / (x1 - x2)
            b_curr = y1 - (y1 - y2) / (x1 - x2) * x1
        if a_prev:
            pass
        ang_curr = math.atan2((y1 - y2), (x1 - x2))
        if a_prev and abs(math.atan(a_curr)+math.atan(1/a_prev)) < offset:
            corner.append([object[approx_points_ind[i-2]:approx_points_ind[i]], approx_points_ind[i-1]-approx_points_ind[i-2]])
            corner_lines.append(([ang_prev, b_prev], [ang_curr, b_curr], object[approx_points_ind[i-1]], connection_coords[i-2:i+1]))
        a_prev = a_curr
        b_prev = b_curr
        ang_prev = ang_curr
    if corner:
        return corner, corner_lines
    else:
        return None, None





all_detected_corners_line = []
def check_existing_corners_by_lines(object):
    object = np.array(object)
    approx_points_ind, _ = douglas_peucker(object)
    _, corners = find_corners(object, approx_points_ind)
    if not corners:
        return None
    max_offset = 40
    min_offset = 10
    for c in range(len(corners)):
        line1, line2, corner, coords = corners[c]
        print(line1, line2)
        for i in range(len(all_detected_corners_line)):
            xof = (corner[0] - all_detected_corners_line[i][2][0]) ** 2
            yof = (corner[1] - all_detected_corners_line[i][2][1]) ** 2
            if max_offset > xof + yof > min_offset:
                line1_f, line2_f, corner_f, coords_f = all_detected_corners_line[i]
                #pyplot.axline((0, line1[1]), slope=math.tan(line1[0]), color="black", linestyle=(0, (5, 5)))
                #pyplot.axline((0, line2[1]), slope=math.tan(line2[0]), color="black", linestyle=(0, (5, 5)))
                pyplot.plot([coords[0][0], coords[1][0], coords[2][0]], [coords[0][1], coords[1][1], coords[2][1]], label='curr', linestyle="solid")
                pyplot.plot([coords_f[0][0], coords_f[1][0], coords_f[2][0]], [coords_f[0][1], coords_f[1][1], coords_f[2][1]], label='node', linestyle="solid")
                pos_err = corner-corner_f
                if line1[0] - line1_f[0] < line1[0] - line2_f[0]:
                    rot_err = (line1[0] - line1_f[0] + line2[0] - line2_f[0]) / 2
                else:
                    rot_err = (line1[0] - line2_f[0] + line2[0] - line1_f[0]) / 2
                print(pos_err, math.degrees(rot_err),math.degrees(line1[0]))
                pyplot.axline((0, line1[1]), slope=math.tan(line1[0]), color="black", linestyle=(0, (5, 5)))
                #pyplot.axline((0, line2[1]), slope=math.tan(line2[0]), color="black", linestyle=(0, (5, 5)))



                pyplot.axis('equal')
                pyplot.legend(numpoints=1)
                pyplot.show()
                return i

        all_detected_corners_line.append(corners[c])

def check_existing_corners(object):
    object = np.array(object)
    approx_points_ind, _ = douglas_peucker(object)
    corners, _ = find_corners(object, approx_points_ind)
    if not corners:
        return None
    max_offset = 40
    min_offset = 10
    for c in range(len(corners)):
        corner_rel_ind = corners[c][1]
        corner = corners[c][0][corner_rel_ind]
        for i in range(len(all_detected_corners)):
            xof = (corner[0]-all_detected_corners[i][0][0])**2
            yof = (corner[1]-all_detected_corners[i][0][1])**2
            if max_offset > xof+yof > min_offset:
                #icp here
                icp_out = icp(corners[c][0], all_detected_corners[i][1], corner_rel_ind, all_detected_corners[i][2])
                print(icp_out[-1])
                if True:
                    pyplot.plot([p[0] for p in corners[c][0]], [p[1] for p in corners[c][0]], 'o', label='points 2')

                    # to homogeneous
                    converted = np.ones((corners[c][0].shape[1] + 1, corners[c][0].shape[0]))
                    converted[:corners[c][0].shape[1], :] = np.copy(corners[c][0].T)
                    # transform
                    converted = np.dot(icp_out[0], converted)
                    # back from homogeneous to cartesian
                    converted = np.array(converted[:converted.shape[1], :]).T
                    pyplot.plot([p[0] for p in converted], [p[1] for p in converted], 'o', label='converted')
                    pyplot.plot([p[0] for p in all_detected_corners[i][1]], [p[1] for p in all_detected_corners[i][1]], '.', label='points 1')

                    pyplot.axis('equal')
                    pyplot.legend(numpoints=1)
                    pyplot.show()

                return i
        all_detected_corners.append([corner, np.array(corners[c][0]), corner_rel_ind])
    return None

def draw_line(a, b, name="line"):
    x = np.linspace(-10, 10, 100)
    y = a * x + b
    pyplot.plot(x, y, label=name)


#inp1 = open_log()
#points1 = np.array(conv_cil_to_dec(inp1))
# points2 = np.array(conv_cil_to_dec(inp2))
# icp_out = icp(points1, points2)
# print(icp_out[0], icp_out[2], icp_out[3])

# pyplot.plot([p[0] for p in points1], [p[1] for p in points1], '.', label='points 1')
# pyplot.plot([p[0] for p in points2], [p[1] for p in points2], 'o', label='points 2')

# to homogeneous
# converted = np.ones((points1.shape[1] + 1, points1.shape[0]))
# converted[:points1.shape[1], :] = np.copy(points1.T)
# transform
# converted = np.dot(icp_out[0], converted)
# back from homogeneous to cartesian
# converted = np.array(converted[:converted.shape[1], :]).T
# pyplot.plot([p[0] for p in converted], [p[1] for p in converted], 'o', label='converted')


# for i in range(len(objects)):
#    obj_conv = conv_cil_to_dec(inp1, objects[i])
#    pyplot.plot([p[0] for p in obj_conv], [p[1] for p in obj_conv], 'o', label='object {}'.format(i))
# pyplot.plot([p[0] for p in points1], [p[1] for p in points1], '.', label='points 1')

# object_coords = list(points1[i] for i in objects[0])


#object_coords = split_objects(inp1)

#print(len(object_coords))
#for object in object_coords:
#    approx_points, _ = douglas_peucker(object)
#    print(approx_points)
#    connection_coords = [object[p] for p in approx_points[1:-1]]
#    pyplot.plot([p[0] for p in object], [p[1] for p in object], 'o', label='object')
#    pyplot.plot([p[0] for p in connection_coords], [p[1] for p in connection_coords], 'o', label='object')
#    pyplot.plot([object[p][0] for p in approx_points], [object[p][1] for p in approx_points], label='object')
# pyplot.plot(corner_coords[0], corner_coords[1], 'o', label='corner')
# print(corner)


# a, corner_id = nearest_neighbor(object_coords, [corner])
# print(a, corner_id)
# pyplot.plot(*corner, 'o', label='corner')

#pyplot.plot(*scle_to_arr(*inp1[0][:-1:]), 'o', label='robot')
#pyplot.axis('equal')
#pyplot.legend(numpoints=1)
#pyplot.show()
