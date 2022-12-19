from import_libs import *
def homogen_matrix_from_pos(pos, for_lidar=False):
    x, y, rot = pos
    cr = math.cos(rot)
    sr = math.sin(rot)
    if not for_lidar:
        return np.array([[cr, -sr, x],
                         [sr, cr, y],
                         [0, 0, 1]])
    robot_rad = 0.3
    X = np.array([[1, 0, x],
                  [0, 1, y],
                  [0, 0, 1]])

    f = np.array([[1, 0, robot_rad],
                  [0, 1, 0],
                  [0, 0, 1]])

    rot = np.array([[cr, -sr, 0],
                    [sr, cr, 0],
                    [0, 0, 1]])

    X = np.dot(X, np.dot(rot, f))
    return X


def undo_lidar(mat):
    cr = mat[0, 0]
    sr = mat[1, 0]
    robot_rad = 0.3
    f = np.array([[1, 0, robot_rad],
                  [0, 1, 0],
                  [0, 0, 1]])

    rot = np.array([[cr, -sr, 0],
                    [sr, cr, 0],
                    [0, 0, 1]])

    mat = np.dot(mat, np.linalg.inv(np.dot(rot, f)))
    mat[:2, :2] = [[cr, -sr], [sr, cr]]
    return mat


def pos_vector_from_homogen_matrix(mat):
    aco = math.acos(max(-1, min(1, mat[0, 0])))
    asi = math.asin(max(-1, min(1, mat[1, 0])))
    asi_sign = -1 + 2 * (asi > 0)
    return np.array([*mat[:2, 2], aco * (asi_sign)])