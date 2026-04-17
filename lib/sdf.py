import numpy as np


def create_grid(resX, resY, resZ, b_min=np.array([0, 0, 0]), b_max=np.array([1, 1, 1]), transform=None):
    '''
    Create a dense grid of given resolution and bounding box
    :param resX: resolution along X axis
    :param resY: resolution along Y axis
    :param resZ: resolution along Z axis
    :param b_min: vec3 (x_min, y_min, z_min) bounding box corner
    :param b_max: vec3 (x_max, y_max, z_max) bounding box corner
    :return: [3, resX, resY, resZ] coordinates of the grid, and transform matrix from mesh index
    '''
    coords = np.mgrid[:resX, :resY, :resZ]
    coords = coords.reshape(3, -1)
    coords_matrix = np.eye(4)
    length = b_max - b_min
    coords_matrix[0, 0] = length[0] / resX
    coords_matrix[1, 1] = length[1] / resY
    coords_matrix[2, 2] = length[2] / resZ
    coords_matrix[0:3, 3] = b_min
    coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
    if transform is not None:
        coords = np.matmul(transform[:3, :3], coords) + transform[:3, 3:4]
        coords_matrix = np.matmul(transform, coords_matrix)
    coords = coords.reshape(3, resX, resY, resZ)
    return coords, coords_matrix


def batch_eval(points, eval_func, num_samples=512 * 512 * 512):
    num_pts = points.shape[1]
    sdf = np.zeros(num_pts)

    num_batches = num_pts // num_samples
    for i in range(num_batches):
        sdf[i * num_samples:i * num_samples + num_samples] = eval_func(
            points[:, i * num_samples:i * num_samples + num_samples])
    if num_pts % num_samples:
        sdf[num_batches * num_samples:] = eval_func(points[:, num_batches * num_samples:])

    return sdf


def eval_grid(coords, eval_func, num_samples=512 * 512 * 512):
    resolution = coords.shape[1:4]
    coords = coords.reshape([3, -1])
    sdf = batch_eval(coords, eval_func, num_samples=num_samples)
    return sdf.reshape(resolution)


def eval_grid_octree(coords, eval_func,
                     init_resolution=64, threshold=0.01,
                     num_samples=512 * 512 * 512):
    resolution = coords.shape[1:4]

    sdf = np.zeros(resolution)

    dirty = np.ones(resolution, dtype=bool)
    grid_mask = np.zeros(resolution, dtype=bool)

    reso = resolution[0] // init_resolution

    while reso > 0:
        # subdivide the grid
        grid_mask[0:resolution[0]:reso, 0:resolution[1]:reso, 0:resolution[2]:reso] = True
        # test samples in this iteration
        test_mask = np.logical_and(grid_mask, dirty)
        points = coords[:, test_mask]

        sdf[test_mask] = batch_eval(points, eval_func, num_samples=num_samples)
        dirty[test_mask] = False

        if reso <= 1:
            break

        # Vectorized interpolation: replaces triple nested Python loop
        # which was O(res^3) pure Python iterations — too slow at res=256
        xs = np.arange(0, resolution[0] - reso, reso)
        ys = np.arange(0, resolution[1] - reso, reso)
        zs = np.arange(0, resolution[2] - reso, reso)
        xg, yg, zg = np.meshgrid(xs, ys, zs, indexing='ij')
        xg, yg, zg = xg.ravel(), yg.ravel(), zg.ravel()

        center_dirty = dirty[xg + reso // 2, yg + reso // 2, zg + reso // 2]
        xg = xg[center_dirty]
        yg = yg[center_dirty]
        zg = zg[center_dirty]

        if len(xg) > 0:
            corners = np.stack([
                sdf[xg,        yg,        zg       ],
                sdf[xg,        yg,        zg + reso],
                sdf[xg,        yg + reso, zg       ],
                sdf[xg,        yg + reso, zg + reso],
                sdf[xg + reso, yg,        zg       ],
                sdf[xg + reso, yg,        zg + reso],
                sdf[xg + reso, yg + reso, zg       ],
                sdf[xg + reso, yg + reso, zg + reso],
            ], axis=1)
            v_min = corners.min(axis=1)
            v_max = corners.max(axis=1)
            uniform = (v_max - v_min) < threshold
            for i in np.where(uniform)[0]:
                x, y, z = xg[i], yg[i], zg[i]
                sdf[x:x + reso, y:y + reso, z:z + reso] = (v_min[i] + v_max[i]) / 2
                dirty[x:x + reso, y:y + reso, z:z + reso] = False

        reso //= 2

    return sdf.reshape(resolution)
