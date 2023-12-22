from typing import Tuple, List
import pdb
import math
import numpy as np

MAP_MIN_X = -4.0
MAP_MIN_Y = -4.0
MAP_MAX_X = 4.0
MAP_MAX_Y = 4.0
RESOLUTION = 0.1
RADIUS = 0.13


def build_map(obstacles: List) -> np.ndarray:
    widthX = math.ceil((MAP_MAX_X - MAP_MIN_X) / RESOLUTION)
    widthY = math.ceil((MAP_MAX_Y - MAP_MIN_Y) / RESOLUTION)

    map = np.zeros((widthY, widthX))

    for obstacle in obstacles:
        idx = math.ceil((obstacle[0] - MAP_MIN_X) / RESOLUTION)
        idy = math.ceil((obstacle[1] - MAP_MIN_X) / RESOLUTION)
        map[idy, idx] = 1.0

    return map


def check_node(px: float, py: float, map: np.ndarray) -> bool:
    widthX = math.ceil((MAP_MAX_X - MAP_MIN_X) / RESOLUTION)
    widthY = math.ceil((MAP_MAX_Y - MAP_MIN_Y) / RESOLUTION)

    if px < MAP_MIN_X:
        return False
    if px > MAP_MAX_X:
        return False
    if py < MAP_MIN_Y:
        return False
    if py > MAP_MAX_Y:
        return False

    for i in range(widthX):
        for j in range(widthY):
            x = MAP_MIN_X + i * RESOLUTION
            y = MAP_MIN_Y + j * RESOLUTION
            if (x - px) ** 2 + (y - py) ** 2 < RADIUS**2 and map[i, j] == 1.0:
                return False
    return True


def get_idx(
    x: float,
    y: float,
) -> Tuple[int, int, int]:
    idx = math.ceil((x - MAP_MIN_X) / RESOLUTION)
    idy = math.ceil((y - MAP_MIN_Y) / RESOLUTION)
    seq = idy * math.ceil((MAP_MAX_Y - MAP_MIN_Y) / RESOLUTION) + idx

    return int(idx), int(idy), int(seq)


def plan(ox: float, oy: float, gx: float, gy: float, obstacles: List) -> List:
    res_set = {}
    opn_set = {}

    path = []

    res_set[get_idx(ox, oy)[2]] = [ox, oy, 0.0, -1]
    opn_set[get_idx(gx, gy)[2]] = [gx, gy, 1e5, -1]

    map = build_map(obstacles)

    while True:
        # loop through result set
        for idx, node in res_set.items():
            # neighbor nodes around current result set
            for x in [-1, 0, 1]:
                for y in [-1, 0, 1]:
                    px = node[0] + x * RESOLUTION
                    py = node[1] + y * RESOLUTION

                    _, _, cur_idx = get_idx(px, py)

                    # check node if its already in res set
                    if cur_idx in res_set:
                        continue

                    # check node if its within map and not too closed to obstacles
                    if not check_node(px, py, map):
                        continue

                    # check node if its already in open set
                    if cur_idx not in opn_set:
                        opn_set[cur_idx] = [px, py, 1e5, -1]

                    if (
                        math.sqrt(x * x + y * y) * RESOLUTION + node[2]
                        < opn_set[cur_idx][2]
                    ):
                        opn_set[cur_idx][2] = (
                            math.sqrt(x * x + y * y) * RESOLUTION + node[2]
                        )
                        opn_set[cur_idx][3] = idx

        cur_node = -1
        cur_val = 1e5

        # loop through open set
        for idx, node in opn_set.items():
            if node[2] < cur_val:
                cur_node = idx
                cur_val = node[2]
                print(f"cur best node is: {cur_node}||{cur_val}")

        if cur_node == get_idx(gx, gy)[2]:
            while res_set[cur_node][-1] != -1:
                path.append([res_set[cur_node][0], res_set[cur_node][1]])
                cur_node = res_set[cur_node][-1]

            print(path)
            return path

        # add cur node to result set
        # pdb.set_trace()
        if cur_node == -1:
            pdb.set_trace()
        res_set[cur_node] = opn_set[cur_node]

        # remove cur node from open set
        del opn_set[cur_node]


if __name__ == "__main__":
    sx = 2.3
    sy = 1.8
    gx = 2.8
    gy = 2.3

    obstacles = [
        [1, 0.7],
        [2, 2],
        [2.1, 2],
        [2.2, 2],
        [2.3, 2],
        [2.4, 2],
        [2.5, 2],
        [2.5, 1.9],
        [2.5, 1.8],
        [2.5, 1.7],
        [2.5, 1.6],
        [2.5, 1.5],
        [2.8, 3.2],
        [2.9, 3.1],
        [3, 3],
        [3.1, 2.9],
        [3.2, 2.8],
    ]

    plan(sx, sy, gx, gy, obstacles)
