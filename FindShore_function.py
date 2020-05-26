import numpy as np

def isShore(land, y, x):
    n = 1

    if y == 0 and land[y, x]:
        n += 5
        n += land[1, x]
        n += np.roll(land, -1, axis=1)[1, x]
    elif y == 180 and land[y, x]:
        n += 5
        n += np.roll(land, 1, axis=1)[179, x]
        n += land[179, x]
        n += np.roll(land, -1, axis=1)[179, x]

    else:
        if land[y, x]:
            n += np.roll(land, 1, axis=1)[y - 1, x]
            n += land[y - 1, x]
            n += np.roll(land, -1, axis=1)[y - 1, x]

            n += np.roll(land, 1, axis=1)[y, x]
            n += np.roll(land, -1, axis=1)[y, x]

            n += np.roll(land, 1, axis=1)[y + 1, x]
            n += land[y + 1, x]
            n += np.roll(land, -1, axis=1)[y + 1, x]

    if n == 9:
        return False
    else:
        return True
