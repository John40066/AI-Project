import cv2
import numpy as np
import json
from tqdm import tqdm


def inside(r, c, h, w):
    if (0 <= r and r < h) and (0 <= c and c < w):
        return True
    return False


def isRed(r, c, img):
    if not (img[r][c][0] == 255):
        return True
    return False


def findRD(r, c, h, w, img, vis):
    tr, tc = r, c
    while(inside(r, c, h, w) and isRed(r, c, img)):
        vis[r][c] = True
        r += 1
    r -= 1
    while(inside(r, c, h, w) and isRed(r, c, img)):
        vis[r][c] = True
        c += 1
    c -= 1
    while(inside(tr, tc, h, w) and isRed(tr, tc, img)):
        vis[tr][tc] = True
        tc += 1
    tc -= 1
    while(inside(tr, tc, h, w) and isRed(tr, tc, img)):
        vis[tr][tc] = True
        tr += 1
    tr -= 1

    return (r, c)


def main():
    for num in tqdm(range(0, 67)):
        img = cv2.imread('./Data/imageRects/' + str(num) + '.png')
        h, w = len(img), len(img[0])
        vis = np.zeros([h, w], dtype=bool)
        rects = []
        for i in range(h):
            for j in range(w):
                if isRed(i, j, img) and vis[i][j] == False:
                    rects.append([(i, j)])
                    rects[-1].append(findRD(i, j, h, w, img, vis))
        file = open("./Rects/rects_" + str(num) + ".json", "w")
        json.dump(rects, file)
        file.close()

    # canvas = np.zeros([h, w, 3], dtype="uint8")
    # canvas += 255
    # for i in range(len(rects)):
    #     cv2.rectangle(canvas, rects[i][0], rects[i][1], [0, 0, 255], -1)
    # cv2.imshow("CANVAS", canvas)

    # cv2.imshow("IMG", img)
    # cv2.waitKey()


if __name__ == '__main__':
    main()
