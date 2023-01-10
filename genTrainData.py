import cv2
import numpy as np
import json
from tqdm import tqdm
from matplotlib import pyplot as plt


def HistCheck(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
    mx, total = 0, 0
    for i in range(len(hist)):
        if hist[i] > mx:
            mx = hist[i]
        total += hist[i]
    if(mx / total >= 0.95):
        return True
    return False


def inSide(x, l, r):
    if l <= x and x <= r:
        return True
    return False


def isX(y, x, sz, rects):
    for rect in rects:
        l, u, r, d = rect[0][1], rect[0][0], rect[1][1], rect[1][0]
        if inSide(x, l, r) or inSide(x+sz, l, r):
            if inSide(y, u, d) or inSide(y+sz, u, d):
                return True
    return False


def main():
    O_num, X_num = 21678, 17148
    sz = 100
    h_sz = sz // 2
    for num in tqdm(range(1, 67)):
        fp = open("./Rects/rects_" + str(num) + ".json")
        rects = json.load(fp)
        src1 = cv2.imread("./Data/image/" + str(num) + "_C.png")
        src2 = cv2.imread("./Data/image/" + str(num) + "_E.png")
        h, w = len(src1), len(src1[0])
        for i in range(h//h_sz - 2):
            for j in range(w//h_sz - 2):
                y, x = i * h_sz, j*h_sz
                c_src1 = src1[y:y+sz, x:x+sz]

                if HistCheck(c_src1):
                    continue

                l, r = max(x - sz, 0), min(x + sz+sz, w)
                u, d = max(y - sz, 0), min(y + sz+sz, h)
                c_src2 = src2[u:d, l:r]
                tmp = cv2.matchTemplate(
                    c_src1, c_src2, cv2.TM_CCOEFF_NORMED)
                _, score, _, pt = cv2.minMaxLoc(tmp)
                c_src2 = c_src2[pt[1]:pt[1]+sz, pt[0]:pt[0]+sz]
                if score < 0.9 and isX(y, x, sz, rects):
                    cv2.imwrite("./Data/X/"+str(X_num) + "c.png", c_src1)
                    cv2.imwrite("./Data/X/"+str(X_num) + "f.png", c_src2)
                    X_num += 1
                else:
                    cv2.imwrite("./Data/O/"+str(O_num) + "c.png", c_src1)
                    cv2.imwrite("./Data/O/"+str(O_num) + "f.png", c_src2)
                    O_num += 1
    print(O_num, X_num)


if __name__ == '__main__':
    main()
