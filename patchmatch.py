
# https://github.com/Aixile/PatchMatch/blob/master/patchmatch.py

import numpy as np

class NNF:
    def __init__(self, img_a, img_b, boxsize=7):
        self.A = img_a
        self.B = img_b
        self.boxsize = boxsize//2
        #return correspondences in three channels: x_coordinates, y_coordinates, offsets
        self.nnf = np.zeros((2, self.A.shape[0], self.A.shape[1])).astype(np.int)
        self.nnf_D = np.zeros((self.A.shape[0], self.A.shape[1]))
        self.init_nnf()

    def init_nnf(self):
        self.nnf[0] = np.random.randint(self.B.shape[0], size=(self.A.shape[0], self.A.shape[1]))
        self.nnf[1] = np.random.randint(self.B.shape[1], size=(self.A.shape[0], self.A.shape[1]))
        self.nnf = self.nnf.transpose((1, 2 ,0))
        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):
                pos = self.nnf[i,j]
                self.nnf_D[i,j] = self.cal_dist(i, j, pos[0], pos[1])

    def cal_dist(self, ai ,aj, bi, bj):
        dx0 = dy0 = self.boxsize//2
        dx1 = dy1 = self.boxsize//2 + 1
        dx0 = min(ai, bi, dx0)
        dx1 = min(self.A.shape[0]-ai, self.B.shape[0]-bi, dx1)
        dy0 = min(aj, bj, dy0)
        dy1 = min(self.A.shape[1]-aj, self.B.shape[1]-bj, dy1)
        return np.sum((self.A[ai-dx0:ai+dx1, aj-dy0:aj+dy1]-self.B[bi-dx0:bi+dx1, bj-dy0:bj+dy1])**2) / (dx1+dx0) / (dy1+dy0)
        #self.nnf_D[i,j] =np.sum((self.A[i-dx0:i+dx1, j-dy0:j+dy1]-self.B[i-dx0:i+dx1, j-dy0:j+dy1])**2)
    #def improve_guess(self,):


    def improve_nnf(self, total_iter=5):
        for iter in range(total_iter):
            print(iter)
            for i in range(self.A.shape[0]):
                for j in range(self.A.shape[1]):

                    pos = self.nnf[i,j]
                    x, y = pos[0], pos[1]
                    bestx, besty, bestd = x, y, self.nnf_D[i,j]

                    for k in reversed(range(4)):
                        d = 2**k
                        if i-d >= 0:
                            rx, ry = self.nnf[i-d, j][0] + d, self.nnf[i-d, j][1]
                            if rx < self.B.shape[0]:
                                val = self.cal_dist(i, j, rx, ry)
                                if val < bestd:
                                    bestx, besty, bestd = rx, ry, val

                        if j-d >= 0:
                            rx, ry = self.nnf[i, j-d][0], self.nnf[i, j-d][1] + d
                            if ry < self.B.shape[1]:
                                val = self.cal_dist(i, j, rx, ry)
                                if val < bestd:
                                    bestx, besty, bestd = rx, ry, val

                        if i+d < self.A.shape[0]:
                            rx, ry = self.nnf[i+d, j][0]-d, self.nnf[i+d, j][1]
                            if rx >= 0:
                                val = self.cal_dist(i, j, rx, ry)
                                if val < bestd:
                                    bestx, besty, bestd = rx, ry, val

                        if j+d < self.A.shape[1]:
                            rx, ry = self.nnf[i, j+d][0], self.nnf[i, j+d][1] - d
                            if ry >= 0:
                                val = self.cal_dist(i, j, rx, ry)
                                if val < bestd:
                                    bestx, besty, bestd = rx, ry, val


                    rand_d = min(self.B.shape[0]//2, self.B.shape[1]//2)
                    while rand_d > 0:
                        try:
                            xmin = max(bestx - rand_d, 0)
                            xmax = min(bestx + rand_d, self.B.shape[0])
                            ymin = max(besty - rand_d, 0)
                            ymax = min(besty + rand_d, self.B.shape[1])
                        #print(xmin, xmax)
                            rx = np.random.randint(xmin, xmax)
                            ry = np.random.randint(ymin, ymax)
                            val = self.cal_dist(i, j, rx, ry)
                            if val < bestd:
                                bestx, besty, bestd = rx, ry, val
                        except:
                            print(rand_d)
                            print(xmin, xmax)
                            print(ymin, ymax)
                            print(bestx, besty)
                            print(self.B.shape)
                        rand_d = rand_d // 2

                    self.nnf[i, j] = [bestx, besty]
                    self.nnf_D[i, j] = bestd

    def solve(self):
        self.improve_nnf(total_iter=5)

    def reconstruct(self):
        ans = np.zeros_like(self.A)
        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):
                pos = self.nnf[i,j]
                ans[i,j] = self.B[pos[0], pos[1]]
        return ans
