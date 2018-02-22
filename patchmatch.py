import numpy as np

class PatchMatch:
    def __init__(self, a, b, patchsize = 3):
        """
        a -- ndarray with dimensions (channels, height, width)
        b -- ndarray with dimensions(channels, height, width)
        patchsize -- (default 3)
        """
        self.a = a
        self.b = b
        self.patchsize = patchsize
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

    def cal_dist(self, ax ,ay, bx, by):
        """Measures distance between 2 patches across all channels
        ax -- x coordinate of patch a
        ay -- y coordinate of patch a

        bx -- x coordinate of patch b
        by -- y coordinate of patch b
        """

        dmax = self.patchsize // 2
        for dy in range(-dmax, dmax):
            for dx in range(-dmax, dmax):
                pixel_exists_in_a = (ay + dy) < self.a_height and (ay + dy) >= 0 and (ax + dx) < self.a_width and (ax + dx) >= 0
                pixel_exists_in_b = (by + dy) < self.b_height and (by + dy) >= 0 and (bx + dx) < self.b_width and (bx + dx) >= 0
                if pixel_exists_in_a and pixel_exists_in_b:
                    for dc in range(0, self.channels):
                        dp_tmp = self.a[dc, ay + dy, ax + dx] * b[dc, by + dy, bx + dx]
                        pixel_sum -= dp_tmp
                        dp_tmp = a1[dc * a_slice + (ay + dy) * a_pitch + (ax + dx)] * b1[dc * b_slice + (by + dy) * b_pitch + (bx + dx)]
                        pixel_sum1 -= dp_tmp



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
