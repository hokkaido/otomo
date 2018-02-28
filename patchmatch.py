import numpy as np

class PatchMatch:
    def __init__(self, a, b, patchsize=3):
        """
        Deep, generalized patch match

        Takes two feature maps with dimensions (channels, height, width).

        a -- ndarray with dimensions (channels, height, width)
        b -- ndarray with dimensions(channels, height, width)
        patchsize -- (default 3)
        """
        assert a.shape[0] == b.shape[0],  "channels don't match"
        self.a = a
        self.a_height = self.a.shape[1]
        self.a_width = self.a.shape[2]
        self.b = b
        self.b_height = self.b.shape[1]
        self.b_width = self.b.shape[2]
        self.patchsize = patchsize
        self.channels = self.a.shape[0]

        self.nnf = np.zeros((2, self.a.shape[2], self.a.shape[1])).astype(np.int)
        self.nnfd = np.zeros((self.a.shape[2], self.a.shape[1]))
        self.init_nnf()

    def init_nnf(self):
        self.nnf[0] = np.random.randint(self.b.shape[2], size=(self.a.shape[2], self.a.shape[1]))
        self.nnf[1] = np.random.randint(self.b.shape[1], size=(self.a.shape[2], self.a.shape[1]))

        for i in range(self.a.shape[2]):
            for j in range(self.a.shape[1]):
                pos = self.nnf[:,i,j]
                self.nnfd[i,j] = self.cal_dist(i, j, pos[0], pos[1])

    def cal_dist(self, ax, ay, bx, by):
        """
        Measures distance between 2 patches across all channels

        ax -- x coordinate of patch a
        ay -- y coordinate of patch a

        bx -- x coordinate of patch b
        by -- y coordinate of patch b
        """
        num_pixels = 0
        pixel_sum = 0
        dmax = self.patchsize // 2
        for dy in range(-dmax, dmax):
            for dx in range(-dmax, dmax):
                pixel_exists_in_a = (ay + dy) < self.a_height and (ay + dy) >= 0 and (ax + dx) < self.a_width and (ax + dx) >= 0
                pixel_exists_in_b = (by + dy) < self.b_height and (by + dy) >= 0 and (bx + dx) < self.b_width and (bx + dx) >= 0
                if pixel_exists_in_a and pixel_exists_in_b:
                    for dc in range(0, self.channels):
                        dp_tmp = self.a[dc, ay + dy, ax + dx] * self.b[dc, by + dy, bx + dx]
                        pixel_sum += dp_tmp

                num_pixels += 1
        return num_pixels / pixel_sum

    #def improve_guess(self, ax, ay, bx, by)

    def improve_nnf(self, total_iter=5):
        for iter in range(total_iter):
            print(iter)
            for i in range(self.a.shape[2]):
                for j in range(self.a.shape[1]):
                    pos = self.nnf[:, i, j]
                    x, y = pos[0], pos[1]
                    bestx, besty, bestd = x, y, self.nnfd[i, j]

                    for k in reversed(range(4)):
                        d = 2**k
                        if i-d >= 0:
                            rx, ry = self.nnf[0, i-d, j] + d, self.nnf[1, i-d, j]
                            if rx < self.b.shape[2]:
                                val = self.cal_dist(i, j, rx, ry)
                                if val < bestd:
                                    bestx, besty, bestd = rx, ry, val

                        if j-d >= 0:
                            rx, ry = self.nnf[0, i, j-d], self.nnf[1, i, j-d] + d
                            if ry < self.b.shape[1]:
                                val = self.cal_dist(i, j, rx, ry)
                                if val < bestd:
                                    bestx, besty, bestd = rx, ry, val

                        if i+d < self.a.shape[2]:
                            rx, ry = self.nnf[0, i+d, j] - d, self.nnf[1, i+d, j]
                            if rx >= 0:
                                val = self.cal_dist(i, j, rx, ry)
                                if val < bestd:
                                    bestx, besty, bestd = rx, ry, val

                        if j+d < self.a.shape[1]:
                            rx, ry = self.nnf[0, i, j+d], self.nnf[1, i, j+d] - d
                            if ry >= 0:
                                val = self.cal_dist(i, j, rx, ry)
                                if val < bestd:
                                    bestx, besty, bestd = rx, ry, val


                    rand_d = min(self.b.shape[1]//2, self.b.shape[2]//2)
                    while rand_d > 0:
                        try:
                            xmin = max(bestx - rand_d, 0)
                            xmax = min(bestx + rand_d, self.b.shape[2])
                            ymin = max(besty - rand_d, 0)
                            ymax = min(besty + rand_d, self.b.shape[1])
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
                            print(self.b.shape)
                        rand_d = rand_d // 2

                    self.nnf[:, i, j] = [bestx, besty]
                    self.nnfd[i, j] = bestd

    def solve(self):
        self.improve_nnf(total_iter=5)

    def reconstruct(self):
        ans = np.zeros_like(self.a)
        for i in range(self.a.shape[2]):
            for j in range(self.a.shape[1]):
                pos = self.nnf[:,i,j]
                ans[i,j] = self.b[:,pos[1], pos[0]]
        return ans
