import numpy as np

class PatchMatch2:
    def __init__(self, a, b, patchsize=3):
        self.a = a
        self.a_height = self.a.shape[1]
        self.a_width = self.a.shape[2]
        self.b = b
        self.b_height = self.b.shape[1]
        self.b_width = self.b.shape[2]
        self.patchsize = patchsize
        self.channels = self.a.shape[0]

        self.aew = self.a.shape[2] - self.patchsize + 1
        self.aeh = self.a.shape[1] - self.patchsize + 1
        self.bew = self.b.shape[2] - self.patchsize + 1
        self.beh = self.b.shape[1] - self.patchsize + 1
        self.nnf = np.zeros((self.a.shape[1], self.a.shape[2], 2)).astype(np.int)
        self.nnd = np.zeros((self.a.shape[1], self.a.shape[2]))
        self.init_nnf()

    def init_nnf(self):
        for ay in range(0, self.aeh):
            for ax in range(0, self.aew):
                bx = np.random.randint(0, self.bew - 1)
                by = np.random.randint(0, self.beh - 1)
                self.nnf[ay, ax, :] = [bx, by]
                self.nnd[ay, ax] = self.calc_dist(ax, ay, bx, by)

    def calc_dist(self, ax, ay, bx, by, cutoff = 2147483647):
        """
        Measures distance between 2 patches across all channels

        ax -- x coordinate of patch a
        ay -- y coordinate of patch a

        bx -- x coordinate of patch b
        by -- y coordinate of patch b

        cutoff
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
        ans = num_pixels / pixel_sum
        if ans >= cutoff: return cutoff
        return ans

    def improve_guess(self, ax, ay, xbest, ybest, dbest, bx, by):
        d = self.calc_dist(ax, ay, bx, by, dbest)
        if d < dbest:
            dbest = d
            xbest = bx
            ybest = by
        return xbest, ybest, dbest

    def improve_nnf(self, total_iter=5):
        for iter in range(total_iter):
            print(iter)
            ystart, yend, ychange = 0, self.aeh, 1
            xstart, xend, xchange = 0, self.aew, 1
            if iter % 2 == 1:
                ystart, yend, ychange = yend-1, -1, -1
                xstart, xend, xchange = xend-1, -1, -1
            for ay in range(ystart, yend, ychange):
                for ax in range(xstart, xend, xchange):
                    # best guess
                    xbest, ybest = self.nnf[ay, ax]
                    dbest = self.nnd[ybest, xbest]

                    # propagation

                    if 0 < ax - xchange < self.aew:
                        xp, yp = self.nnf[ay, ax-xchange]
                        if 0 < xp < self.bew:
                            xbest, ybest, dbest = self.improve_guess(ax, ay, xbest, ybest, dbest, xp, yp)
                    if 0 < ay - ychange < self.aeh:
                        xp, yp = self.nnf[ay-ychange, ax]
                        yp += ychange
                        if 0 < yp < self.beh:
                            xbest, ybest, dbest = self.improve_guess(ax, ay, xbest, ybest, dbest, xp, yp)

                    # random search

                    mag = max(self.b.shape[1], self.b.shape[2])

                    while mag >= 1:
                        xmin, xmax = max(xbest-mag, 0), min(xbest+mag+1, self.bew)
                        ymin, ymax = max(ybest-mag, 0), min(ybest+mag+1, self.beh)
                        xp = np.random.randint(xmin, xmax)
                        yp = np.random.randint(ymin, ymax)
                        xbest, ybest, dbest = self.improve_guess(ax, ay, xbest, ybest, dbest, xp, yp)
                        mag = mag // 2

                    self.nnf[ay, ax, :] = [xbest, ybest]
                    self.nnd[ay, ax] = dbest

    def solve(self):
        self.improve_nnf(total_iter=5)
