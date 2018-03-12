import numpy as np

def bds_vote(snn, rnn, snnd, rnnd, src, patchsize=5):
    """
    Reconstructs an image or feature map by bidirectionaly
    similarity voting
    """

    src_height = src.shape[1]
    src_width = src.shape[2]
    channels = src.shape[0]

    dest = np.zeros(src.shape)

    dest_height = src.shape[1]
    dest_width = src.shape[2]

    pmax = patchsize // 2

    weights = np.zeros((dest_height, dest_width))

    print(snn.shape)
    ws = 1 / ((snn.shape[1] - patchsize + 1) * (snn.shape[2] - patchsize + 1))
    wr = 1 / ((rnn.shape[1] - patchsize + 1) * (rnn.shape[2] - patchsize + 1))
    # coherence
    # The S->R forward NNF enforces coherence
    for i in range(src_width):
        for j in range(src_height):

            px = snn[0, i, j]
            py = snn[1, i, j]

            for dy in range(j-pmax, j+pmax):
                if j + dy < 0:
                    continue
                if j + dy >= dest_height:
                    break
                if py + dy < 0:
                    continue
                if py + dy >= src_height:
                    break
                for dx in range(i-pmax, i+pmax):
                    if i + dx < 0:
                        continue
                    if i + dx >= dest_width:
                        break
                    if px + dx < 0:
                        continue
                    if px + dx >= src_width:
                        break
                    for ch in range(channels):
                        dest[ch, dy + j, dx + i] += ws * src[ch, py + dy, px + dx]
                    weights[dy + j, dx + i] += ws


    # completeness
    # The R->S backward NNF enforces completeness
    for i in range(src_width):
        for j in range(src_height):

            px = rnn[0, i, j]
            py = rnn[1, i, j]

            for dy in range(j-pmax, j+pmax):
                if j + dy < 0:
                    continue
                if j + dy >= src_height:
                    break
                if py + dy < 0:
                    continue
                if py + dy >= dest_height:
                    break
                for dx in range(i-pmax, i+pmax):
                    if i + dx < 0:
                        continue
                    if i + dx >= src_width:
                        break
                    if px + dx < 0:
                        continue
                    if px + dx >= dest_width:
                        break
                    for ch in range(channels):
                        dest[ch, py + dy, px + dx] += wr * src[ch, dy + j, dx + i]
                    weights[py + dy, px + dx] += wr

    for x in range(dest_width):
        for y in range(dest_height):
            s = 1 if weights[y, x] == 0 else (1 / weights[y, x])
            for ch in range(channels):
                dest[ch, y, x] *= s

    return dest