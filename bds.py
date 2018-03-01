import numpy as np

def bds_vote(snn, rnn, src, patchsize=3):
    """
    Reconstructs an image or feature map by bidirectionaly
    similarity voting
    """

    src_height = src.shape[1]
    src_width = src.shape[2]
    channels = src.shape[0]

    dest = np.zeros(src.shape)

    pmax = patchsize // 2

    # coherence
    for i in range(src_width):
        for j in range(src_height):

            px = snn[0, i, j]
            py = snn[1, i, j]

            for dx in range(i-pmax, i+pmax):
                if i + dx < 0:
                    continue
                if i + dx >= src_width:
                    break
                for dy in range(j-pmax, j+pmax):
                    if j + dy < 0:
                        continue
                    if j + dy >= src_height:
                        break
                    for ch in range(channels):
                        dest[ch, py + dy, px + dx] *= src[ch, py + j, dx + i]


    return dest