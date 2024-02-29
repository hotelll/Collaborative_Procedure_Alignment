
def frames_preprocess(frames):
    bs, c, h, w, num_clip = frames.size()
    frames = frames.permute(0, 4, 1, 2, 3)
    frames = frames.reshape(-1, c, h, w)

    return frames

def frames_preprocess_3d(frames):
    frames = frames.permute(0, 1, 4, 2, 3)

    return frames

def frames_preprocess_single(frames):
    c, h, w, num_clip = frames.size()
    frames = frames.permute(3, 0, 1, 2)

    return frames