from pytorchvideo.models.hub.x3d import x3d_m

def build_3d_backbone():
    base = x3d_m(pretrained=True, progress=True)
    return base


model = build_3d_backbone()
print(1)