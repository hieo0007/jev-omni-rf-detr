import sys, time, platform
import torch
from PIL import Image
from rfdetr import RFDETRNano
from rfdetr.assets.coco_classes import COCO_CLASSES
print("python", platform.python_version(), "torch", torch.__version__, "cuda", torch.cuda.is_available())
t0=time.time(); model = RFDETRNano(); print(f"load_s={time.time()-t0:.1f}")
img = Image.open(sys.argv[1]).convert("RGB")
for i in range(4):
    t=time.time(); det = model.predict(img, threshold=0.5); print(f"run{i} {time.time()-t:.3f}s n={len(det)}")
for cid, conf, box in zip(det.class_id, det.confidence, det.xyxy):
    print(COCO_CLASSES[int(cid)], round(float(conf),3), [round(float(v),1) for v in box])
