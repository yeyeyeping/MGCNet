import os
import sys
from pathlib import Path
from albumentations import functional
import numpy as np
import torch
from train import read_yml
from reader import reader
from skimage.io import imsave
import numpy as np
import sys
import os
import json
import trainer
from metric import metrics

mean = (0.485, 0.456, 0.406)

std = (0.229, 0.224, 0.225)


assert len(sys.argv) - \
    1 == 6, "cfg_path, img_folder, ckpath, out_dir,input_size,class_num"
cfg_path, img_folder, ckpath, out_dir, input_size, class_num = sys.argv[
    1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(
        sys.argv[6])

assert os.path.exists(img_folder) and os.path.exists(ckpath)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device(
    "cpu")

os.makedirs(out_dir, exist_ok=True)

cfg = read_yml(cfg_path)

trainer_obj = trainer.__dict__[cfg["Training"]["Trainer"]](cfg, logger=None)

model = trainer_obj.model.to(device)

model.load_state_dict(
    torch.load(ckpath, map_location=device)["model_state_dict"])
model.eval()

val_dice, val_assd = [], []
val_json = {}
case = []

with torch.no_grad():
    for g, gt in zip((Path(img_folder) / "images").iterdir(),
                     (Path(img_folder) / "mask").iterdir()):
        print(f"predicting {str(g)}")
        img_npy = reader(g)().read_image(g)
        mask_npy = reader(g)().read_image(gt)

        img_npy = functional.normalize(img_npy, mean, std, max_pixel_value=1)
        img_npy = np.ascontiguousarray(img_npy.transpose(2, 0, 1))[None]

        img = torch.from_numpy(img_npy).to(device, torch.float32)
        output, _ = model(img)
        output = torch.stack(output).mean(0)

        # output = model(img)
        batch_pred_mask = output.argmax(axis=1)[0]
        imsave(os.path.join(out_dir,
                            str(g.name)[:-4] + ".png"),
               batch_pred_mask.cpu().numpy().astype(np.uint8),
               check_contrast=False)

        class_dice, class_assd = metrics(batch_pred_mask,
                                         mask_npy,
                                         class_num=class_num)
        val_assd.append(class_assd)
        val_dice.append(class_dice)
        case.append({
            "filename":
            g.name,
            "class_dice":
            class_dice,
            "mean_dice":
            round(sum(class_dice[1:]) / (len(class_dice) - 1), 4),
            "class_asdd":
            class_assd,
            "mean_assd":
            round(sum(class_assd[1:]) / (len(class_dice) - 1), 4)
        })

d, a = np.array(val_dice), np.array(val_assd)
m_assd, m_dice = a[:, 1:].mean(1), d[:, 1:].mean(1)

val_json["metrics"] = {
    "assd_class": {
        str(i):
        f"{np.round(np.mean(a[:, i]), 4)}±{np.round(np.std(a[:, i]), 4)}"
        for i in range(class_num)
    },
    "case_assd": {
        "mean": np.round(np.mean(m_assd), 4),
        "std": np.round(np.std(m_assd), 4),
    },
    "dice_class": {
        str(i):
        f"{np.round(np.mean(d[:, i]), 4)}±{np.round(np.std(d[:, i]), 4)}"
        for i in range(class_num)
    },
    "case_dice": {
        "mean": np.round(np.mean(m_dice), 4),
        "std": np.round(np.std(m_dice), 4)
    },
}
sort_case = sorted(case, key=lambda x: x["class_dice"])

val_json["case"] = sort_case
outpath = f"{os.path.dirname(ckpath)}/../evaluation.json"
with open(outpath, "w") as fp:
    json.dump(val_json, fp, indent=4)
