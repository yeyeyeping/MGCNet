import numpy as np
import torch
import os
from train import read_yml
import sys
from pathlib import Path
import SimpleITK as sitk
from scipy.ndimage import zoom
import trainer
import json
from metric import metrics

assert len(sys.argv) - 1 == 4, "cfg_path, img_folder, ckpath, out_dir"
cfg_path, img_folder, ckpath, out_dir = sys.argv[1], sys.argv[2], sys.argv[
    3], sys.argv[4]
class_num = 4
input_size = 192

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
    gt = Path(img_folder).glob("*_gt.nii.gz")
    for g in gt:
        print(f"predicting {g.name}")
        img_path = str(g)[:-10] + ".nii.gz"
        img = sitk.ReadImage(img_path)
        img_npy = sitk.GetArrayFromImage(img)

        gt_obj = sitk.ReadImage(str(g))
        gt_npy = sitk.GetArrayFromImage(gt_obj)

        *_, h, w = img_npy.shape

        # 强度标准化
        img_npy = np.asarray(
            (img_npy - img_npy.min()) / (img_npy.max() - img_npy.min()),
            dtype=np.float32)
        # slice标准化
        img_npy = (img_npy - img_npy.mean(axis=(1, 2), keepdims=True)
                   ) / img_npy.std(axis=(1, 2), keepdims=True)

        zoomed_img = zoom(img_npy[:, None],
                          (1, 1, input_size / h, input_size / w),
                          order=1,
                          mode='nearest')
        zoomed_img = torch.from_numpy(zoomed_img).cuda()
        output, _ = model(zoomed_img)
        output = torch.stack(output).mean(0)
        # output = model(zoomed_img)
        pred_volume = zoom(output.cpu().numpy(),
                           (1, 1, h / input_size, w / input_size),
                           order=1,
                           mode='nearest')
        batch_pred_mask = pred_volume.argmax(axis=1)
        out_obj = sitk.GetImageFromArray(batch_pred_mask)
        out_obj.CopyInformation(gt_obj)
        sitk.WriteImage(out_obj,
                        os.path.join(out_dir, os.path.basename(img_path)))

        class_dice, class_assd = metrics(batch_pred_mask,
                                         gt_npy,
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
            "class_assd":
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
