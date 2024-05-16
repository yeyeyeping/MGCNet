import SimpleITK as sitk
import numpy as np


class AbstractReader:
    support = []

    def read_image(self, img_path):
        raise NotImplementedError

    @classmethod
    def is_support(cls, p: str):
        raise NotImplementedError


class NpyReader(AbstractReader):
    support = ["npy", "npz"]

    def read_image(self, img_path):
        return np.load(img_path)
    @classmethod
    def is_support(cls, p):
        p =str(p)
        if p.split(".")[-1] in cls.support:
            return True
        else:
            return False


class SimpleITKReader(AbstractReader):
    support = ["nii.gz"]

    def read_image(self, img_path):
        img_obj = sitk.ReadImage(img_path)
        return sitk.GetArrayFromImage(img_obj)

    @classmethod
    def is_support(cls, p: str):
        p =str(p)
        for ending in cls.support:
            if p.endswith(ending):
                return True
        return False


def reader(p):
    for v in __all__:
        if issubclass(v, AbstractReader):
            if v.is_support(p):
                return v
    else:
        raise f"Could not read {p}"


__all__ = [NpyReader, SimpleITKReader]

