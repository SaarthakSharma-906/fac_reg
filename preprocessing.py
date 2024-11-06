from PIL import Image

# Constants for EXIF orientation normalization
exif_orientation_tag = 0x0112
exif_transpose_sequences = [
    [], [],  # 0 and 1
    [Image.FLIP_LEFT_RIGHT],  # 2
    [Image.ROTATE_180],  # 3
    [Image.FLIP_TOP_BOTTOM],  # 4
    [Image.FLIP_LEFT_RIGHT, Image.ROTATE_90],  # 5
    [Image.ROTATE_270],  # 6
    [Image.FLIP_TOP_BOTTOM, Image.ROTATE_90],  # 7
    [Image.ROTATE_90],  # 8
]

class ExifOrientationNormalize(object):
    def __call__(self, img):
        if 'parsed_exif' in img.info and exif_orientation_tag in img.info['parsed_exif']:
            orientation = img.info['parsed_exif'][exif_orientation_tag]
            transposes = exif_transpose_sequences[orientation]
            for trans in transposes:
                img = img.transpose(trans)
        return img

class Whitening(object):
    def __call__(self, img):
        mean = img.mean()
        std = img.std()
        std_adj = std.clamp(min=1.0 / (float(img.numel()) ** 0.5))
        y = (img - mean) / std_adj
        return y
