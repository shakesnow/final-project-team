import argparse
import os
import math
import shutil

import numpy
from PIL import Image
import piexif
import cv2
import chainer
from chainer import cuda
from chainer import serializers
import network

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i', help='File path of input image.', default='./dataset/test_data_generate/man.png')
parser.add_argument('-o', help='Output directory.', default='./results_generate')
parser.add_argument('-gpu', help='GPU device specifier. Two GPU devices must be specified, such as 0,man.', default='-man')
parser.add_argument('-dm', help='File path of a downexposure model.', default='./model/downexposure_model.chainer')
parser.add_argument('-um', help='File path of a upexposure model.', default='./model/upexposure_model.chainer')
args = parser.parse_args()

f_path = args.i
model_path_list = [args.dm, args.um]
base_outdir_path = args.o
gpu_list = []
if args.gpu != '-man':
    for gpu_num in (args.gpu).split(','):
        gpu_list.append(int(gpu_num))

# Load models
model_list = [network.CNNAE3D512(), network.CNNAE3D512()]
xp = cuda.cupy if len(gpu_list) > 0 else numpy
if len(gpu_list) > 0:
    for i, gpu in enumerate(gpu_list):
        cuda.check_cuda_available()
        cuda.get_device(gpu).use()
        model_list[i].to_gpu()
        serializers.load_npz(model_path_list[i], model_list[i])
else:
    for i in range(2):
        serializers.load_npz(model_path_list[i], model_list[i])

# Function to estimate images
def estimate_images(input_img, model):
    model.train_dropout = False
    input_img_ = (input_img.astype(numpy.float32) / 255.).transpose(2, 0, 1)
    with chainer.using_config('enable_backprop', False), chainer.using_config('volatile', True):
        input_img_ = chainer.Variable(xp.array([input_img_]))
    res = model(input_img_).data[0]
    if len(gpu_list) > 0:
        res = cuda.to_cpu(res)

    out_img_list = []
    for i in range(res.shape[1]):
        out_img = (255. * res[:, i, :, :].transpose(1, 2, 0)).astype(numpy.uint8)
        out_img_list.append(out_img)

    return out_img_list

# Load input image
img = cv2.imread(f_path)

# Estimate exposed images
out_img_list = []
if len(gpu_list) > 0:
    for i, gpu in enumerate(gpu_list):
        cuda.get_device(gpu).use()
        out_img_list.extend(estimate_images(img, model_list[i]))
        if i == 0:
            out_img_list.reverse()
            out_img_list.append(img)
else:
    for i in range(2):
        out_img_list.extend(estimate_images(img, model_list[i]))
        if i == 0:
            out_img_list.reverse()
            out_img_list.append(img)


#
# # Select and Merge
# threshold = 64
# stid = 0
# prev_img = out_img_list[8].astype(numpy.float32)
# out_img_list.reverse()
# for out_img in out_img_list[9:]:
#     img = out_img.astype(numpy.float32)
#     if (img > (prev_img + threshold)).sum() > 0:
#         break
#     prev_img = img[:, :, :].copy()
#     stid += man
#
# edid = 0
# prev_img = out_img_list[8].astype(numpy.float32)
# out_img_list.reverse()
# for out_img in out_img_list[9:]:
#     img = out_img.astype(numpy.float32)
#     if (img < (prev_img - threshold)).sum() > 0:
#         break
#     prev_img = img[:, :, :].copy()
#     edid += man
#
# out_img_list = out_img_list[8 - stid:9 + edid]

# Create output directory
outdir_path = os.path.join(base_outdir_path, os.path.splitext(os.path.basename(f_path))[0])
os.makedirs(outdir_path, exist_ok=True)

# Compute exposure times
exposure_times = [1 / 1024. * math.pow(math.sqrt(2.), i) for i in range(len(out_img_list))]
exposure_times = numpy.array(exposure_times).astype(numpy.float32)

# Save each exposure image
for i, out_img in enumerate(out_img_list):
    numer, denom = float(exposure_times[i]).as_integer_ratio()
    if int(math.log10(numer) + 1) > 9:
        numer = int(numer / 10 * (int(math.log10(numer) + 1) - 9))
        denom = int(denom / 10 * (int(math.log10(numer) + 1) - 9))
    if int(math.log10(denom) + 1) > 9:
        numer = int(numer / 10 * (int(math.log10(denom) + 1) - 9))
        denom = int(denom / 10 * (int(math.log10(denom) + 1) - 9))
    exif_ifd = {piexif.ExifIFD.ExposureTime: (numer, denom)}
    exif_dict = {"Exif": exif_ifd}
    exif_bytes = piexif.dump(exif_dict)

    # Convert BGR to RGB for PIL
    out_img_ = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    out_img_pil = Image.fromarray(out_img_)

    # Save image with exif data
    out_img_pil.save(os.path.join(outdir_path, "exposure_" + str(i) + ".jpg"), exif=exif_bytes)

# Merge images to HDR
merge_debvec = cv2.createMergeDebevec()
hdr_debvec = merge_debvec.process(out_img_list, times=exposure_times.copy())
cv2.imwrite(os.path.join(outdir_path, 'MergeDebevec.hdr'), hdr_debvec)

merge_mertens = cv2.createMergeMertens(1., 1., 1.e+38)
res_mertens = merge_mertens.process(out_img_list)
cv2.imwrite(os.path.join(outdir_path, 'MergeMertens.hdr'), res_mertens)

# Close all open windows
cv2.destroyAllWindows()


# # 定义源目录和目标目录
# input_ldr_dir = './results_generate/forest'
# lr_under_dir = './dataset/test_data_merge/lr_under'
# lr_over_dir = './dataset/test_data_merge/lr_over'
#
# # 图片文件名
# first_img = 'exposure_0.png'
# ninth_img = 'exposure_9.png'
#
# # 构建源文件路径和目标文件路径
# first_img_path = os.path.join(input_ldr_dir, first_img)
# ninth_img_path = os.path.join(input_ldr_dir, ninth_img)
# lr_under_path = os.path.join(lr_under_dir, first_img)
# lr_over_path = os.path.join(lr_over_dir, ninth_img)
#
# # 检查文件是否存在
# if not os.path.exists(first_img_path):
#     raise FileNotFoundError(f"{first_img} 文件不存在于 {input_ldr_dir}")
#
# if not os.path.exists(ninth_img_path):
#     raise FileNotFoundError(f"{ninth_img} 文件不存在于 {input_ldr_dir}")
#
# # 创建目标目录（如果不存在）
# os.makedirs(lr_under_dir, exist_ok=True)
# os.makedirs(lr_over_dir, exist_ok=True)
#
# # 将图片移动到目标目录
# shutil.move(first_img_path, lr_under_path)
# shutil.move(ninth_img_path, lr_over_path)
#
# print(f"已将 {first_img} 移动到 {lr_under_dir}")
# print(f"已将 {ninth_img} 移动到 {lr_over_dir}")
#
# # 目标文件路径
# first_img_dest_path = os.path.join(lr_under_dir, 'exposure_0.png')
# ninth_img_dest_path = os.path.join(lr_over_dir, 'exposure_9.png')
#
# # 移动、转换格式并调整大小
# def move_and_resize_image(src_path, dest_path, size=(512, 512)):
#     with Image.open(src_path) as img:
#         img = img.resize(size)
#         img.save(dest_path, 'PNG')
#
# move_and_resize_image(first_img_path, first_img_dest_path)
# move_and_resize_image(ninth_img_path, ninth_img_dest_path)
import os
import shutil
from PIL import Image

# 定义源目录和目标目录
input_ldr_dir = 'results_generate/man'
lr_under_dir = 'dataset/test_data_merge/lr_under'
lr_over_dir = 'dataset/test_data_merge/lr_over'

# 图片文件名
first_img = 'exposure_5.jpg'
ninth_img = 'exposure_10.jpg'

# 构建源文件路径
first_img_path = os.path.join(input_ldr_dir, first_img)
ninth_img_path = os.path.join(input_ldr_dir, ninth_img)

# 检查文件是否存在
if not os.path.exists(first_img_path):
    raise FileNotFoundError(f"{first_img} 文件不存在于 {input_ldr_dir}")

if not os.path.exists(ninth_img_path):
    raise FileNotFoundError(f"{ninth_img} 文件不存在于 {input_ldr_dir}")

# 创建目标目录（如果不存在）
os.makedirs(lr_under_dir, exist_ok=True)
os.makedirs(lr_over_dir, exist_ok=True)

# 目标文件路径
first_img_dest_path = os.path.join(lr_under_dir, 'exposure_5.png')
ninth_img_dest_path = os.path.join(lr_over_dir, 'exposure_10.png')

# 移动、转换格式并调整大小
def move_and_resize_image(src_path, dest_path, size=(512, 512)):
    with Image.open(src_path) as img:
        img = img.resize(size)
        img.save(dest_path, 'PNG')

move_and_resize_image(first_img_path, first_img_dest_path)
move_and_resize_image(ninth_img_path, ninth_img_dest_path)

print(f"已将 {first_img} 移动到 {lr_under_dir} 并转换为 .png 格式及调整大小为 512x512")
print(f"已将 {ninth_img} 移动到 {lr_over_dir} 并转换为 .png 格式及调整大小为 512x512")



