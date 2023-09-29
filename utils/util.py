import math
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
import numpy as np
import cv2


def get_index_of_pair(num):
    """
    By given a number, to know the index of the pair in the list of pair
    eg. [(1,2), (3,4), (5,6), (7,8), (9,10)]
    num = 1, idx = ceil(1/2) = 1st pair = index 0 pair
    num = 5, idx = ceil(5/2) = 3rd pair = index 2 pair
    """
    idx = math.ceil(num / 2)
    return idx - 1 # index start from 0

def cal_detection_suc(no_detection, total_img_num):
    # analyze how many images in each experiment (e.g. adv/adv_gn/adv_edge/adv_gn_edge) are not detected
    n_no_detection_clean = len([img for img in no_detection if "clean" in img])
    n_no_detection_adv = len([img for img in no_detection if "adv" in img and 'edge' not in img and 'gn' not in img])
    n_no_detection_adv_gn = len([img for img in no_detection if "adv" in img and 'edge' not in img and 'gn' in img])
    n_no_detection_adv_edge = len([img for img in no_detection if "adv" in img and 'edge' in img and 'gn' not in img])
    n_no_detection_adv_edge_gn = len([img for img in no_detection if "adv" in img and 'edge_gn' in img])
    n_no_detection_adv_gn_edge = len([img for img in no_detection if "adv" in img and 'gn_edge' in img])
    detect_suc_clean = 1 - n_no_detection_clean / total_img_num
    detect_suc_adv = 1 - n_no_detection_adv / total_img_num
    detect_suc_adv_gn = 1 - n_no_detection_adv_gn / total_img_num
    detect_suc_adv_edge = 1 - n_no_detection_adv_edge / total_img_num
    detect_suc_adv_edge_gn = 1 - n_no_detection_adv_edge_gn / total_img_num
    detect_suc_adv_gn_edge = 1 - n_no_detection_adv_gn_edge / total_img_num
    print(f"Total {len(no_detection)} images with no detection, including "
          f"clean pair images detection success rate: {detect_suc_clean} , "
          f"adv images detection success rate: {detect_suc_adv}, "
          f"adv_gn images detection success rate: {detect_suc_adv_gn}, "
          f"adv_edge images detection success rate: {detect_suc_adv_edge}, "
          f"adv_edge_gn images detection success rate: {detect_suc_adv_edge_gn}, "
          f"adv_gn_edge images detection success rate: {detect_suc_adv_gn_edge}, "
          f"no detection img list: {no_detection}"
          )
    return (detect_suc_clean, detect_suc_adv, detect_suc_adv_gn, detect_suc_adv_edge,
            detect_suc_adv_edge_gn, detect_suc_adv_gn_edge)

def cal_psnr(img1_paths, img2_paths):
    psnr_li = []
    for img1_path, img2_path in zip(img1_paths, img2_paths):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        result = PSNR(img1, img2)
        psnr_li.append(result)
    return np.mean(psnr_li)

def cal_ssim(img1_paths, img2_paths):
    ssim_li = []
    for img1_path, img2_path in zip(img1_paths, img2_paths):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        result = SSIM(img1, img2, channel_axis=2)
        ssim_li.append(result)
    return np.mean(ssim_li)