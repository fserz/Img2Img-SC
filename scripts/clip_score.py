import argparse

import cv2

import scripts.metrics as Metrics
from PIL import Image
import numpy as np
import glob
import torch
from tqdm import tqdm
import clip


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 生成图像路径
    parser.add_argument('-p', '--path', type=str,
                        default='D:\code\Img2Img\Img2Img-SC\scripts\outputs\img2img-samples\\rician\\t2i\Test-TEXTONLY-sample-25-50')
    # 文本描述路径
    parser.add_argument('-t', '--text_folder', type=str,
                        default='D:\code\Img2Img\Img2Img-SC\scripts\outputs\img2img-samples\\rician\\t2i\Test-TEXTONLY-text-25-50')

    args = parser.parse_args()
    # 获取图像文件 glob.glob(): 查找指定路径中所有 PNG 文件。
    # real_names = list(glob.glob('{}/*.png'.format(args.path_gt)))
    # real_names = list(glob.glob('{}/*.jpg'.format(args.path_gt)))
    # print(real_names, args.path_gt)
    
    
    
    fake_names = list(glob.glob('{}/*.png'.format(args.path)))
    print(fake_names, args.path)

    # 获取文本描述文件
    text_files = list(glob.glob('{}/*.txt'.format(args.text_folder)))
    descriptions = []

    # 读取每个文本文件的内容
    for text_file in text_files:
        with open(text_file, 'r') as f:
            descriptions.extend(f.readlines())

    # 对文件名排序，确保匹配顺序一致。
    # real_names.sort()
    fake_names.sort()

    # 加载 CLIP 模型和预处理
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device)


    avg_clip_score = 0.0
    total_clip_score = 0.0
    batch_clip_score = 0.0
    batch_size = 10


    for idx, fname in enumerate(tqdm(fake_names, total=len(fake_names))):

        sr_img = np.array(Image.open(fname))


        # 计算 CLIP Score
        fake_img = preprocess(Image.open(fname)).unsqueeze(0).to(device)
        description = descriptions[idx % len(descriptions)].strip()
        text = clip.tokenize([description]).to(device)

        with torch.no_grad():
            fake_features = model.encode_image(fake_img)
            text_features = model.encode_text(text)

        fake_clip_score = torch.cosine_similarity(fake_features, text_features)
        total_clip_score += fake_clip_score.item()
        batch_clip_score += fake_clip_score.item()

        # 每处理 10 张图像，打印当前图像的 PSNR 和 SSIM 值。
        if idx % 10 == 0:
            avg_batch_clip_score = batch_clip_score / batch_size
            print(f'Batch {idx // batch_size + 1}, Average CLIP Score: {avg_batch_clip_score:.4f}')
            batch_clip_score = 0.0  # 重置批次得分


    # log
    # 计算并打印总的平均CLIP Score
    avg_clip_score = total_clip_score / len(fake_names)
    print('# Validation # Average CLIP Score: {:.4f}'.format(avg_clip_score))