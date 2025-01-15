import argparse
import scripts.metrics as Metrics
from PIL import Image
import numpy as np
import glob
import torch
from tqdm import tqdm
import clip


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-p_gt', '--path_gt', type=str,
    #                     default='G:\Giordano\stablediffusion\outputs\img2img-samples\samples-orig-10')
    # parser.add_argument('-p', '--path', type=str,
    #                     default='G:\Giordano\stablediffusion\outputs\img2img-samples\samples-10')

    # 真实图像路径
    parser.add_argument('-p_gt', '--path_gt', type=str,
                        default='D:\code\Img2Img\Img2Img-SC\scripts\outputs\img2img-samples\i2i\Test-samples-orig-5-30')
    # 生成图像路径
    parser.add_argument('-p', '--path', type=str,
                        default='D:\code\Img2Img\Img2Img-SC\scripts\outputs\img2img-samples\i2i\Test-samples-5-30')
    # 文本描述路径
    parser.add_argument('-t', '--text_folder', type=str,
                        default='D:\code\Img2Img\Img2Img-SC\scripts\outputs\img2img-samples\i2i\Test-text-samples-5-30')

    args = parser.parse_args()
    # 获取图像文件 glob.glob(): 查找指定路径中所有 PNG 文件。
    real_names = list(glob.glob('{}/*.png'.format(args.path_gt)))
    # real_names = list(glob.glob('{}/*.jpg'.format(args.path_gt)))
    print(real_names, args.path_gt)
    
    
    
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
    real_names.sort()
    fake_names.sort()

    # 加载 CLIP 模型和预处理
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device)

    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_fid = 0.0
    avg_clip_score = 0.0
    fid_img_list_real, fid_img_list_fake = [],[]
    idx = 0
    fid_idx = 0


    for rname, fname in tqdm(zip(real_names, fake_names), total=len(real_names)):
        idx += 1

        hr_img = np.array(Image.open(rname))
        sr_img = np.array(Image.open(fname))
        psnr = Metrics.calculate_psnr(sr_img, hr_img)
        ssim = Metrics.calculate_ssim(sr_img, hr_img)
        fid_img_list_real.append(torch.from_numpy(hr_img).permute(2,0,1).unsqueeze(0))
        fid_img_list_fake.append(torch.from_numpy(sr_img).permute(2,0,1).unsqueeze(0))
        avg_psnr += psnr
        avg_ssim += ssim

        # 计算 CLIP Score
        real_img = preprocess(Image.open(rname)).unsqueeze(0).to(device)
        fake_img = preprocess(Image.open(fname)).unsqueeze(0).to(device)
        description = descriptions[idx % len(descriptions)].strip()
        text = clip.tokenize([description]).to(device)

        with torch.no_grad():
            real_features = model.encode_image(real_img)
            fake_features = model.encode_image(fake_img)
            text_features = model.encode_text(text)

        real_clip_score = torch.cosine_similarity(real_features, text_features)
        fake_clip_score = torch.cosine_similarity(fake_features, text_features)

        avg_clip_score += (real_clip_score.item() + fake_clip_score.item()) / 2

        # 每处理 10 张图像，打印当前图像的 PSNR 和 SSIM 值。
        if idx % 10 == 0:
            if len(fid_img_list_real) > 1 and len(fid_img_list_fake) > 1:
                fid = Metrics.calculate_FID(torch.cat(fid_img_list_real, dim=0), torch.cat(fid_img_list_fake, dim=0))
                fid_img_list_real, fid_img_list_fake = [], []  # 清空列表
                avg_fid += fid
                fid_idx += 1
            print('Image:{}, PSNR:{:.4f}, SSIM:{:.4f}, FID:{:.4f}'.format(idx, psnr, ssim, fid))

    # 计算 FID 和最终平均值
    #last FID
    # fid = Metrics.calculate_FID(torch.cat(fid_img_list_real,dim=0), torch.cat(fid_img_list_fake,dim=0))
    # avg_fid += fid

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
    avg_fid = avg_fid / fid_idx

    # log
    print('# Validation # PSNR: {}'.format(avg_psnr))
    print('# Validation # SSIM: {}'.format(avg_ssim))
    print('# Validation # FID: {}'.format(avg_fid))
    print('# Validation # Average CLIP Score: {:.4f}'.format(avg_clip_score))