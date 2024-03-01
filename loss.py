import pytorch_ssim
import torch
import torch.nn as nn
import torchvision.transforms.functional as Ft
from PIL import ImageFilter
from torchvision.models import vgg16, VGG16_Weights
# import wandb  # wandb是一个免费的，用于记录实验数据的工具。需要在官网创建team


def gram_matrix(i_input):
    # 论文3.3中说明，该矩阵检测相似性，比较的是预测值 & GT值
    # 用于perceptual loss 中 style loss 的计算
    a, b, c, d = i_input.size()
    features = i_input.view(a * b, c * d)
    Gm = torch.mm(features, features.t())
    return Gm.div(a * b * c * d)


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(vgg16(weights=VGG16_Weights.DEFAULT).features)[:23]
        self.features = nn.ModuleList(features).eval()
    
    def forward(self, x):
        results = []
        for i, model in enumerate(self.features):
            x = model(x)
            if i in {3, 8, 15, 22}:
                results.append(x)  
        return results


# def custom_mse_loss(out, gt, mask):
#     # 均方误差，回归损失函数常用误差，预测值与目标值之间差值平方和的均值
#     diff = (out - gt) ** 2
#     penalty = torch.exp(mask)
#     return torch.mean(diff * penalty)


# def compute_ssim_loss(out, gt):
#     # 结构衡量指标SSIM (Structure Similarity Index Measure) ，可以衡量图片的失真程度，也可以衡量两张图片的相似程度。
#     # 与MSE和PSNR衡量绝对误差不同，SSIM是感知模型，即更符合人眼的直观感受。、
#     # pytorch中有现成方法可以调用
#     ssim = pytorch_ssim.ssim(out, gt)
#     return 1 - ssim


class PerceptualLossModule:
    # 训练过程感知损失模块
    # 论文说明包含三大模块：color loss, VGG19 based content loss, style loss
    def __init__(self):
        self.model = Vgg16()
        self.criterion = torch.nn.MSELoss()

        if torch.cuda.is_available():
           self.model.cuda()
           self.criterion.cuda()
    
    def compute_color_loss(self, i_input, target):
        num_img = i_input.size()[0]
        color_loss = 0
        for i in range(num_img):
            im_input = Ft.to_pil_image(i_input[i, :, :, :].data.cpu())
            im_target = Ft.to_pil_image(target[i, :, :, :].data.cpu())
            input_blur = Ft.to_tensor(im_input.filter(ImageFilter.GaussianBlur())).cuda()
            target_blur = Ft.to_tensor(im_target.filter(ImageFilter.GaussianBlur())).cuda()
            color_loss += self.criterion(input_blur, target_blur)
        return color_loss/num_img
    
    def compute_content_loss(self, i_input, target):
        input_feats = self.model(i_input)
        target_feats = self.model(target)
        nr_feats = len(input_feats)
        content_loss = 0
        for i in range(nr_feats):
            content_loss += self.criterion(input_feats[i], target_feats[i]).item()
        return content_loss/nr_feats

    def compute_style_loss(self, i_input, target):
        input_feats = self.model(i_input)
        target_feats = self.model(target)
        nr_feats = len(input_feats)
        style_loss = 0
        for i in range(nr_feats):
            gi = gram_matrix(input_feats[i])
            gt = gram_matrix(target_feats[i])
            style_loss += self.criterion(gt, gi).item()
        return style_loss/nr_feats

    # def compute_perceptual_loss(self, synthetic, real):
    #     # 三大loss相加
    #     color_loss = self.compute_color_loss(synthetic.detach(), real)
    #     style_loss = self.compute_style_loss(synthetic.detach(), real)
    #     content_loss = self.compute_content_loss(synthetic.detach(), real)
    #     # wandb.log({
    #     #     "color_loss": color_loss,
    #     #     "style_loss": style_loss,
    #     #     "content_loss": content_loss
    #     # })
    #     perceptual_loss = color_loss + style_loss + content_loss
    #     return perceptual_loss

    def compute_perceptual_loss_v(self, synthetic, real):
        # 多了个v是啥意思？？？？？？？
        color_loss = self.compute_color_loss(synthetic.detach(), real)
        style_loss = self.compute_style_loss(synthetic.detach(), real)
        content_loss = self.compute_content_loss(synthetic.detach(), real)
        perceptual_loss = color_loss + 1e7 * style_loss + 0.1 * content_loss
        return perceptual_loss
