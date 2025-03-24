import torchshow
from modules import *
from helpers import *
from time import time
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np

from utils import get_current_time

def train_ddpm(time_steps = 2000, epochs = 20, batch_size = 16, device = "cuda", image_dims = (3, 128, 128), low_res_dims = (3, 32, 32)):
    
    tb_writer = SummaryWriter(f'runs/{get_current_time()}')

    ddpm = DiffusionModel(time_steps = time_steps)
    c, hr_sz, _ = image_dims
    _, lr_sz, _ = low_res_dims
    
    ds = SRDataset("./datasets/Nature/x128_all", hr_sz = hr_sz, lr_sz = lr_sz)
    loader = DataLoader(ds, batch_size = batch_size, shuffle = True, drop_last = True, num_workers = 2)

    opt = torch.optim.Adam(ddpm.model.parameters(), lr = 1e-3)
    criterion = nn.MSELoss(reduction="mean")

    ddpm.model.to(device)
    print()
    for ep in range(epochs):
        ddpm.model.train()
        print(f"Epoch {ep}:")
        losses = []
        stime = time()
        
        for i, (x, y) in enumerate(loader):
            
            # 'y' represents the high-resolution target image, while 'x' represents the low-resolution image to be conditioned upon.
            
            bs = y.shape[0]
            x, y = x.to(device), y.to(device)

            ts = torch.randint(low = 1, high = ddpm.time_steps, size = (bs, ))
            gamma = ddpm.alpha_hats[ts].to(device)
            ts = ts.to(device = device)

            y, target_noise = ddpm.add_noise(y, ts)
            y = torch.cat([x, y], dim = 1)
            # print(x.shape, target_noise.shape)
            # print(x.shape)
            predicted_noise = ddpm.model(y, gamma)
            loss = criterion(target_noise, predicted_noise)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            losses.append(loss.item())

            if i % 250 == 0:
                print(f"Loss: {loss.item()}; step {i}; epoch {ep}")

        ftime = time()
        print(f"Epoch trained in {ftime - stime}s; Avg loss => {sum(losses)/len(losses)}")
        tb_writer.add_scalar('Loss/epoch_loss', loss, ep)

        torch.save(ddpm.state_dict(), f"./weight/sr_ep_{ep}.pt")
        print()


def sample(model, lr_img, device = "cuda"):
    # lr_img is expected to be batched
    # set to eval mode
    model.to(device)
    model.eval()
    
    stime = time()
    with torch.no_grad():
    
        y = torch.randn_like(lr_img, device = device)
        lr_img = lr_img.to(device)
        for i, t in enumerate(range(model.time_steps - 1, 0 , -1)):
            alpha_t, alpha_t_hat, beta_t = model.alphas[t], model.alpha_hats[t], model.betas[t]
    
            t = torch.tensor(t, device = device).long()
            pred_noise = model(torch.cat([lr_img, y], dim = 1), alpha_t_hat.view(-1).to(device))
            y = (torch.sqrt(1/alpha_t))*(y - (1-alpha_t)/torch.sqrt(1 - alpha_t_hat) * pred_noise)
            if t > 1:
                noise = torch.randn_like(y)
                y = y + torch.sqrt(beta_t) * noise
            
    ftime = time()
    print(f"y.shape  = {y.shape}")

    torchshow.save(y, f"./sr_sample.jpeg")
    print(f"Done denoising in {ftime - stime}s ")



if __name__ == "__main__":
    # train_ddpm(time_steps=1000, epochs=20, batch_size=32)

    # 创建模型实例
    ddpm = DiffusionModel(time_steps=1000)
    # 加载模型权重
    # 由于直接加载的权重是一个OrderedDict，我们需要将其正确地加载到模型中
    weights = torch.load('/mnt/houjinliang/MyCVProject/Re_SR3/weight/sr_ep_19.pt')
    # 将OrderedDict的权重加载到模型中
    ddpm.load_state_dict(weights)

    img = cv2.imread('/mnt/houjinliang/MyCVProject/Re_SR3/datasets/Nature/test/City_1.jpg')
    img = np.array(img)
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    print(f"img.shape = {img_tensor.shape}")
    sample(model=ddpm, lr_img=img_tensor)