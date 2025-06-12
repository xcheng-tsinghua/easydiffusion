import torch
from GaussianDiffusion import GaussianDiffusion
from UNet import UNet
from torchvision import utils
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import logging
from datetime import datetime
import argparse
import os
from data.diff_dataset import DiffDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=200, help='batch size in training')  # bs=200: 占用 21311M 显存
    parser.add_argument('--epoch', default=50, type=int, help='number of epoch in training')
    parser.add_argument('--n_skh_gen', default=20, type=int, help='assert n_skh_gen % 10 == 0')
    parser.add_argument('--save_str', type=str, default='unet_retrain', help='save string')

    return parser.parse_args()


def main(args):
    save_str = args.save_str
    logger = get_log()

    model = UNet(channels=1)
    diffusion = GaussianDiffusion(model)

    # 训练与生成图片
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diffusion = diffusion.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    # 数据集加载 (MNIST)
    transform = transforms.Compose([transforms.Resize(diffusion.img_size), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    # dataset = DiffDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True)

    # sampler = torch.utils.data.RandomSampler(dataset, num_samples=50, replacement=False)
    # dataloader = DataLoader(dataset, batch_size=args.bs, num_workers=4, sampler=sampler)

    try:
        model.load_state_dict(torch.load(f'./model_trained/{save_str}.pth'))
        print(f'training from: ./model_trained/{save_str}.pth')
    except:
        print('no existing model, training from scratch')

    for epoch_idx in range(args.epoch):
        diffusion = diffusion.train()

        for batch_idx, data in enumerate(dataloader):
            x = data[0].to(device)

            optimizer.zero_grad()
            loss = diffusion(x)
            loss.backward()
            optimizer.step()

            state_str = f"Epoch {epoch_idx + 1}/{args.epoch}, batch_idx {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}"
            print(state_str)
            logger.info(state_str)

        scheduler.step()
        torch.save(model.state_dict(), f'model_trained/{save_str}.pth')
        print(f'save state dict: model_trained/{save_str}.pth')

    # 推理部分
    print('generate images')
    sample_epoch = args.n_skh_gen // 10
    gen_idx = 0
    diffusion = diffusion.eval()
    for i in range(sample_epoch):
        print(f'generate {i * 10} to {(i + 1) * 10 - 1}')

        sampled_images = diffusion.sample(batch_size=10)
        for batch_fig_idx in range(10):
            utils.save_image(sampled_images[batch_fig_idx, :, :, :], f'imgs_gen/{save_str}-{gen_idx}.png')
            gen_idx += 1


def get_log():
    print(f'time now: {datetime.now()}')
    # 日志记录
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'log/{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.txt')  # 日志文件路径
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def clear_log(k=5):
    """
    遍历文件夹内的所有 .txt 文件，删除行数小于 k 的文件。
    :param folder_path: 要处理的文件夹路径
    :param k: 行数阈值，小于 k 的文件会被删除
    """
    for filename in os.listdir('log/'):
        # 构造文件的完整路径
        file_path = os.path.join('log/', filename)

        # 检查是否为 .txt 文件
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            try:
                # 统计文件的行数
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    num_lines = len(lines)

                # 如果行数小于 k，则删除文件
                if num_lines < k:
                    print(f"Deleting file: {file_path} (contains {num_lines} lines)")
                    os.remove(file_path)
            except Exception as e:
                # 捕获读取文件时的错误（如编码问题等）
                print(f"Error reading file {file_path}: {e}")


if __name__ == '__main__':
    os.makedirs('log/', exist_ok=True)
    os.makedirs('imgs_gen/', exist_ok=True)
    os.makedirs('model_trained/', exist_ok=True)
    clear_log()
    main(parse_args())

