import argparse
import os
from collections import OrderedDict

from tqdm import tqdm

from agent import get_agent
from config import load_config
from dataset import get_dataloader
from util.utils import cycle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default='./code/yaml/chair/train_vqvae.yml', help='yaml config file')
    args = parser.parse_args()
    config = load_config(args.yaml)
    num2device_dict = {-1: 'cpu', 0: 'cuda:0', 1: 'cuda:1', 2: 'cuda:2', 3: 'cuda:3'}
    config[config['mode']]['device'] = num2device_dict[config[config['mode']]['device']]

    # create network and training agent
    tr_agent = get_agent(config)

    # load from checkpoint if provided
    # if config['train']['load_ckpt']:
    tr_agent.load_ckpt('latest')
    tr_agent.net.eval()

    # data
    test_loader = get_dataloader(config, mode='train')
    pbar = tqdm(test_loader)
    # tr_agent.save_ckpt('latest')
    for b, data in enumerate(pbar):
        imgs = data[0]
        # print('{} {}'.format(imgs[0, :, 3, :, :].min(), imgs[0, :, 3, :, :].max()))
        filenames = data[-1]
        # flat
        flat_filenames = []
        
        for j in range(len(filenames)):
            flat_filenames.append(filenames[j])
        filenames = flat_filenames
        
        flag = 1
        # flag = 0
        # selected_ids = ['4442b044230ac5c043dbb6421d614c0d', '46c3080551df8a62e8258fa1af480210']
        # for i in range(imgs.shape[0]):
        #     filename = filenames[i]
        #     model_id = os.path.basename(filename).split('.')[0].split('_')[0]
        #     # print(model_id)
        #     if model_id in selected_ids:
        #         flag = 1
        
        if flag == 1:
            print(1)
            # validation sss
            outputs, losses = tr_agent.val_func(data)
            tr_agent.visualize_batch(data, 'test', outputs=outputs)
            # tr_agent.test_func(data, 'generate', outputs=outputs)


if __name__ == '__main__':
    main()
