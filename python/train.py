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
    parser.add_argument('--yaml', type=str, default='./code/yaml/table/spvae.yml', help='yaml config file')
    args = parser.parse_args()
    config = load_config(args.yaml)
    num2device_dict = {-1: 'cpu', 0: 'cuda:0', 1: 'cuda:1', 2: 'cuda:2', 3: 'cuda:3'}
    config[config['mode']]['device'] = num2device_dict[config[config['mode']]['device']]

    # create network and training agent
    tr_agent = get_agent(config)

    # load from checkpoint if provided
    if config['train']['load_ckpt']:
        tr_agent.load_ckpt('latest')

    # data
    train_loader = get_dataloader(config, mode='train')
    val_loader = get_dataloader(config,  mode='val')
    val_loader = cycle(val_loader)
    # start training
    clock = tr_agent.clock
    
    for e in range(clock.epoch, config['train']['epoch']):
        # begin iteration
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            # train step
            outputs, losses = tr_agent.train_func(data)

            # visualize
            if config['train']['vis'] and clock.step % config['train']['vis_frequency'] == 0:
                tr_agent.visualize_batch(data, 'train', outputs=outputs)

            pbar.set_description("EPOCH[{}][{}]".format(e, b))
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))
            
            # validation step
            if clock.step % config['val']['val_frequency'] == 0:
                data = next(val_loader)

                outputs, losses = tr_agent.val_func(data)

                if config['train']['vis'] and clock.step % config['train']['vis_frequency'] == 0:
                    tr_agent.visualize_batch(data, 'val', outputs=outputs)
            clock.tick()

        # update lr by scheduler
        # tr_agent.update_learning_rate()
        clock.tock()
        if clock.epoch % config['train']['save_frequency'] == 0:
            tr_agent.save_ckpt()
        
        if clock.epoch % 10 == 0:
            tr_agent.save_ckpt('latest')



if __name__ == '__main__':
    main()
