from train import Trainer
import argparse

def main(config):
    trainer = Trainer(config)
    if config['train']:
        trainer.train()
    else:
        if config.pretrained_path is None:
            raise Exception("[!] You should specify `pretrained_path` to load a pretrained model")
        trainer.test()

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--conv_hidden_num', type=int, default=128)
    parser.add_argument('--z_num', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=75)
    parser.add_argument('--use_cuda', '-uc', action='store_true')
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--print_every', type=int, default=1)
    
    args = parser.parse_args()
    config = vars(args)
    
    main(config)