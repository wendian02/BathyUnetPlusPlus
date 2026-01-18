import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sys
import os
import yaml
import argparse
import importlib
import logging

import wandb


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

from dataset import create_from_csv
from torch.utils.data import DataLoader


def get_device(device_setting="auto"):
    if device_setting == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_setting)


def get_loss_function(loss_type):
    if loss_type == "MSELoss":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_type}")


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def trainer(model, train_dataloader, test_dataloader,
            optimizer, loss_fn, device, epochs,
            writer, logger=None, global_step=0, stage='',
            save_best=False, best_model_fn=None):
    """train function"""
    total_train_step = global_step
    best_test_loss = float('inf')
    best_epoch = 0

    for idx in range(epochs):
        logger.info(f"start training epochs {idx + 1}")
        # print(f"start [{stage}] epochs {idx + 1}")

        total_train_loss = 0
        train_batch_count = 0
        model.train()


        for data in train_dataloader:
            features, target = data
            features = features.to(device)
            target = target.to(device)
            outputs = model(features)
            loss = loss_fn(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_batch_count += 1
            total_train_step += 1
            if total_train_step % 1000 == 0:
                logger.info(f"training steps: {total_train_step}, Loss: {loss.item():.4f}")
                writer.add_scalar(f"{stage}_training_loss_step", loss.item(), total_train_step)

        train_loss_epoch = total_train_loss / train_batch_count
        logger.info(f"epoch {idx + 1} train loss:{train_loss_epoch}")
        writer.add_scalar(f"{stage}_training_loss_epoch", train_loss_epoch, idx)

        # val each epoch
        model.eval()
        total_test_loss = 0
        test_batch_count = 0
        with torch.no_grad():
            for data in test_dataloader:
                features, target = data
                features = features.to(device)
                target = target.to(device)
                outputs = model(features)
                loss = loss_fn(outputs, target)
                total_test_loss += loss.item()
                test_batch_count += 1

        test_loss_epoch = total_test_loss / test_batch_count
        logger.info(f"epoch {idx + 1} test loss:{test_loss_epoch:.4f}")
        writer.add_scalar(f"{stage}_testing_loss_epoch", test_loss_epoch, idx)

        # save best model
        if (test_loss_epoch < best_test_loss) and save_best:
            best_test_loss = test_loss_epoch
            best_epoch = idx + 1
            torch.save(model.state_dict(), best_model_fn)
            logger.info(f"save best model {best_model_fn}, test loss: {test_loss_epoch:.4f}, epoch: {best_epoch}")
            writer.add_scalar(f"{stage}_testing_loss_best", best_test_loss, idx)

    logger.info("training complete!")
    logger.info(f"Best testing loss: {best_test_loss:.4f}, at epoch {best_epoch}") if save_best else None

    return model, total_train_step


def main_train(config, logger, writer, device, stage=''):
    data_paths = config['data_paths']
    model = config['model']
    loss = config['loss']
    bs = config['bs']
    n_workers = config['n_workers']
    lr = config['lr']
    epochs = config['epochs']
    bands_planet = [444, 492, 533, 566, 612, 666, 707, 866]

    train_data, test_data = create_from_csv(data_paths,
                                            [f"rhorc_{band}" for band in bands_planet],
                                            'depth',
                                            mode="train/val",
                                            val_ratio=0.1)

    logger.info(f"training_size: {len(train_data)}, testing_size: {len(test_data)}")

    # dataloader
    train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=n_workers, persistent_workers=True)
    test_dataloader = DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=n_workers, persistent_workers=True)

    # load model

    model = importlib.import_module(model)
    depth_model = model.get_model().to(device)

    # loss function
    loss = get_loss_function(loss)
    loss.to(device)

    # optimizer
    optimizer = torch.optim.Adam(depth_model.parameters(), lr=lr)

    depth_model, total_train_step = trainer(
        model=depth_model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss,
        device=device,
        epochs=epochs,
        writer=writer,
        logger=logger,
        global_step=0,
        stage=stage,
        save_best=False,
    )

    return depth_model, total_train_step


def main(config):
    # logging
    logs_name = config['logging']['logs_name']
    save_dir = config['logging']['save_dir']
    save_name = config['logging']['save_name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(f"logs/{logs_name}_{timestamp}.log", mode='w'),
            logging.StreamHandler()
        ]
    )
    writer = SummaryWriter(f"logs/{logs_name}_{timestamp}")
    device = get_device()  # 'cpu'
    logging.info(f"using device: {device}")

    # # training
    logger = logging.getLogger('training')
    trained_model, train_step = main_train(config['training'], logger, writer, device,
                                           stage='training')

    # save training model
    os.makedirs(save_dir, exist_ok=True)
    torch.save(trained_model.state_dict(), os.path.join(save_dir, f"{save_name}.params"))
    logger.info("training model saved")

    writer.close()


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='model training parameters')
    parser.add_argument('--config', type=str, default='configs/mlp.yaml',
                        help='config path')
    args = parser.parse_args()

    # load config
    config = load_config(args.config)

    # wandb
    wandb.init(
        project="bathymetry",
        name="mlp",
        sync_tensorboard=True,
    )

    main(config)
