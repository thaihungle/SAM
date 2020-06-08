import os, sys
PATH_CUR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(PATH_CUR)
sys.path.append('../')
from argparse import ArgumentParser
import logging
import json
from pathlib import Path
import shutil
from tqdm import tqdm
from typing import Dict
import numpy as np
from tensorboardX import SummaryWriter
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from babi.data_preprocess.preprocess import parse
from baselines.sam import qamodel
from baselines.sam.utils import WarmupScheduler

logger = logging.getLogger(__name__)


def train(config: Dict[str, Dict],
          serialization_path: str,
          eval_test: bool = False,
          force: bool = False) -> None:
    # Create serialization dir
    dir_path = Path(serialization_path)
    print(dir_path)
    if dir_path.exists() and force:
        shutil.rmtree(dir_path)
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=False)
    model_path = dir_path / "model.pt"
    config_path = dir_path / "config.json"
    writer = SummaryWriter(log_dir=str(dir_path))

    # Read config
    data_config = config["data"]
    trainer_config = config["trainer"]
    model_config = config["model"]
    optimizer_config = config["optimizer"]
    # Load data
    if data_config["task-id"]=="all":
        task_ids = range(1,21)
    else:
        task_ids = [data_config["task-id"]]
    # train_raw_data, valid_raw_data, test_raw_data, word2id = parse_all(data_config["data_path"],list(range(1,21)))
    word2id = None
    train_data_loaders = {}
    valid_data_loaders = {}
    test_data_loaders = {}

    num_train_batches = num_valid_batches = num_test_batches = 0
    max_seq = 0
    for i in task_ids:
        train_raw_data, valid_raw_data, test_raw_data, word2id = parse(data_config["data_path"],
                                                                       str(i), word2id=word2id,
                                                                       use_cache=True, cache_dir_ext="")
        train_epoch_size = train_raw_data[0].shape[0]
        valid_epoch_size = valid_raw_data[0].shape[0]
        test_epoch_size = test_raw_data[0].shape[0]

        max_story_length = np.max(train_raw_data[1])
        max_sentences = train_raw_data[0].shape[1]
        max_seq = max(max_seq, train_raw_data[0].shape[2])
        max_q = train_raw_data[0].shape[1]
        valid_batch_size = valid_epoch_size // 73  # like in the original implementation
        test_batch_size = test_epoch_size // 73



        train_dataset = TensorDataset(*[torch.LongTensor(a) for a in train_raw_data[:-1]])
        valid_dataset = TensorDataset(*[torch.LongTensor(a) for a in valid_raw_data[:-1]])
        test_dataset = TensorDataset(*[torch.LongTensor(a) for a in test_raw_data[:-1]])

        train_data_loader = DataLoader(train_dataset, batch_size=trainer_config["batch_size"], shuffle=True)
        valid_data_loader = DataLoader(valid_dataset, batch_size=valid_batch_size)
        test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size)

        train_data_loaders[i] = [iter(train_data_loader), train_data_loader]
        valid_data_loaders[i] = valid_data_loader
        test_data_loaders[i] = test_data_loader

        num_train_batches += len(train_data_loader)
        num_valid_batches += len(valid_data_loader)
        num_test_batches += len(test_data_loader)

    print(f"total train data: {num_train_batches*trainer_config['batch_size']}")
    print(f"total valid data: {num_valid_batches*valid_batch_size}")
    print(f"total test data: {num_test_batches*test_batch_size}")
    print(f"voca size {len(word2id)}")

    model_config["vocab_size"] = len(word2id)
    model_config["max_seq"] = max_seq
    model_config["symbol_size"] = 64
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = qamodel.QAmodel(model_config).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(),
                           lr=optimizer_config["lr"], betas=(optimizer_config["beta1"], optimizer_config["beta2"]))
    # optimizer = Nadam(model.parameters(),
    #                        lr=optimizer_config["lr"], betas=(optimizer_config["beta1"], optimizer_config["beta2"]))

    # optimizer = optim.RMSprop(model.parameters(), lr=1e-4, momentum=0.9)

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    warm_up = optimizer_config.get("warm_up", False)

    scheduler = WarmupScheduler(optimizer=optimizer,
                                steps=optimizer_config["warm_up_steps"] if warm_up else 0,
                                multiplier=optimizer_config["warm_up_factor"] if warm_up else 1)

    decay_done = False
    max_acc = 0

    with config_path.open("w") as fp:
        json.dump(config, fp, indent=4)
    if eval_test:
        print(f"testing ... load {model_path.absolute()}")
        model.load_state_dict(torch.load(model_path.absolute()))
        # Evaluation on test data
        model.eval()
        correct = 0
        test_loss = 0
        with torch.no_grad():
            total_test_samples = 0
            single_task_acc = [0] * len(test_data_loaders)
            for k, te in test_data_loaders.items():
                test_data_loader = te
                task_acc = 0
                single_task_samples = 0
                for story, story_length, query, answer in tqdm(test_data_loader):
                    logits = model(story.to(device), query.to(device))
                    answer = answer.to(device)
                    correct_batch = (torch.argmax(logits, dim=-1) == answer).sum()
                    correct += correct_batch.item()
                    task_acc += correct_batch.item()
                    loss = loss_fn(logits, answer)
                    test_loss += loss.sum().item()
                    total_test_samples+=story.shape[0]
                    single_task_samples+=story.shape[0]
                print(f"validate acc task {k}: {task_acc/single_task_samples}")
                single_task_acc[k - 1] = task_acc/single_task_samples
            test_acc = correct / total_test_samples
            test_loss = test_loss / total_test_samples
        print(f"Test accuracy: {test_acc:.3f}, loss: {test_loss:.3f}")
        print(f"test avg: {np.mean(single_task_acc)}")
        raise True

    for i in range(trainer_config["epochs"]):
        logging.info(f"##### EPOCH: {i} #####")
        scheduler.step()
        # Train
        model.train()
        correct = 0
        train_loss = 0
        for _ in tqdm(range(num_train_batches)):
            loader_i = random.randint(0,len(train_data_loaders)-1)+1
            try:
                story, story_length, query, answer = next(train_data_loaders[loader_i][0])
            except StopIteration:
                train_data_loaders[loader_i][0] = iter(train_data_loaders[loader_i][1])
                story, story_length, query, answer = next(train_data_loaders[loader_i][0])
            optimizer.zero_grad()
            logits = model(story.to(device), query.to(device))
            answer = answer.to(device)
            correct_batch = (torch.argmax(logits, dim=-1) == answer).sum()
            correct += correct_batch.item()

            loss = loss_fn(logits, answer)
            train_loss += loss.sum().item()
            loss = loss.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), optimizer_config["max_gradient_norm"])
            # nn.utils.clip_grad_value_(model.parameters(), 10)

            optimizer.step()

        train_acc = correct / (num_train_batches*trainer_config["batch_size"])
        train_loss = train_loss / (num_train_batches*trainer_config["batch_size"])

        # Validation
        model.eval()
        correct = 0
        valid_loss = 0
        with torch.no_grad():
            total_valid_samples = 0
            for  k, va in valid_data_loaders.items():
                valid_data_loader = va
                task_acc = 0
                single_valid_samples = 0
                for story, story_length, query, answer in valid_data_loader:
                    logits = model(story.to(device), query.to(device))
                    answer = answer.to(device)
                    correct_batch = (torch.argmax(logits, dim=-1) == answer).sum()
                    correct += correct_batch.item()
                    loss = loss_fn(logits, answer)
                    valid_loss += loss.sum().item()
                    task_acc += correct_batch.item()
                    total_valid_samples+= story.shape[0]
                    single_valid_samples+= story.shape[0]
                print(f"validate acc task {k}: {task_acc/single_valid_samples}")
            valid_acc = correct / total_valid_samples
            valid_loss = valid_loss / total_valid_samples
            if valid_acc>max_acc:
                print(f"saved model...{model_path}")
                torch.save(model.state_dict(), model_path.absolute())
                max_acc = valid_acc

        writer.add_scalars("accuracy", {"train": train_acc,
                                        "validation": valid_acc}, i)
        writer.add_scalars("loss", {"train": train_loss,
                                    "validation": valid_loss}, i)

        logging.info(f"\nTrain accuracy: {train_acc:.3f}, loss: {train_loss:.3f}"
                     f"\nValid accuracy: {valid_acc:.3f}, loss: {valid_loss:.3f}")
        if optimizer_config.get("decay", False) and valid_loss < optimizer_config["decay_thr"] and not decay_done:
            scheduler.decay_lr(optimizer_config["decay_factor"])
            decay_done = True




    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script.")
    parser.add_argument("--config-file", type=str, metavar='PATH', default="./babi/configs/config_all.json",
                        help="Path to the model config file")
    parser.add_argument("--serialization-path", type=str, metavar='PATH', default="./saved_models/",
                        help="Serialization directory path")
    parser.add_argument("--eval-test", default=False, action='store_true',
                        help="Whether to eval model on test dataset after training (default: False)")
    parser.add_argument("--logging-level", type=str, metavar='LEVEL', default=20, choices=range(10, 51, 10),
                        help="Logging level (default: 20)")
    args = parser.parse_args()

    logging.basicConfig(level=args.logging_level)

    with open(args.config_file, "r") as fp:
        config = json.load(fp)

    train(config, args.serialization_path, args.eval_test)
