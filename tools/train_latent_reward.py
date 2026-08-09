import argparse
import itertools
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from latsearch.reward.checkpoint import save_trainable_checkpoint
from latsearch.reward.dataset import LatentJsonDataLoader
from latsearch.reward.model import LatentReward

parser = argparse.ArgumentParser(description="Latent Reward Model")
parser.add_argument("--job_id", type=str, default="12345")
parser.add_argument("--seed", type=int, default=1203)
parser.add_argument("--split_seed", type=int, default=1203)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--milestones", nargs="+", type=int, default=[10])
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--json_root_path", type=str, default="./prompt")
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--load_from_pretrained", type=str, default="./checkpoints/VideoReward")
parser.add_argument("--number_frames", type=int, default=9)
parser.add_argument("--weight_VQ", type=float, default=1.0)
parser.add_argument("--weight_MQ", type=float, default=1.0)
parser.add_argument("--weight_TA", type=float, default=1.0)
parser.add_argument("--weight_CLS_VQ", type=float, default=1.0)
parser.add_argument("--weight_CLS_MQ", type=float, default=1.0)
parser.add_argument("--weight_CLS_TA", type=float, default=1.0)
parser.add_argument("--output_dir", type=str, default="./checkpoints/latent_reward")
parser.add_argument("--checkpoint_name", type=str, default="latent_reward.pt")
parser.add_argument("--log_dir", type=str, default="./logs")
parser.add_argument("--report_to_wandb", action="store_true")
parser.add_argument("--wandb_project", type=str, default="LatSearch")
parser.add_argument("--wandb_entity", type=str)
args = parser.parse_args()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device(args.device)
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)
log_txt_path = os.path.join(args.log_dir, args.job_id + "-log.txt")


def main():

    if args.report_to_wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=f"latent-reward-{args.job_id}-seed{args.seed}-lr{args.lr}-VQ{args.weight_VQ}-MQ{args.weight_MQ}-TA{args.weight_TA}-CVQ{args.weight_CLS_VQ}-CMQ{args.weight_CLS_MQ}-CTA{args.weight_CLS_TA}",
            config=vars(args),
        )

    train_l, val_l_10, val_l_15, val_l_20, val_l_25, val_l_30 = load_dataset()

    model = LatentReward(
        load_from_pretrained=args.load_from_pretrained,
        device=device,
        dtype=torch.bfloat16,
    ).to(device)

    with open(log_txt_path, "a") as f:
        for k, v in vars(args).items():
            print(str(k) + "=" + str(v))
            f.write(str(k) + "=" + str(v) + "\n")

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr)
    criterion = nn.MSELoss().to(device)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=0.1
    )

    model_save_path = os.path.join(args.output_dir, args.checkpoint_name)
    for epoch in range(0, args.epochs):
        print("********************" + str(epoch) + "********************")
        start_time = time.time()

        # train for one epoch
        loss_pack = train_with_cls_loss(args, model, train_l, optimizer, criterion, device, epoch)
        (
            train_loss,
            train_loss_VQ,
            train_loss_MQ,
            train_loss_TA,
            train_loss_CLS_VQ,
            train_loss_CLS_MQ,
            train_loss_CLS_TA,
        ) = loss_pack
        # evaluate on validation set
        print("*****" + "validation on step 10" + "*****")
        acc_VQ_10, acc_MQ_10, acc_TA_10 = validation_pairs(args, model, val_l_10, device)
        print("*****" + "validation on step 15" + "*****")
        acc_VQ_15, acc_MQ_15, acc_TA_15 = validation_pairs(args, model, val_l_15, device)
        print("*****" + "validation on step 20" + "*****")
        acc_VQ_20, acc_MQ_20, acc_TA_20 = validation_pairs(args, model, val_l_20, device)
        print("*****" + "validation on step 25" + "*****")
        acc_VQ_25, acc_MQ_25, acc_TA_25 = validation_pairs(args, model, val_l_25, device)
        print("*****" + "validation on step 30" + "*****")
        acc_VQ_30, acc_MQ_30, acc_TA_30 = validation_pairs(args, model, val_l_30, device)

        if args.report_to_wandb:
            wandb.log(
                {
                    "train/loss": train_loss,
                    "train/loss_VQ": train_loss_VQ,
                    "train/loss_MQ": train_loss_MQ,
                    "train/loss_TA": train_loss_TA,
                    "train/loss_CLS_VQ": train_loss_CLS_VQ,
                    "train/loss_CLS_MQ": train_loss_CLS_MQ,
                    "train/loss_CLS_TA": train_loss_CLS_TA,
                    "epoch": epoch,
                }
            )
            wandb.log(
                {
                    "val/pair_acc_VQ_step10": acc_VQ_10,
                    "val/pair_acc_MQ_step10": acc_MQ_10,
                    "val/pair_acc_TA_step10": acc_TA_10,
                    "val/pair_acc_VQ_step15": acc_VQ_15,
                    "val/pair_acc_MQ_step15": acc_MQ_15,
                    "val/pair_acc_TA_step15": acc_TA_15,
                    "val/pair_acc_VQ_step20": acc_VQ_20,
                    "val/pair_acc_MQ_step20": acc_MQ_20,
                    "val/pair_acc_TA_step20": acc_TA_20,
                    "val/pair_acc_VQ_step25": acc_VQ_25,
                    "val/pair_acc_MQ_step25": acc_MQ_25,
                    "val/pair_acc_TA_step25": acc_TA_25,
                    "val/pair_acc_VQ_step30": acc_VQ_30,
                    "val/pair_acc_MQ_step30": acc_MQ_30,
                    "val/pair_acc_TA_step30": acc_TA_30,
                    "epoch": epoch,
                }
            )

        save_trainable_checkpoint(model, model_save_path)
        scheduler.step()

        epoch_time = time.time() - start_time
        print(f"An epoch time: {epoch_time:.2f}s")

    if args.report_to_wandb:
        wandb.finish()


def train(args, model, dataloader, optimizer, criterion, device, epoch):

    losses = AverageMeter("Loss", ":.4f")
    losses_VQ = AverageMeter("Loss_VQ", ":.4f")
    losses_MQ = AverageMeter("Loss_MQ", ":.4f")
    losses_TA = AverageMeter("Loss_TA", ":.4f")
    progress = ProgressMeter(
        len(dataloader), [losses], prefix=f"Epoch: [{epoch}]", log_txt_path=log_txt_path
    )

    model.train()

    for _step, batch in enumerate(dataloader):
        (
            latent_tensors,
            prompts,
            reward_VQ,
            reward_MQ,
            reward_TA,
            latent_similarity,
            denoising_steps,
        ) = batch

        # latent_tensors: torch.float32
        # reward_VQ, reward_MQ, reward_TA: torch.float64
        # latent_similarity: torch.float64

        latent_tensors = latent_tensors.to(device).to(torch.bfloat16)
        reward_VQ = reward_VQ.to(device).to(torch.bfloat16)
        reward_MQ = reward_MQ.to(device).to(torch.bfloat16)
        reward_TA = reward_TA.to(device).to(torch.bfloat16)
        latent_similarity = latent_similarity.to(device).to(torch.bfloat16)
        denoising_steps = denoising_steps.to(device)

        latent_list = [latent_tensors[i] for i in range(latent_tensors.size(0))]
        prompt_list = list(prompts)

        batch_inputs = model.prepare_batch(
            videos=latent_list,
            prompts=prompt_list,
            denoising_steps=denoising_steps,
            num_frames=args.number_frames,
        )

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch_inputs)  # torch.float16

            loss = (
                criterion(output[:, 0], reward_VQ * latent_similarity)
                + criterion(output[:, 1], reward_MQ * latent_similarity)
                + criterion(output[:, 2], reward_TA * latent_similarity)
            )

            loss_VQ = criterion(output[:, 0], reward_VQ * latent_similarity)
            loss_MQ = criterion(output[:, 1], reward_MQ * latent_similarity)
            loss_TA = criterion(output[:, 2], reward_TA * latent_similarity)

            losses.update(loss.item(), latent_tensors.size(0))

            losses_VQ.update(loss_VQ.item(), latent_tensors.size(0))
            losses_MQ.update(loss_MQ.item(), latent_tensors.size(0))
            losses_TA.update(loss_TA.item(), latent_tensors.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if _step % args.log_interval == 0:
            progress.display(_step)

    return losses.avg, losses_VQ.avg, losses_MQ.avg, losses_TA.avg


def train_with_cls_loss(args, model, dataloader, optimizer, criterion, device, epoch):

    losses = AverageMeter("Loss", ":.4f")
    losses_VQ = AverageMeter("Loss_VQ", ":.4f")
    losses_MQ = AverageMeter("Loss_MQ", ":.4f")
    losses_TA = AverageMeter("Loss_TA", ":.4f")
    losses_CLS_VQ = AverageMeter("Loss_CLS_VQ", ":.4f")
    losses_CLS_MQ = AverageMeter("Loss_CLS_MQ", ":.4f")
    losses_CLS_TA = AverageMeter("Loss_CLS_TA", ":.4f")

    progress = ProgressMeter(
        len(dataloader), [losses], prefix=f"Epoch: [{epoch}]", log_txt_path=log_txt_path
    )

    # criterion_cls = torch.nn.MarginRankingLoss(margin=0.0)
    criterion_cls = torch.nn.BCEWithLogitsLoss()

    model.train()

    for _step, batch in enumerate(dataloader):
        (
            latent_tensors,
            prompts,
            reward_VQ,
            reward_MQ,
            reward_TA,
            latent_similarity,
            denoising_steps,
        ) = batch

        latent_tensors = latent_tensors.to(device).to(torch.bfloat16)
        reward_VQ = reward_VQ.to(device).to(torch.bfloat16)
        reward_MQ = reward_MQ.to(device).to(torch.bfloat16)
        reward_TA = reward_TA.to(device).to(torch.bfloat16)
        latent_similarity = latent_similarity.to(device).to(torch.bfloat16)
        denoising_steps = denoising_steps.to(device)

        latent_list = [latent_tensors[i] for i in range(latent_tensors.size(0))]
        prompt_list = list(prompts)

        batch_inputs = model.prepare_batch(
            videos=latent_list,
            prompts=prompt_list,
            denoising_steps=denoising_steps,
            num_frames=args.number_frames,
        )

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch_inputs)

            ##################
            # Binary Classification Loss
            ##################
            # Create pair indices
            all_pairs = list(itertools.combinations(range(reward_VQ.size(0)), 2))
            idx_a = torch.tensor([i for i, j in all_pairs], device=device)
            idx_b = torch.tensor([j for i, j in all_pairs], device=device)
            pred_VQ = output[:, 0]
            pred_MQ = output[:, 1]
            pred_TA = output[:, 2]
            # Binary classification loss for pairwise ranking
            score_diff_VQ = pred_VQ[idx_a] - pred_VQ[idx_b]
            score_diff_MQ = pred_MQ[idx_a] - pred_MQ[idx_b]
            score_diff_TA = pred_TA[idx_a] - pred_TA[idx_b]

            # Binary labels: 1 if A > B, else 0
            label_VQ = (reward_VQ[idx_a] > reward_VQ[idx_b]).float()
            label_MQ = (reward_MQ[idx_a] > reward_MQ[idx_b]).float()
            label_TA = (reward_TA[idx_a] > reward_TA[idx_b]).float()

            # BCEWithLogitsLoss expects raw logits (score_diff), not passed through sigmoid
            loss_CLS_VQ = criterion_cls(score_diff_VQ, label_VQ)
            loss_CLS_MQ = criterion_cls(score_diff_MQ, label_MQ)
            loss_CLS_TA = criterion_cls(score_diff_TA, label_TA)

            ##################
            # Regression Loss
            ##################
            loss_VQ = criterion(output[:, 0], reward_VQ * latent_similarity)
            loss_MQ = criterion(output[:, 1], reward_MQ * latent_similarity)
            loss_TA = criterion(output[:, 2], reward_TA * latent_similarity)

            ##################
            # Combined Loss
            ##################
            loss = (
                args.weight_VQ * loss_VQ
                + args.weight_MQ * loss_MQ
                + args.weight_TA * loss_TA
                + args.weight_CLS_VQ * loss_CLS_VQ
                + args.weight_CLS_MQ * loss_CLS_MQ
                + args.weight_CLS_TA * loss_CLS_TA
            )

            losses.update(loss.item(), latent_tensors.size(0))

            losses_VQ.update(loss_VQ.item(), latent_tensors.size(0))
            losses_MQ.update(loss_MQ.item(), latent_tensors.size(0))
            losses_TA.update(loss_TA.item(), latent_tensors.size(0))
            losses_CLS_VQ.update(loss_CLS_VQ.item(), latent_tensors.size(0))
            losses_CLS_MQ.update(loss_CLS_MQ.item(), latent_tensors.size(0))
            losses_CLS_TA.update(loss_CLS_TA.item(), latent_tensors.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if _step % args.log_interval == 0:
            progress.display(_step)

    return (
        losses.avg,
        losses_VQ.avg,
        losses_MQ.avg,
        losses_TA.avg,
        losses_CLS_VQ.avg,
        losses_CLS_MQ.avg,
        losses_CLS_TA.avg,
    )


def validation(args, model, dataloader, criterion, device):

    losses = AverageMeter("Loss", ":.4f")
    progress = ProgressMeter(len(dataloader), [losses], prefix="Test: ", log_txt_path=log_txt_path)

    model.eval()

    with torch.no_grad():
        for _step, batch in enumerate(dataloader):
            (
                latent_tensors,
                prompts,
                reward_VQ,
                reward_MQ,
                reward_TA,
                latent_similarity,
                denoising_steps,
            ) = batch
            latent_tensors = latent_tensors.to(device).to(dtype=torch.bfloat16)
            reward_VQ = reward_VQ.to(device).float()
            reward_MQ = reward_MQ.to(device).float()
            reward_TA = reward_TA.to(device).float()
            latent_similarity = latent_similarity.to(device).float()
            denoising_steps = denoising_steps.to(device)

            latent_list = [latent_tensors[i] for i in range(latent_tensors.size(0))]
            prompt_list = list(prompts)

            batch_inputs = model.prepare_batch(
                videos=latent_list,
                prompts=prompt_list,
                denoising_steps=denoising_steps,
                num_frames=args.number_frames,
            )

            output = model(batch_inputs)

            loss = (
                criterion(output[:, 0], reward_VQ * latent_similarity)
                + criterion(output[:, 1], reward_MQ * latent_similarity)
                + criterion(output[:, 2], reward_TA * latent_similarity)
            )

            losses.update(loss.item(), latent_tensors.size(0))

            if _step % args.log_interval == 0:
                progress.display(_step)

    return losses.avg


def validation_pairs(args, model, dataloader, device):

    model.eval()

    all_reward_VQ_target = []
    all_reward_MQ_target = []
    all_reward_TA_target = []

    all_reward_VQ_predict = []
    all_reward_MQ_predict = []
    all_reward_TA_predict = []

    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader, desc="Running inference")):
            latent_tensors, prompts, reward_VQ, reward_MQ, reward_TA, _, denoising_steps = batch
            latent_tensors = latent_tensors.to(device).to(dtype=torch.bfloat16)
            reward_VQ = reward_VQ.to(device).float()
            reward_MQ = reward_MQ.to(device).float()
            reward_TA = reward_TA.to(device).float()
            denoising_steps = denoising_steps.to(device)

            latent_list = [latent_tensors[i] for i in range(latent_tensors.size(0))]
            prompt_list = list(prompts)
            batch_inputs = model.prepare_batch(
                videos=latent_list,
                prompts=prompt_list,
                denoising_steps=denoising_steps,
                num_frames=args.number_frames,
            )
            output = model(batch_inputs)

            all_reward_VQ_target.append(reward_VQ)
            all_reward_MQ_target.append(reward_MQ)
            all_reward_TA_target.append(reward_TA)

            all_reward_VQ_predict.append(output[:, 0])
            all_reward_MQ_predict.append(output[:, 1])
            all_reward_TA_predict.append(output[:, 2])

    # Concatenate all results
    all_reward_VQ_target = torch.cat(all_reward_VQ_target).cpu().numpy()
    all_reward_MQ_target = torch.cat(all_reward_MQ_target).cpu().numpy()
    all_reward_TA_target = torch.cat(all_reward_TA_target).cpu().numpy()

    all_reward_VQ_predict = torch.cat(all_reward_VQ_predict).to(torch.float32).cpu().numpy()
    all_reward_MQ_predict = torch.cat(all_reward_MQ_predict).to(torch.float32).cpu().numpy()
    all_reward_TA_predict = torch.cat(all_reward_TA_predict).to(torch.float32).cpu().numpy()

    # Compute pairwise accuracies
    pairwise_acc_VQ = compute_pairwise_accuracy(all_reward_VQ_target, all_reward_VQ_predict)
    pairwise_acc_MQ = compute_pairwise_accuracy(all_reward_MQ_target, all_reward_MQ_predict)
    pairwise_acc_TA = compute_pairwise_accuracy(all_reward_TA_target, all_reward_TA_predict)

    print("[Validation] Pairwise accuracy:")
    print(f" - VQ: {pairwise_acc_VQ:.4f}")
    print(f" - MQ: {pairwise_acc_MQ:.4f}")
    print(f" - TA: {pairwise_acc_TA:.4f}")

    return pairwise_acc_VQ, pairwise_acc_MQ, pairwise_acc_TA


# For 946 videos, there are around 446K pairs generated for valuation
def compute_pairwise_accuracy(targets, predictions):
    assert len(targets) == len(predictions)
    correct, total = 0, 0
    for i, j in itertools.combinations(range(len(targets)), 2):
        gt_i, gt_j = targets[i], targets[j]
        pred_i, pred_j = predictions[i], predictions[j]
        if gt_i == gt_j:
            continue
        total += 1
        if (gt_i > gt_j and pred_i > pred_j) or (gt_i < gt_j and pred_i < pred_j):
            correct += 1
    return correct / total if total > 0 else 0.0


def load_dataset():
    train_set = LatentJsonDataLoader(
        args.json_root_path,
        data_mode="train",
        val_step_chose=None,
        split_seed=args.split_seed,
    )

    # We test the 5 steps of the latent separately, then we can see each result
    val_set_step_10 = LatentJsonDataLoader(
        args.json_root_path,
        data_mode="validation",
        val_step_chose="t10",
        split_seed=args.split_seed,
    )
    val_set_step_15 = LatentJsonDataLoader(
        args.json_root_path,
        data_mode="validation",
        val_step_chose="t15",
        split_seed=args.split_seed,
    )
    val_set_step_20 = LatentJsonDataLoader(
        args.json_root_path,
        data_mode="validation",
        val_step_chose="t20",
        split_seed=args.split_seed,
    )
    val_set_step_25 = LatentJsonDataLoader(
        args.json_root_path,
        data_mode="validation",
        val_step_chose="t25",
        split_seed=args.split_seed,
    )
    val_set_step_30 = LatentJsonDataLoader(
        args.json_root_path,
        data_mode="validation",
        val_step_chose="t30",
        split_seed=args.split_seed,
    )

    prompt_overlap = train_set.prompt_ids.intersection(val_set_step_10.prompt_ids)
    if prompt_overlap:
        raise RuntimeError(f"Prompt leakage detected across splits: {len(prompt_overlap)} prompts")
    print(
        f"Prompt split: {len(train_set.prompt_ids)} train / "
        f"{len(val_set_step_10.prompt_ids)} validation; "
        f"samples: {len(train_set)} train / {len(val_set_step_10)} validation"
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_loader_step_10 = DataLoader(
        val_set_step_10,
        batch_size=args.batch_size * 6,
        shuffle=False,
        drop_last=False,
    )
    val_loader_step_15 = DataLoader(
        val_set_step_15,
        batch_size=args.batch_size * 6,
        shuffle=False,
        drop_last=False,
    )
    val_loader_step_20 = DataLoader(
        val_set_step_20,
        batch_size=args.batch_size * 6,
        shuffle=False,
        drop_last=False,
    )
    val_loader_step_25 = DataLoader(
        val_set_step_25,
        batch_size=args.batch_size * 6,
        shuffle=False,
        drop_last=False,
    )
    val_loader_step_30 = DataLoader(
        val_set_step_30,
        batch_size=args.batch_size * 6,
        shuffle=False,
        drop_last=False,
    )

    return (
        train_loader,
        val_loader_step_10,
        val_loader_step_15,
        val_loader_step_20,
        val_loader_step_25,
        val_loader_step_30,
    )


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix="", log_txt_path=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_txt_path = log_txt_path

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = "\t".join(entries)
        print(print_txt)
        with open(self.log_txt_path, "a") as f:
            f.write(print_txt + "\n")

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == "__main__":
    main()
