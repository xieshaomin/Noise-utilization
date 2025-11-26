import logging
import time
import torch
import numpy as np
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from sklearn.mixture import GaussianMixture


def split_prob(prob, threshld):
    if prob.min() > threshld:
        threshld = np.sort(prob)[len(prob) // 100]
    pred = prob > threshld
    return (pred + 0)


def get_loss(model, data_loader):
    logger = logging.getLogger("RDE.train")
    model.eval()
    device = "cuda"
    data_size = data_loader.dataset.__len__()

    lossA, lossB = torch.zeros(data_size), torch.zeros(data_size)
    simsA, simsB = torch.zeros(data_size), torch.zeros(data_size)

    for i, batch in enumerate(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        index = batch["index"]
        with torch.no_grad():
            la, lb, sa, sb = model.compute_per_loss(batch)
            for b in range(la.size(0)):
                lossA[index[b]] = la[b]
                lossB[index[b]] = lb[b]
                simsA[index[b]] = sa[b]
                simsB[index[b]] = sb[b]
            if i % 100 == 0:
                logger.info(f"compute loss batch {i}")

    losses_A = (lossA - lossA.min()) / (lossA.max() - lossA.min())
    losses_B = (lossB - lossB.min()) / (lossB.max() - lossB.min())

    input_loss_A = losses_A.reshape(-1, 1)
    input_loss_B = losses_B.reshape(-1, 1)

    logger.info("\nFitting GMM ...")

    max_samples = min(8000, input_loss_A.shape[0])
    subset_idx = np.random.choice(input_loss_A.shape[0], size=max_samples, replace=False)
    input_loss_A_np = input_loss_A.cpu().numpy()
    input_loss_B_np = input_loss_B.cpu().numpy()
    input_loss_A_sub = input_loss_A_np[subset_idx]
    input_loss_B_sub = input_loss_B_np[subset_idx]

    if model.args.noisy_rate > 0.4 or model.args.dataset_name == "RSTPReid":
        gmm_A = getattr(model, "_gmm_A", None)
        gmm_B = getattr(model, "_gmm_B", None)
        if gmm_A is None:
            gmm_A = GaussianMixture(n_components=2, covariance_type="diag", max_iter=30, tol=1e-3, reg_covar=1e-6, init_params="kmeans", warm_start=True)
        if gmm_B is None:
            gmm_B = GaussianMixture(n_components=2, covariance_type="diag", max_iter=30, tol=1e-3, reg_covar=1e-6, init_params="kmeans", warm_start=True)
    else:
        gmm_A = GaussianMixture(n_components=2, covariance_type="diag", max_iter=10, tol=1e-2, reg_covar=5e-4, init_params="kmeans", warm_start=True)
        gmm_B = GaussianMixture(n_components=2, covariance_type="diag", max_iter=10, tol=1e-2, reg_covar=5e-4, init_params="kmeans", warm_start=True)

    gmm_A.fit(input_loss_A_sub)
    prob_A = gmm_A.predict_proba(input_loss_A_np)
    prob_A = prob_A[:, gmm_A.means_.argmin()]

    gmm_B.fit(input_loss_B_sub)
    prob_B = gmm_B.predict_proba(input_loss_B_np)
    prob_B = prob_B[:, gmm_B.means_.argmin()]

    setattr(model, "_gmm_A", gmm_A)
    setattr(model, "_gmm_B", gmm_B)

    pred_A = split_prob(prob_A, 0.5)
    pred_B = split_prob(prob_B, 0.5)

    return torch.Tensor(pred_A), torch.Tensor(pred_B)


def do_train(
    start_epoch,
    args,
    model,
    train_loader,
    evaluator,
    optimizer,
    scheduler,
    checkpointer,
):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {"num_epoch": num_epoch, "iteration": 0}

    logger = logging.getLogger("RDE.train")
    logger.info("start training")

    meters = {
        "loss": AverageMeter(),
        "bge_loss": AverageMeter(),
        "tse_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0

    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()

        model.epoch = epoch

        pred_A, pred_B = get_loss(model, train_loader)

        consensus_division = pred_A + pred_B
        mask_mid = (consensus_division == 1)
        if mask_mid.any():
            consensus_division[mask_mid] += torch.randint(0, 2, size=((mask_mid + 0).sum(),))
        label_hat = consensus_division.clone()
        label_hat[consensus_division > 1] = 1
        label_hat[consensus_division <= 1] = 0
        label_hat = label_hat.float()

        N = label_hat.size(0)
        label_hat_refined = label_hat.clone()
        proxy_dict = {}

        clean_embs_list, clean_caps_list = [], []
        noisy_embs_list = []
        noisy_idx_list = []

        with torch.no_grad():
            for batch_all in train_loader:
                idx_all = batch_all["index"]
                imgs_all = batch_all["images"]
                caps_all = batch_all["caption_ids"]

                clean_mask = label_hat[idx_all] == 1
                noisy_mask = label_hat[idx_all] == 0

                if clean_mask.any():
                    embs_c = model.encode_image(imgs_all[clean_mask])
                    clean_embs_list.append(embs_c.cpu())
                    clean_caps_list.append(caps_all[clean_mask])

                if noisy_mask.any():
                    embs_n = model.encode_image(imgs_all[noisy_mask])
                    noisy_embs_list.append(embs_n.cpu())
                    noisy_idx_list.append(idx_all[noisy_mask])

            if len(clean_embs_list) > 0 and len(noisy_embs_list) > 0:
                img_embs_sc = torch.cat(clean_embs_list, 0)
                captions_sc = torch.cat(clean_caps_list, 0)
                img_embs_n = torch.cat(noisy_embs_list, 0)
                noisy_indices = torch.cat(noisy_idx_list, 0)

                corrected_labels, proxy_text = model.process_noisy_p1_from_embs(
                    img_embs_sc, captions_sc, img_embs_n, min_prob=0.5
                )

                corrected_labels = corrected_labels.squeeze(0).cpu()
                proxy_text = proxy_text.cpu()

                for k in range(noisy_indices.size(0)):
                    idx_k = int(noisy_indices[k])
                    w = float(corrected_labels[k])
                    if w > 0:
                        label_hat_refined[idx_k] = 1.0
                        proxy_dict[idx_k] = proxy_text[k]

        model.train()
        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            index = batch["index"]

            batch["label_hat"] = label_hat_refined[index.cpu()]

            idx_list = index.cpu().tolist()
            for i, idx_i in enumerate(idx_list):
                if idx_i in proxy_dict:
                    batch["caption_ids"][i] = proxy_dict[idx_i].to(device)

            ret = model(batch)
            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch["images"].shape[0]
            meters["loss"].update(total_loss.item(), batch_size)
            meters["bge_loss"].update(ret.get("bge_loss", 0), batch_size)
            meters["tse_loss"].update(ret.get("tse_loss", 0), batch_size)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)

        tb_writer.add_scalar("lr", scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar("temperature", ret["temperature"], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                    epoch,
                    time_per_batch,
                    train_loader.batch_size / time_per_batch,
                )
            )
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                global_best = getattr(do_train, "_best_top1", 0.0)
                if global_best < top1:
                    setattr(do_train, "_best_top1", top1)
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)

    if get_rank() == 0:
        logger.info(f"best R1: {getattr(do_train, '_best_top1', 0.0)} at epoch {arguments.get('epoch', start_epoch)}")

    arguments["epoch"] = epoch
    checkpointer.save("last", **arguments)


def do_inference(model, test_img_loader, test_txt_loader):
    logger = logging.getLogger("RDE.test")
    logger.info("Enter inferencing")
    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
