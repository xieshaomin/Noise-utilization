from model import objectives

from .CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 11.26 新加：余弦相似度矩阵 ====================

def cosine_matrix(clean_input, noise_input, eps=1e-8):
    """
    clean_input: [N_c, D]
    noise_input: [N_n, D]
    返回: [N_n, N_c]，第 j 行和第 i 列是 noise_j vs clean_i 的余弦相似度
    """
    # 先 L2 标准化再做矩阵乘，更稳定
    clean = F.normalize(clean_input.float(), dim=-1)
    noise = F.normalize(noise_input.float(), dim=-1)
    return noise @ clean.t()      # [N_n, N_c]


# ==================================================================


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class RDE(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        # CLIP backbone
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(
            args.pretrain_choice, args.img_size, args.stride_size
        )
        self.embed_dim = base_cfg["embed_dim"]

        # 温度参数
        self.logit_scale = torch.ones([]) * (1.0 / args.temperature)

        # TSE 模块
        self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
        self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)

        # 损失类型
        if "TAL" in self.current_task:
            loss_type = "TAL"
        elif "TRL" in self.current_task:
            loss_type = "TRL"
        elif "InfoNCE" in self.current_task:
            loss_type = "InfoNCE"
        elif "SDM" in self.current_task:
            loss_type = "SDM"
        else:
            exit()
        self.loss_type = loss_type

    # ---------------- 任务配置 ----------------
    def _get_model_device(self):
        # 以 base_model 的参数为准，获取当前 CLIP 模型所在的 device
        return next(self.base_model.parameters()).device
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split("+")]
        print(f"Training Model with {self.current_task} tasks")

    # ---------------- 基本 encode 接口 ----------------

    def encode_image(self, image):
        device = self._get_model_device()
        # 保证 image 和 CLIP 在同一个 device 上
        image = image.to(device, non_blocking=True)
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()

    def encode_text(self, text):
        device = self._get_model_device()
        text = text.long().to(device, non_blocking=True)
        x, _ = self.base_model.encode_text(text)
        idx = text.argmax(dim=-1)  # 在同一 device 上取 argmax
        return x[torch.arange(x.shape[0], device=device), idx].float()

    def encode_image_tse(self, image):
        device = self._get_model_device()
        image = image.to(device, non_blocking=True)
        x, atten_i = self.base_model.encode_image(image)
        i_tse_f = self.visul_emb_layer(x, atten_i)
        return i_tse_f.float()

    def encode_text_tse(self, text):
        device = self._get_model_device()
        text = text.long().to(device, non_blocking=True)
        x, atten_t = self.base_model.encode_text(text)
        idx = text.argmax(dim=-1)
        t_tse_f = self.texual_emb_layer(x, text, atten_t)
        return t_tse_f.float()


    # ---------------- Proxy / Noisy 处理 ----------------

    @torch.no_grad()
    def predict_pairs(self, img_embs, txt_embs):
        """
        给一批 (image, text) 对打“是正对”的置信度。
        约定：第 i 张图 对 第 i 个文本 是候选正样本。
        输入:
          img_embs: [B, D]
          txt_embs: [B, D]
        返回:
          p: [B]，每一对 (i, i) 的匹配概率
        """
        img = F.normalize(img_embs.float(), dim=-1)  # [B, D]
        txt = F.normalize(txt_embs.float(), dim=-1)  # [B, D]
        logit_scale = self.logit_scale.exp().float()
        sims = logit_scale * (img @ txt.t())         # [B, B]
        probs = sims.softmax(dim=1)                  # 按文本维度 softmax
        p = probs.diag()                             # 取对角线概率
        return p                                     # [B]

    @torch.no_grad()
    def process_noisy_p1(
        self,
        images_sc,       # 干净小样本图像 [N_sc, C, H, W]  —— CPU
        captions_sc,     # 干净小样本文本 ids [N_sc, L]   —— CPU
        images_n,        # 噪声图像 [N_n, C, H, W]        —— CPU
        min_prob=0.5,
        chunk_size=256,
    ):
        """
        用干净样本给噪声样本找“代理文本”，并输出修正后的置信度。
        """
        N_sc = images_sc.size(0)
        img_embs_sc_list = []
        for i in range(0, N_sc, chunk_size):
            imgs = images_sc[i : i + chunk_size]
            embs = self.encode_image(imgs)
            img_embs_sc_list.append(embs.cpu())
            del imgs, embs
            torch.cuda.empty_cache()
        img_embs_sc = torch.cat(img_embs_sc_list, dim=0)

        N_n = images_n.size(0)
        img_embs_n_list = []
        for i in range(0, N_n, chunk_size):
            imgs = images_n[i : i + chunk_size]
            embs = self.encode_image(imgs)
            img_embs_n_list.append(embs.cpu())
            del imgs, embs
            torch.cuda.empty_cache()
        img_embs_n = torch.cat(img_embs_n_list, dim=0)

        clean = F.normalize(img_embs_sc.float(), dim=-1)
        noise = F.normalize(img_embs_n.float(), dim=-1)

        N_sc = clean.size(0)
        N_n = noise.size(0)

        clean_chunk = max(2048, chunk_size)
        noise_chunk = max(1024, chunk_size)

        best_vals = torch.full((N_n,), -float("inf"), dtype=torch.float32)
        best_idx = torch.zeros((N_n,), dtype=torch.long)

        for i in range(0, N_sc, clean_chunk):
            cb = clean[i : i + clean_chunk]
            for j in range(0, N_n, noise_chunk):
                nb = noise[j : j + noise_chunk]
                sims_block = nb @ cb.t()
                vals, idxs = sims_block.max(dim=1)
                sl = slice(j, j + nb.size(0))
                mask = vals > best_vals[sl]
                best_vals[sl][mask] = vals[mask]
                best_idx[sl][mask] = i + idxs[mask]
                del sims_block, vals, idxs
                torch.cuda.empty_cache()

        indices = best_idx
        proxy_text = captions_sc[indices]

        corrected_labels = torch.zeros(N_n, dtype=torch.float32)
        top_values = best_vals

        for i in range(0, N_n, chunk_size):
            sl = slice(i, i + chunk_size)

            imgs = images_n[sl]
            caps = proxy_text[sl]

            img_embs_chunk = self.encode_image(imgs)
            txt_embs_chunk = self.encode_text(caps)

            p_chunk = self.predict_pairs(img_embs_chunk, txt_embs_chunk).cpu()

            corrected_labels[sl] = p_chunk * top_values[sl]

            del imgs, caps, img_embs_chunk, txt_embs_chunk, p_chunk
            torch.cuda.empty_cache()

        corrected_labels[corrected_labels < min_prob] = 0.0
        corrected_labels = corrected_labels.unsqueeze(0)

        return corrected_labels, proxy_text

    def process_noisy_p1_from_embs(
        self,
        img_embs_sc,
        captions_sc,
        img_embs_n,
        min_prob=0.5,
        chunk_size=256,
    ):
        clean = F.normalize(img_embs_sc.float(), dim=-1)
        noise = F.normalize(img_embs_n.float(), dim=-1)

        N_sc = clean.size(0)
        N_n = noise.size(0)

        clean_chunk = max(2048, chunk_size)
        noise_chunk = max(1024, chunk_size)

        best_vals = torch.full((N_n,), -float("inf"), dtype=torch.float32)
        best_idx = torch.zeros((N_n,), dtype=torch.long)

        for i in range(0, N_sc, clean_chunk):
            cb = clean[i : i + clean_chunk]
            for j in range(0, N_n, noise_chunk):
                nb = noise[j : j + noise_chunk]
                sims_block = nb @ cb.t()
                vals, idxs = sims_block.max(dim=1)
                sl = slice(j, j + nb.size(0))
                mask = vals > best_vals[sl]
                best_vals[sl][mask] = vals[mask]
                best_idx[sl][mask] = i + idxs[mask]
                del sims_block, vals, idxs
                torch.cuda.empty_cache()

        indices = best_idx
        proxy_text = captions_sc[indices]

        corrected_labels = torch.zeros(N_n, dtype=torch.float32)
        top_values = best_vals

        device = self._get_model_device()
        for i in range(0, N_n, chunk_size):
            sl = slice(i, i + chunk_size)
            caps = proxy_text[sl]
            txt_embs_chunk = self.encode_text(caps)
            img_embs_chunk = noise[sl].to(device)
            p_chunk = self.predict_pairs(img_embs_chunk, txt_embs_chunk).cpu()
            corrected_labels[sl] = p_chunk * top_values[sl]
            del caps, txt_embs_chunk, img_embs_chunk, p_chunk
            torch.cuda.empty_cache()

        corrected_labels[corrected_labels < min_prob] = 0.0
        corrected_labels = corrected_labels.unsqueeze(0)
        return corrected_labels, proxy_text

    # ---------------- 原始 per-loss & forward ----------------

    def compute_per_loss(self, batch):
        """
        用于 GMM 估计的两个分支 loss（和原版 RDE 一致）
        """
        images = batch["images"]
        caption_ids = batch["caption_ids"]

        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)

        # CLIP 全局特征
        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[
            torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)
        ].float()

        # TSE 特征
        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)

        lossA, simsA = objectives.compute_per_loss(
            i_feats,
            t_feats,
            batch["pids"],
            tau=self.args.tau,
            margin=self.args.margin,
            loss_type=self.loss_type,
            logit_scale=self.logit_scale,
        )
        lossB, simsB = objectives.compute_per_loss(
            i_tse_f,
            t_tse_f,
            batch["pids"],
            tau=self.args.tau,
            margin=self.args.margin,
            loss_type=self.loss_type,
            logit_scale=self.logit_scale,
        )

        return lossA.detach().cpu(), lossB.detach().cpu(), simsA, simsB

    def forward(self, batch):
        """
        主训练前向：RBS（原来的 bge_loss + tse_loss）
        batch 里现在多了一个字段：batch['label_hat']，由 processor.py 注入。
        """
        ret = dict()
        ret.update({"temperature": 1 / self.logit_scale})

        images = batch["images"]
        caption_ids = batch["caption_ids"]

        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)

        # CLIP 全局特征
        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[
            torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)
        ].float()

        # TSE 特征
        i_tse_f = self.visul_emb_layer(image_feats, atten_i).float()
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t).float()

        # label_hat: [B]，来自 GMM + proxy
        if "label_hat" in batch and batch["label_hat"] is not None:
            label_hat = batch["label_hat"].to(i_feats.device).float().view(-1)
        else:
            B = i_feats.size(0)
            label_hat = torch.ones(B, device=i_feats.device)

        loss1, loss2 = objectives.compute_rbs(
            i_feats,
            t_feats,
            i_tse_f,
            t_tse_f,
            batch["pids"],
            label_hat=label_hat,
            margin=self.args.margin,
            tau=self.args.tau,
            loss_type=self.loss_type,
            logit_scale=self.logit_scale,
        )

        ret.update({"bge_loss": loss1})
        ret.update({"tse_loss": loss2})

        return ret


def build_model(args, num_classes=11003):
    model = RDE(args, num_classes)
    # covert model to fp16（保持你原来的做法）
    convert_weights(model)
    return model
