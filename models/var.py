import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class VAR(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        
        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    def get_all_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], 
                cond_BD: Optional[torch.Tensor],
                prefix_len: Optional[int] = None):  # 添加prefix_len参数
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual
            h = resi + self.blocks[-1].drop_path(h)
        else:
            h = h_or_h_and_residual
            
        # 获取所有位置的logits
        all_logits = self.head(self.head_nm(h.float(), cond_BD).float()).float()
        
        # 如果指定了prefix_len,则返回prefix部分的logits
        if prefix_len is not None:
            return all_logits[:, :prefix_len], all_logits[:, prefix_len:]
        return all_logits
        
    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            cur_L += pn*pn
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)
            
            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth: # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks: b.attn.kv_caching(False)
        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
    
    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            
            if self.prog_si == 0: x_BLC = sos
            else: x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC
        
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        
        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        
        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        return x_BLC    # logits BLV, V is vocab_size
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'


class VARHF(VAR, PyTorchModelHubMixin):
            # repo_url="https://github.com/FoundationVision/VAR",
            # tags=["image-generation"]):
    def __init__(
        self,
        vae_kwargs,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        vae_local = VQVAE(**vae_kwargs)
        super().__init__(
            vae_local=vae_local,
            num_classes=num_classes, depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_eps=norm_eps, shared_aln=shared_aln, cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        )

class SDVAR(nn.Module):
    def __init__(
        self,
        draft_model,
        target_model,
        similarity_thresh: float = 0.8
    ):

        super().__init__()
        self.draft_model = draft_model
        self.target_model = target_model
        self.similarity_thresh = similarity_thresh
        

    # 这是一个最简单的内容，用draft_model生成除了最后一层以外的全部内容，然后用target_model生成最后一层的内容
    # 总体内容按照code的方式进行完成
    @torch.no_grad()
    def sdvar_autoregressive_infer_cfg_lastWithTarget(
        self,
        B: int,
        label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None,
        cfg: float = 1.5,
        top_k: int = 0,
        top_p: float = 0.0,
        more_smooth: bool = False,
        gamma: int = 4,
        warmup_steps: int =4
    ) -> torch.Tensor:
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        这里可以考虑top_k, top_p是否需要将target_model和draft_model分开，这样可以更加有效？
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :param gamma: draft_model 每次向前预测的数量
        :param warmup_steps: 参照code表示最开始目标模型预测的数量
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        
###### 初始化参数

        if g_seed is not None:
            self.draft_model.rng.manual_seed(g_seed)
            self.target_model.rng.manual_seed(g_seed)
            rng = self.draft_model.rng
        else:
            rng = None

        if label_B is None:
            label_B = torch.multinomial(
                self.draft_model.uniform_prob, num_samples=B, replacement=True, generator=rng
            ).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full(
                (B,),
                fill_value=self.draft_model.num_classes if label_B < 0 else label_B,
                device=self.draft_model.lvl_1L.device
            )
        # sos = self.draft_model.class_emb(
        #     torch.cat((label_B, torch.full_like(label_B, fill_value=self.draft_model.num_classes)), dim=0)
        # )   # shape: (2B, C)
        sos = cond_BD = self.draft_model.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.draft_model.num_classes)), dim=0))
        # 原来是sos = cond_BD但是我不知道是为了什么
        lvl_pos = self.draft_model.lvl_embed(self.draft_model.lvl_1L) + self.draft_model.pos_1LC
        next_token_map = (
            sos.unsqueeze(1).expand(2*B, self.draft_model.first_l, -1)
            + self.draft_model.pos_start.expand(2*B, self.draft_model.first_l, -1)
            + lvl_pos[:, :self.draft_model.first_l]
        )
        draft_f_hat = sos.new_zeros(B, self.draft_model.Cvae, self.draft_model.patch_nums[-1], self.draft_model.patch_nums[-1])
        
        # 我们实际上要确保两个model的尺度数量一致，确保后提取出来方便使用
        assert self.draft_model.patch_nums == self.target_model.patch_nums
        assert self.draft_model.num_stages_minus_1 == self.target_model.num_stages_minus_1
        self.patch_nums = self.draft_model.patch_nums
        self.num_stages_minus_1 = self.draft_model.num_stages_minus_1
        
        # 实际上draft_model和target_model使用的是同一个vae，而且似乎vae不会影响其他内容，是可以公用的？
        # sos和cond_BD在整个生成过程中发生改变了吗？是可以公用的吗？

###### 小模型生成除最后一层以外的全部内容
        draft_cur_L = 0
        draft_cond_BD_or_gss = self.draft_model.shared_ada_lin(sos)
        total_stages = len(self.patch_nums)
        
        token_hub =[]

        # 先做一个简易版本，通过draft_model生成出了最后一层以外的所有层数，然后接着让target_model生成最后已成
        for blk in self.draft_model.blocks:
            blk.attn.kv_caching(True)

        # 小模型生成draft_steps步的内容
        # 这里是draft_model使用最开始的token_map使用生成向后draft_steps层的预测
        # 按照llm_fast_infer的方法应该是要有一个用前缀生成n步的generate函数，这里可以按照常规的code里边对arinference的修改进行调整,额外需要注意的是他需要的是logits而不是原来的图片
        for si, pn in enumerate(self.patch_nums):
            # len(self.patch_nums)-1是最后一层，
            if si == len(self.patch_nums)-2:
                break

            ratio = (
                si / self.num_stages_minus_1
                if self.num_stages_minus_1 > 0 else 0
            )
            draft_cur_L = draft_cur_L+ pn*pn
            # cond_BD_or_gss = self.draft_model.shared_ada_lin(sos)  # 平常是在循环外的，我也不知道为什么会出现在这里，但是似乎sos在循环过程中没有改变，是不是无所谓？
            x = next_token_map # x = local_map

            for blk in self.draft_model.blocks:
                x = blk(x=x, cond_BD=draft_cond_BD_or_gss, attn_bias=None)
            # logits_draft = self.draft_model.get_logits(x, sos) # 原来是是get_logits(x, cond_BD)为什么会变成sos呢？
            logits_draft = self.draft_model.get_logits(x, cond_BD)
            
            
            t = cfg * ratio
            logits_draft = (1+t)*logits_draft[:B] - t*logits_draft[B:]  # (B, l, V)

            idx_Bl = sample_with_top_k_top_p_(
                logits_draft, rng=rng, top_k=top_k, top_p=top_p, num_samples=1
            )[:, :, 0]

            if not more_smooth:
                emb_BlC = self.draft_model.vae_quant_proxy[0].embedding(idx_Bl)
            else:
                emb_BlC = self.draft_model.vae_quant_proxy[0].embedding(idx_Bl)
            
            h_BChw = emb_BlC.transpose(1,2).reshape(B, self.draft_model.Cvae, pn, pn)

            draft_f_hat, next_token_map = self.draft_model.vae_quant_proxy[0].get_next_autoregressive_input(
                si, total_stages, draft_f_hat, h_BChw
            )

            # prepare for next stage
            # 由于这个不会运行到最后所以不需要做检查了
            next_pn = self.patch_nums[si+1]
            next_token_map = next_token_map.view(B, self.draft_model.Cvae, -1).transpose(1,2)
            token_hub.append(next_token_map)
            next_token_map = (
                self.draft_model.word_embed(next_token_map)
                + lvl_pos[:, draft_cur_L : draft_cur_L + next_pn*next_pn]
            )
            next_token_map = next_token_map.repeat(2,1,1)
        
        token_hub = torch.cat(token_hub, dim = 1)

        # draft模型生成完毕        
        for blk in self.draft_model.blocks:
            blk.attn.kv_caching(False)
    
###### target模型接受draft模型生成的内容然后生成最后一层的内容
        for blk in self.target_model.blocks:
            blk.attn.kv_caching(True)
        

        sos = cond_BD = self.target_model.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        lvl_pos = self.target_model.lvl_embed(self.target_model.lvl_1L) + self.target_model.pos_1LC
        # 这里存在疑惑，为什么我们需要生成一个first_token_map呢？难道说之前token_map不包括在里边吗？但似乎我们每次预测和保存的都是next_token_map而不是当前层的，这可能是其中的一个原因。
        first_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]

        # exit_points表示的是之前已经有的长度，我们应该是可以简化的 
        exit_points = [1,5,14,30,55,91,155,255,424,680]
        # 我们这里只需要target_model生成最后一层的内容所以我们这里设置成9
        # entry_num表示需要多少
        entry_num = 9
        pindex = exit_points[entry_num]

        # 接受之前生成的做为target_model输出的prefix
        target_next_token_map = token_hub
        target_next_token_map = self.word_embed(target_next_token_map) + lvl_pos[:,1:pindex]   
        target_next_token_map = target_next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        target_next_token_map = torch.cat([first_token_map,target_next_token_map],dim=1)

        attn_bias = self.attn_bias_for_masking[:,:,0:pindex,0:pindex]

        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        target_cur_L = 0
        
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            target_cur_L += pn*pn
            if si>entry_num:
                break
            if si<entry_num:
                continue
            x = next_token_map
            AdaLNSelfAttn.forward
            if si == entry_num:
                for b in self.target_model.blocks:
                    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
            elif si > entry_num:
                for b in self.target_model.blocks:
                    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            
            logits_BlV = self.target_model.get_logits(x, cond_BD)

            if si == entry_num:
                ratio = si / self.num_stages_minus_1
                t = cfg * ratio 
                logits_BlV[:B,target_cur_L-pn*pn:target_cur_L] = (1+t) * logits_BlV[:B,target_cur_L-pn*pn:target_cur_L] - t * logits_BlV[B:,target_cur_L-pn*pn:target_cur_L]


                new_L = 0
                for a, b in enumerate(self.patch_nums[0:entry_num+1]):
                    idx_Bl=sample_with_top_k_top_p_(logits_BlV[:B,new_L:new_L + self.patch_nums[a] ** 2], rng=rng, top_k=top_k[a], top_p=top_p, num_samples=1)[:, :, 0]
                    new_L += b*b

                
            elif si > entry_num:
                ratio = si / self.num_stages_minus_1
                t = cfg * ratio
                logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
                idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k[si], top_p=top_p, num_samples=1)[:, :, 0]


            if not more_smooth: # this is the default case
                h_BChw = self.target_model.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.target_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)

            f_hat, next_token_map = self.target_model.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.target_model.Cvae, -1).transpose(1, 2)
                token_hub = torch.cat([token_hub,next_token_map],dim=1)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, target_cur_L:target_cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
            
        # target模型生成完成
        for blk in self.target_model.blocks:
            blk.attn.kv_caching(False)   
                    
        return f_hat, token_hub   # de-normalize, from [-1, 1] to [0, 1]



        
### 复杂版本
        # # 这里可以按照code的方式来进行初始化

        # while current_stages < total_stages:
        #     accepted_count = 0
        #     draft_steps = min(gamma, total_stages - si)

        #     # 小模型生成draft_steps步的内容
        #     # 这里是draft_model使用最开始的token_map使用生成向后draft_steps层的预测
        #     # 按照llm_fast_infer的方法应该是要有一个用前缀生成n步的generate函数，这里可以按照常规的code里边对arinference的修改进行调整,额外需要注意的是他需要的是logits而不是原来的图片
        #     for step_i in range(draft_steps):
        #         pn = self.draft_model.patch_nums[local_si]
        #         ratio = (
        #             local_si / self.draft_model.num_stages_minus_1
        #             if self.draft_model.num_stages_minus_1 > 0 else 0
        #         )
        #         local_cur_L_next = local_cur_L + pn*pn

        #         cond_BD_or_gss = self.draft_model.shared_ada_lin(sos)
        #         x = local_map
        #         for blk in self.draft_model.blocks:
        #             x = blk(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
        #         logits_draft = self.draft_model.get_logits(x, sos)

        #         t = cfg * ratio
        #         logits_draft = (1+t)*logits_draft[:B] - t*logits_draft[B:]  # (B, l, V)

        #         idx_Bl = sample_with_top_k_top_p_(
        #             logits_draft, rng=rng, top_k=top_k, top_p=top_p, num_samples=1
        #         )[:, :, 0]

        #         if not more_smooth:
        #             emb_BlC = self.draft_model.vae_quant_proxy[0].embedding(idx_Bl)
        #         else:
        #             emb_BlC = self.draft_model.vae_quant_proxy[0].embedding(idx_Bl)
        #         h_BChw = emb_BlC.transpose(1,2).reshape(B, self.draft_model.Cvae, pn, pn)
        #         local_f_hat, local_map = self.draft_model.vae_quant_proxy[0].get_next_autoregressive_input(
        #             local_si, total_stages, local_f_hat, h_BChw
        #         )

        #         if local_si != (total_stages-1):
        #             next_pn = self.draft_model.patch_nums[local_si+1]
        #             local_map = local_map.view(B, self.draft_model.Cvae, -1).transpose(1,2)
        #             local_map = (
        #                 self.draft_model.word_embed(local_map)
        #                 + lvl_pos[:, local_cur_L_next : local_cur_L_next + next_pn*next_pn]
        #             )
        #             local_map = local_map.repeat(2,1,1)

                

        #     # 大模型一次性验证
        #     # arinference需要额外添加的参数，首先是prefix表示当前已经生成的
        #     for step_i in range(draft_steps):
        #         pn = self.draft_model.patch_nums[local_si]
        #         ratio = (
        #             local_si / self.draft_model.num_stages_minus_1
        #             if self.draft_model.num_stages_minus_1 > 0 else 0
        #         )
        #         local_cur_L_next = local_cur_L + pn*pn

        #         cond_BD_or_gss = self.draft_model.shared_ada_lin(sos)
        #         x = local_map
        #         for blk in self.draft_model.blocks:
        #             x = blk(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
        #         logits_draft = self.draft_model.get_logits(x, sos)

        #         t = cfg * ratio
        #         logits_draft = (1+t)*logits_draft[:B] - t*logits_draft[B:]  # (B, l, V)

        #         idx_Bl = sample_with_top_k_top_p_(
        #             logits_draft, rng=rng, top_k=top_k, top_p=top_p, num_samples=1
        #         )[:, :, 0]

        #         if not more_smooth:
        #             emb_BlC = self.draft_model.vae_quant_proxy[0].embedding(idx_Bl)
        #         else:
        #             emb_BlC = self.draft_model.vae_quant_proxy[0].embedding(idx_Bl)
        #         h_BChw = emb_BlC.transpose(1,2).reshape(B, self.draft_model.Cvae, pn, pn)
        #         local_f_hat, local_map = self.draft_model.vae_quant_proxy[0].get_next_autoregressive_input(
        #             local_si, total_stages, local_f_hat, h_BChw
        #         )

        #         if local_si != (total_stages-1):
        #             next_pn = self.draft_model.patch_nums[local_si+1]
        #             local_map = local_map.view(B, self.draft_model.Cvae, -1).transpose(1,2)
        #             local_map = (
        #                 self.draft_model.word_embed(local_map)
        #                 + lvl_pos[:, local_cur_L_next : local_cur_L_next + next_pn*next_pn]
        #             )
        #             local_map = local_map.repeat(2,1,1)


        #     # 这里是target_model使用draft_model生成的内容生成下一层的预测
        #     expansions = []
        #     backup_f_hat = f_hat.clone()
        #     backup_next_map = next_token_map.clone()
        #     backup_cur_L = cur_L

        #     local_si = si
        #     local_f_hat = f_hat
        #     local_map = next_token_map
        #     local_cur_L = cur_L



def measure_similarity_with_target_parallel(
    expansions,
    target_model,
    B,
    more_smooth=False,
):
    pass
