"""把 pyiqa MUSIQ 的显式注意力换成 SDPA（memory-efficient），数值等价，全分辩率 4K 反向显存从 ~90GB 降到几 GB。"""
import torch, torch.nn.functional as F
from pyiqa.archs import musiq_arch

def _sdpa_forward(self, x, mask=None):
    B, N, C = x.shape; h = self.num_heads
    q = self.query(x).reshape(B, N, h, C // h).permute(0, 2, 1, 3)
    k = self.key(x).reshape(B, N, h, C // h).permute(0, 2, 1, 3)
    v = self.value(x).reshape(B, N, h, C // h).permute(0, 2, 1, 3)
    am = None
    if mask is not None:
        # 原实现 mask2d = mask_h*mask_w 再填 -1e3：被 mask 的 key 对所有 query 无贡献；被 mask 的 query 行本身不影响有效 token
        am = (mask.reshape(B, 1, 1, N) > 0)
        if bool(am.all()): am = None
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=am, scale=self.scale)
    x = x.transpose(1, 2).reshape(B, N, C)
    return self.out_drop(self.out(x))

def _clip_rn_forward(self, x, return_token=False, pos_embedding=False):
    """CLIP ModifiedResNet：stem/layer1/layer2 用 checkpoint 重算，4K 全图反向显存 14GB -> ~5GB。"""
    from torch.utils.checkpoint import checkpoint
    def stem(x):
        for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]: x = self.relu(bn(conv(x)))
        return self.avgpool(x)
    x = x.type(self.conv1.weight.dtype)
    import os
    if torch.is_grad_enabled() and x.requires_grad and os.environ.get("TTO_CKPT", "1") != "0":
        x = checkpoint(stem, x, use_reentrant=False); x = checkpoint(self.layer1, x, use_reentrant=False); x = checkpoint(self.layer2, x, use_reentrant=False)
    else:
        x = self.layer2(self.layer1(stem(x)))
    x = self.layer4(self.layer3(x))
    if return_token:
        x, tokens = self.attnpool(x, return_token, pos_embedding); return x, tokens
    return self.attnpool(x, return_token, pos_embedding)

def patch():
    musiq_arch.MultiHeadAttention.forward = _sdpa_forward
    from pyiqa.archs import clip_model
    clip_model.ModifiedResNet.forward = _clip_rn_forward

if __name__ == "__main__":
    import pyiqa, time, warnings, sys, numpy as np
    from PIL import Image
    warnings.filterwarnings("ignore"); dev = "cuda"
    m = pyiqa.create_metric("musiq", as_loss=True, device=dev); m.requires_grad_(False); m.eval()
    x = torch.from_numpy(np.asarray(Image.open(sys.argv[1]).convert("RGB"))).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
    with torch.no_grad(): ref = float(m(x))
    patch()
    with torch.no_grad(): new = float(m(x))
    print(f"orig={ref:.5f} sdpa={new:.5f} diff={abs(ref-new):.2e}")
    theta = torch.zeros_like(x, requires_grad=True)
    for bf in (True, False):
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
        for _ in range(2):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf): l = m((x + 0.03 * torch.tanh(theta)).clamp(0, 1)).float().mean()
            l.backward(); theta.grad = None
        torch.cuda.synchronize(); print(f"fwd+bwd {(time.time()-t)/2*1000:.0f} ms peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB bf16={bf}")
