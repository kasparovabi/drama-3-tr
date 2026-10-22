import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

BLOCK = 128


@triton.autotune(
    configs=[triton.Config({"BN": bn}, num_warps=w, num_stages=st) for bn in (32, 64, 128) for w in (4, 8) for st in (2, 3, 4)],
    key=["N", "K", "M"],
)
@triton.jit
def _gemv(x_ptr, w_ptr, s_ptr, y_ptr, N, K, M: tl.constexpr, MP: tl.constexpr, BN: tl.constexpr):
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    rm = tl.arange(0, 16)
    rk = tl.arange(0, 128)
    nmask = rn < N
    acc = tl.zeros((16, BN), tl.float32)
    for kb in range(0, K // 128):
        k = kb * 128 + rk
        w = tl.load(w_ptr + rn[:, None] * K + k[None, :], mask=nmask[:, None], other=0.0).to(tl.bfloat16)
        x = tl.load(x_ptr + rm[:, None] * K + k[None, :], mask=rm[:, None] < M, other=0.0).to(tl.bfloat16)
        s = tl.load(s_ptr + rn * (K // 128) + kb, mask=nmask, other=0.0)
        acc += tl.dot(x, tl.trans(w)) * s[None, :]
    tl.store(y_ptr + rm[:, None] * N + rn[None, :], acc.to(y_ptr.dtype.element_ty), mask=(rm[:, None] < M) & nmask[None, :])


def quantize(w):
    N, K = w.shape
    b = w.float().view(N, K // BLOCK, BLOCK)
    s = b.abs().amax(-1).clamp(min=1e-12) / 448.0
    return (b / s[..., None]).view(N, K).to(torch.float8_e4m3fn), s.contiguous()


def dequantize(q, s):
    N, K = q.shape
    return (q.float().view(N, K // BLOCK, BLOCK) * s[..., None]).view(N, K)


class FP8Linear(nn.Module):
    """Block-fp8 weights. Decode rows (at most 2) run a bandwidth-bound Triton matvec, longer inputs dequantize."""

    def __init__(self, linear):
        super().__init__()
        q, s = quantize(linear.weight.data)
        self.register_buffer("q", q)
        self.register_buffer("s", s)
        self.bias = linear.bias
        self.weight = torch.empty(0, dtype=linear.weight.dtype, device=linear.weight.device)
        self.out_features, self.in_features = q.shape

    def forward(self, x):
        rows = x.reshape(-1, self.in_features)
        if rows.shape[0] <= 2:
            y = torch.empty((rows.shape[0], self.out_features), dtype=x.dtype, device=x.device)
            grid = lambda meta: (triton.cdiv(self.out_features, meta["BN"]),)
            _gemv[grid](rows.contiguous(), self.q, self.s, y, self.out_features, self.in_features, M=rows.shape[0], MP=2)
        else:
            y = F.linear(rows, dequantize(self.q, self.s).to(x.dtype))
        if self.bias is not None:
            y = y + self.bias
        return y.view(*x.shape[:-1], self.out_features)


def convert(module, skip=lambda name, m: False, min_out=512):
    swapped = 0
    for name, child in list(module.named_modules()):
        for attr, sub in list(child.named_children()):
            full = f"{name}.{attr}" if name else attr
            if isinstance(sub, nn.Linear) and sub.in_features % BLOCK == 0 and sub.out_features >= min_out and not skip(full, sub):
                setattr(child, attr, FP8Linear(sub))
                swapped += 1
    return swapped


if __name__ == "__main__":
    torch.manual_seed(0)
    BF = torch.bfloat16
    for N, K in [(9216, 2048), (8192, 2048), (6144, 2048), (2048, 6144), (2048, 4096), (512, 2048)]:
        layers = [nn.Linear(K, N, bias=False, device="cuda", dtype=BF) for _ in range(40)]
        fast = [FP8Linear(l) for l in layers]
        x = torch.randn(2, K, device="cuda", dtype=BF)
        for m in (1, 2):
            ref = layers[0](x[:m]).float()
            got = fast[0](x[:m]).float()
            assert (got - ref).norm() / ref.norm() < 0.05, float((got - ref).norm() / ref.norm())
        big = torch.randn(64, K, device="cuda", dtype=BF)
        assert (fast[0](big).float() - layers[0](big).float()).norm() / layers[0](big).float().norm() < 0.05

        def timeit(mods):
            for m in mods:
                m(x)
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(5):
                for m in mods:
                    m(x)
            b.record()
            torch.cuda.synchronize()
            return a.elapsed_time(b) / (5 * len(mods)) * 1000

        floor = N * K * 2 / 896e9 * 1e6
        err = float((fast[0](x).float() - layers[0](x).float()).norm() / layers[0](x).float().norm())
        print(f"{N}x{K}  bf16 {timeit(layers):.1f}us (floor {floor:.1f})  fp8 {timeit(fast):.1f}us (floor {floor / 2:.1f})  err {err:.4f}", flush=True)
