"""
Hybrid RT-DETR backbone (v2, patched)
=======================================
 
PATCH NOTE (this version only):
SingleEncoderPyramid now optionally uses the same PretrainedDetailStem
(ResNet-18, conv1..layer2, stride 8) that GatedFusion/PairFusion already
use, so single-encoder ablations (MobileSAM-only, CLIP-only) can get the
same fine-detail P3 signal the hybrid model gets. ConvNeXt-only is
excluded by default (input_type != "convnext" check in forward()),
since ConvNeXt is already a CNN with native fine-grained spatial detail
and doesn't have the resolution problem the detail branch exists to fix.
Remove that one condition if you want ConvNeXt-only to use it too.
 
Everything else in this file is unchanged from the original v2.
 
Parallel visual encoders:
 
        RGB
         |
    +----+-------------------------+
    |                              |
MobileSAM TinyViT              CLIP ViT (variant and resolution set in the YAML;
(stride 16, letterboxed)       hybrid YAML: ViT-B/32 @ 384 -> stride 32, 12x12 grid;
                               letterboxed, NO center crop, full image)
    |                              |
    +--------------+---------------+
                   |
           Bidirectional
           Cross Attention
                   |
             Gated Fusion
                   |
        CLIP CLS Global Context
                   |
    RGB Detail Branch (pretrained ResNet-18 stem)
                   |
              P3 / P4 / P5
 
 
WHAT CHANGED FROM v1 AND WHY
=============================
 
1. CLIP no longer center-crops to 224x224.
   v1 resized-then-cropped, which silently deleted anything outside
   the center square of the image before the model ever saw it.
   v2 letterboxes the *whole* image (resize longest side, pad to a
   square, like the MobileSAM branch already did) and uses
   `interpolate_pos_encoding=True` so CLIP can run on the larger,
   uncropped input. Padded regions are cropped back out of the
   token grid afterwards (same trick MobileSAM already used).
 
2. CLIP's CLS token and patch tokens are now handled separately.
   CLS is a global/semantic summary, not a spatial one - mixing it
   into the per-patch cross-attention sequence (as v1 did) diluted
   the spatial signal. v2 uses CLS only as an additive global
   context vector, and patch tokens only for the spatial gate.
 
3. The RGB "detail" branch is no longer 3 randomly-initialized
   conv layers. It's the biggest fix here: both SAM (segmentation-
   pretrained) and CLIP (classification-pretrained, heavily
   downsampled) are poor sources of the fine, local detail a
   detector needs at the P3 level. v1 asked a from-scratch stem to
   recover that detail with no pretraining at all. v2 uses an
   ImageNet-pretrained ResNet-18 stem (conv1..layer2, stride 8) for
   this instead.
 
4. Added `get_param_groups()` - a differential-LR helper. Pretrained
   encoders should move slowly; new fusion/head modules need a much
   higher LR to catch up. Training everything at one LR is a common
   reason a hybrid backbone trains slower than a from-scratch one.
 
Training modes for MobileSAM / CLIP / the detail stem are unchanged
in spirit: "frozen" / "partial" / "full", so existing YAML configs
that only reference these three encoders + GatedFusion still work.
 
Ultralytics usage:
 
    import hybrid_modules
    from ultralytics import RTDETR
 
    model = RTDETR("hybrid-rtdetr.yaml")
    model.train(...)
"""
 

import math
import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mobile_sam import sam_model_registry
except ImportError as e:
    raise ImportError(
        "MobileSAM is not installed. Install it from the MobileSAM "
        "repository/package first."
    ) from e

try:
    from transformers import CLIPVisionModel
except ImportError as e:
    raise ImportError(
        "transformers is not installed. Install with: pip install transformers"
    ) from e


MOBILE_SAM_URL = (
    "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLIP_HUB = {
    "base16": "openai/clip-vit-base-patch16",
    "base32": "openai/clip-vit-base-patch32",
    "large14": "openai/clip-vit-large-patch14",
}

# torchvision constructor name + stage-3 (stride-16) output channel count
# for each ConvNeXt variant. tiny/small share channel widths; base/large
# are wider.
CONVNEXT_CTORS = {
    "tiny": "convnext_tiny",
    "small": "convnext_small",
    "base": "convnext_base",
    "large": "convnext_large",
}
CONVNEXT_CHANNELS = {
    "tiny": 384,
    "small": 384,
    "base": 512,
    "large": 768,
}


# ============================================================
# Utilities
# ============================================================

def _group_count(channels, maximum=32):
    """Largest GroupNorm group count <= maximum that divides channels."""
    for g in range(min(maximum, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


def _set_requires_grad(module, value):
    for p in module.parameters():
        p.requires_grad_(value)


def _unpack_input(x):
    """
    Lets any encoder be placed either first (raw image tensor input)
    or second (receiving the dict a previous encoder produced) in a
    two-encoder pairing YAML.

    Returns (image_tensor, incoming_dict_or_None). Every encoder's
    forward() should merge its own output onto `incoming_dict` when
    it isn't None, so downstream fusion modules see both encoders'
    keys regardless of ordering.
    """
    if isinstance(x, dict):
        return x["img"], x
    return x, None


def _count_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def _print_trainable_status(name, module):
    total = _count_parameters(module)
    trainable = _count_trainable(module)
    ratio = 100.0 * trainable / total if total > 0 else 0.0
    print(f"[{name}] trainable={trainable/1e6:.2f}M / total={total/1e6:.2f}M ({ratio:.1f}%)")


def _freeze_except_last_blocks(module, block_keywords, last_n_blocks, train_extra_keywords=None):
    """
    Generic partial fine-tuning helper: freeze everything, then unfreeze
    the last N matched "block-like" submodules (name-matched, since
    MobileSAM / transformers versions expose blocks differently).
    """
    _set_requires_grad(module, False)

    named_modules = list(module.named_modules())
    candidates = []

    for name, submodule in named_modules:
        if not name:
            continue
        lname = name.lower()
        if any(k.lower() in lname for k in block_keywords) and len(list(submodule.parameters())) > 0:
            candidates.append((name, submodule))

    # Drop nested duplicates (keep only top-level matches).
    selected = []
    for name, submodule in candidates:
        if not any(name.startswith(p + ".") for p, _ in selected):
            selected.append((name, submodule))

    if last_n_blocks > 0 and selected:
        for name, submodule in selected[-last_n_blocks:]:
            _set_requires_grad(submodule, True)
            print(f"[FineTune] Unfrozen block: {name}")

    if train_extra_keywords:
        for name, submodule in named_modules:
            lname = name.lower()
            if any(k.lower() in lname for k in train_extra_keywords) and len(list(submodule.parameters())) > 0:
                _set_requires_grad(submodule, True)


# ============================================================
# Conv -> GroupNorm -> SiLU
# ============================================================

class ConvGNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, groups=1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, groups=groups, bias=False)
        self.norm = nn.GroupNorm(_group_count(c2), c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# ============================================================
# MobileSAM Encoder
# ============================================================

class MobileSAMEncoder(nn.Module):
    """
    MobileSAM TinyViT image encoder (stride 16).

    Output: {"img": original RGB tensor, "mobile_sam": feature map}
    """

    def __init__(self, ckpt=None, train_mode="frozen", train_blocks=2):
        super().__init__()

        if train_mode not in {"frozen", "partial", "full"}:
            raise ValueError(f"MobileSAM train_mode must be frozen/partial/full. Got: {train_mode}")

        self.train_mode = train_mode
        self.train_blocks = int(train_blocks)

        path = ckpt or os.path.join("models", "mobile_sam.pt")
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            print(f"[MobileSAM] Downloading checkpoint -> {path}")
            urllib.request.urlretrieve(MOBILE_SAM_URL, path)

        print(f"[MobileSAM] Loading: {path}")
        model = sam_model_registry["vit_t"](checkpoint=path)
        self.encoder = model.image_encoder

        self.register_buffer("mean", model.pixel_mean.clone().float(), persistent=False)
        self.register_buffer("std", model.pixel_std.clone().float(), persistent=False)
        del model

        self._configure_training()

    def _configure_training(self):
        if self.train_mode == "frozen":
            _set_requires_grad(self.encoder, False)
        elif self.train_mode == "full":
            _set_requires_grad(self.encoder, True)
        elif self.train_mode == "partial":
            _freeze_except_last_blocks(
                self.encoder,
                block_keywords=["blocks", "block", "layers"],
                last_n_blocks=self.train_blocks,
                train_extra_keywords=["norm", "neck"],
            )
        _print_trainable_status("MobileSAM", self.encoder)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval() if self.train_mode == "frozen" else self.encoder.train(mode)
        return self

    def forward(self, x):
        img, incoming = _unpack_input(x)
        original = img
        h, w = img.shape[-2:]

        scale = 1024.0 / max(h, w)
        nh, nw = round(h * scale), round(w * scale)

        x = F.interpolate(img.float(), size=(nh, nw), mode="bilinear", align_corners=False)
        x = (x * 255.0 - self.mean.float()) / self.std.float()
        x = F.pad(x, (0, 1024 - nw, 0, 1024 - nh))

        dtype = next(self.encoder.parameters()).dtype
        x = x.to(dtype)

        if self.train_mode == "frozen":
            with torch.no_grad():
                f = self.encoder(x)
        else:
            f = self.encoder(x)

        valid_h, valid_w = math.ceil(nh / 16), math.ceil(nw / 16)
        f = f[..., :valid_h, :valid_w]

        out = {"mobile_sam": f}
        return {**incoming, **out} if incoming is not None else {"img": original, **out}


# ============================================================
# CLIP Vision Encoder (v2 - no center crop)
# ============================================================

class CLIPEncoder(nn.Module):
    """
    Frozen / partially / fully trainable CLIP vision tower.

    v2: letterboxes the *full* image (no center crop) at `input_res`
    and relies on `interpolate_pos_encoding=True` to handle the
    non-native resolution. Falls back to the old fixed-224 crop-free
    resize (still no crop) only if the installed `transformers`
    version doesn't support position-embedding interpolation.

    Output adds:
        "clip_patches": [B, hidden, Hc, Wc]  spatial patch tokens
        "clip_cls":     [B, hidden]          global CLS token
    """

    def __init__(self, variant="base16", train_mode="partial", train_blocks=4, input_res=336):
        super().__init__()

        if variant not in CLIP_HUB:
            raise ValueError(f"Unknown CLIP variant '{variant}'. Choose from {list(CLIP_HUB.keys())}")
        if train_mode not in {"frozen", "partial", "full"}:
            raise ValueError(f"CLIP train_mode must be frozen/partial/full. Got: {train_mode}")

        self.variant = variant
        self.train_mode = train_mode
        self.train_blocks = int(train_blocks)

        model_name = CLIP_HUB[variant]
        print(f"[CLIP] Loading vision model: {model_name}")
        self.vm = CLIPVisionModel.from_pretrained(model_name)

        self.patch_size = self.vm.config.patch_size
        self.hidden_size = self.vm.config.hidden_size

        # input_res must be a multiple of patch_size.
        self.input_res = max(self.patch_size, (input_res // self.patch_size) * self.patch_size)

        self._supports_interp = None  # probed lazily on first forward

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        # NOTE: CLIP's own published mean/std are close to but not
        # identical to ImageNet's; CLIP's own stats are used below.
        self.register_buffer(
            "clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
            persistent=False,
        )

        self._configure_training()

    def _configure_training(self):
        if self.train_mode == "frozen":
            _set_requires_grad(self.vm, False)
        elif self.train_mode == "full":
            _set_requires_grad(self.vm, True)
        elif self.train_mode == "partial":
            _freeze_except_last_blocks(
                self.vm,
                block_keywords=["vision_model.encoder.layers", "encoder.layers", "layers"],
                last_n_blocks=self.train_blocks,
                train_extra_keywords=["post_layernorm", "pre_layrnorm", "visual_projection"],
            )
        _print_trainable_status("CLIP", self.vm)

    def train(self, mode=True):
        super().train(mode)
        self.vm.eval() if self.train_mode == "frozen" else self.vm.train(mode)
        return self

    def _safe_forward(self, x):
        """Try full-resolution forward with interpolated position embeddings;
        fall back to CLIP's native 224x224 (still no crop) if unsupported."""
        if self._supports_interp is None:
            try:
                out = self.vm(pixel_values=x, interpolate_pos_encoding=True)
                self._supports_interp = True
                return out
            except TypeError:
                self._supports_interp = False
                print(
                    "[CLIP] This `transformers` version does not support "
                    "interpolate_pos_encoding for CLIPVisionModel - falling "
                    "back to native 224x224 input. Upgrade `transformers` "
                    "to use the full-resolution path."
                )

        if self._supports_interp:
            return self.vm(pixel_values=x, interpolate_pos_encoding=True)

        x224 = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.vm(pixel_values=x224)

    def forward(self, x):
        original, incoming = _unpack_input(x)
        B, _, h, w = original.shape
        S, P = self.input_res, self.patch_size

        # Letterbox: resize longest side to S, pad to (S, S). No crop.
        scale = S / max(h, w)
        nh = max(P, (round(h * scale) // P) * P)
        nw = max(P, (round(w * scale) // P) * P)

        x = F.interpolate(original.float(), size=(nh, nw), mode="bicubic", align_corners=False, antialias=True)
        x = x.clamp(0.0, 1.0)
        x = (x - self.clip_mean) / self.clip_std
        x = F.pad(x, (0, S - nw, 0, S - nh))

        dtype = next(self.vm.parameters()).dtype
        x = x.to(dtype)

        if self.train_mode == "frozen":
            with torch.no_grad():
                out = self._safe_forward(x)
        else:
            out = self._safe_forward(x)

        tokens = out.last_hidden_state          # [B, 1 + n_patch, hidden]
        cls = tokens[:, 0]                       # [B, hidden]
        patches = tokens[:, 1:]                  # [B, n_patch, hidden]

        n_patch = patches.shape[1]
        grid = int(math.sqrt(n_patch))
        if grid * grid != n_patch:
            raise RuntimeError(f"CLIP patch-token count is not a square grid: {n_patch}")

        patches = patches.transpose(1, 2).reshape(B, self.hidden_size, grid, grid)

        if self._supports_interp:
            # Full-res path was letterboxed -> crop off the padded region.
            valid_h = math.ceil(nh / P)
            valid_w = math.ceil(nw / P)
            patches = patches[..., :valid_h, :valid_w]
        # else: fallback path resized (no pad) directly to 224x224, so the
        # whole grid is already valid.

        out = {"clip_patches": patches, "clip_cls": cls}
        return {**incoming, **out} if incoming is not None else {"img": original, **out}


# ============================================================
# ConvNeXt Encoder (third standalone ablation option)
# ============================================================

class ConvNeXtEncoder(nn.Module):
    """
    ImageNet-pretrained torchvision ConvNeXt, truncated to its
    stride-16 stage (stage3, before the final downsample+stage4) so
    its output resolution/role matches MobileSAM's - a single
    spatial feature map that `SingleEncoderPyramid` can turn into
    P3/P4/P5, for use as a standalone third ablation backbone
    alongside the CLIP-only and MobileSAM-only configs.

    Unlike MobileSAM/CLIP, ConvNeXt is fully convolutional, so there
    is no letterbox/crop step - it just runs on whatever resolution
    the dataloader feeds it (only needs to be divisible by 16).

    Args
    ----
    variant: "tiny" / "small" / "base" / "large"
    train_mode: "frozen" / "partial" / "full"
    train_blocks: in "partial" mode, how many of the truncated
        feature stack's top-level stage groups (of 6: stem, stage1,
        downsample, stage2, downsample, stage3) to leave trainable,
        counting from the end.

    Output: {"img": original RGB tensor, "convnext": feature map}
    """

    def __init__(self, variant="tiny", train_mode="frozen", train_blocks=2):
        super().__init__()

        if variant not in CONVNEXT_CTORS:
            raise ValueError(f"Unknown ConvNeXt variant '{variant}'. Choose from {list(CONVNEXT_CTORS.keys())}")
        if train_mode not in {"frozen", "partial", "full"}:
            raise ValueError(f"ConvNeXt train_mode must be frozen/partial/full. Got: {train_mode}")

        self.variant = variant
        self.train_mode = train_mode
        self.train_blocks = int(train_blocks)
        self.out_channels = CONVNEXT_CHANNELS[variant]

        import torchvision.models as tvm
        ctor = getattr(tvm, CONVNEXT_CTORS[variant])

        try:
            weights_cls = getattr(tvm, f"ConvNeXt_{variant.capitalize()}_Weights")
            model = ctor(weights=weights_cls.IMAGENET1K_V1)
        except AttributeError:
            model = ctor(pretrained=True)  # older torchvision

        # features = [stem, stage1, downsample, stage2, downsample, stage3,
        #             downsample, stage4]; keep through stage3 -> stride 16.
        self.features = nn.Sequential(*list(model.features.children())[:6])
        del model

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        self._configure_training()

    def _configure_training(self):
        if self.train_mode == "frozen":
            _set_requires_grad(self.features, False)
        elif self.train_mode == "full":
            _set_requires_grad(self.features, True)
        elif self.train_mode == "partial":
            _set_requires_grad(self.features, False)
            children = list(self.features.children())
            n = min(self.train_blocks, len(children))
            for child in children[len(children) - n:]:
                _set_requires_grad(child, True)
        _print_trainable_status("ConvNeXt", self.features)

    def train(self, mode=True):
        super().train(mode)
        self.features.eval() if self.train_mode == "frozen" else self.features.train(mode)
        return self

    def forward(self, x):
        img, incoming = _unpack_input(x)
        inp = (img - self.mean) / self.std

        if self.train_mode == "frozen":
            with torch.no_grad():
                f = self.features(inp)
        else:
            f = self.features(inp)

        out = {"convnext": f}
        return {**incoming, **out} if incoming is not None else {"img": img, **out}


# ============================================================
# Feed Forward Network
# ============================================================

class FeedForward(nn.Module):
    def __init__(self, dim, expansion=2, dropout=0.1):
        super().__init__()
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Pretrained RGB Detail Branch
# ============================================================

class PretrainedDetailStem(nn.Module):
    """
    Stride-8 RGB detail branch backed by an ImageNet-pretrained
    ResNet-18 stem (conv1 + bn1 + relu + maxpool + layer1 + layer2).

    Replaces a from-scratch 3-conv stem. Neither SAM (stride 16,
    segmentation-pretrained) nor CLIP (heavily downsampled,
    classification-pretrained) preserve fine local detail - this
    branch is what gives the P3 level that detail back, and doing
    it with real pretrained low-level filters (edges/textures)
    instead of random init is the single highest-leverage change
    in this file.

    Falls back to a from-scratch conv stem only if torchvision /
    its pretrained weights can't be loaded (e.g. no internet).
    """

    def __init__(self, out_channels=192, trainable=True, pretrained=True):
        super().__init__()
        self.out_channels = out_channels
        self.trainable = trainable
        backbone, in_channels = None, None

        if pretrained:
            try:
                import torchvision.models as tvm
                try:
                    resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
                except AttributeError:
                    resnet = tvm.resnet18(pretrained=True)  # older torchvision

                backbone = nn.Sequential(
                    resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                    resnet.layer1, resnet.layer2,
                )
                in_channels = 128
                print("[DetailStem] Using pretrained ResNet-18 conv1..layer2 (stride 8, 128ch).")
            except Exception as e:
                print(f"[DetailStem] Pretrained backbone unavailable ({e}); using a from-scratch conv stem.")
                backbone = None

        if backbone is None:
            backbone = nn.Sequential(
                ConvGNAct(3, 32, 3, 2),
                ConvGNAct(32, 64, 3, 2),
                ConvGNAct(64, 128, 3, 2),
            )
            in_channels = 128

        self.backbone = backbone
        _set_requires_grad(self.backbone, trainable)

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def train(self, mode=True):
        super().train(mode)
        if not self.trainable:
            self.backbone.eval()
        return self

    def forward(self, x):
        x = (x - self.mean) / self.std
        if self.trainable:
            f = self.backbone(x)
        else:
            with torch.no_grad():
                f = self.backbone(x)
        return self.project(f)


# ============================================================
# Gated Fusion
# ============================================================

class GatedFusion(nn.Module):
    """MobileSAM + CLIP + pretrained RGB detail fusion -> [P3, P4, P5]."""

    def __init__(self, hd=192, clip_dim=768, heads=6, dropout=0.1,
                 detail_trainable=True, detail_pretrained=True):
        super().__init__()

        if hd % heads != 0:
            raise ValueError(f"hd={hd} must be divisible by heads={heads}")

        self.hd = hd
        self.clip_dim = clip_dim
        self.heads = heads

        # --- Projections ---
        self.sam_proj = nn.Sequential(
            nn.Conv2d(256, hd, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(hd), hd),
            nn.SiLU(inplace=True),
        )
        self.clip_patch_proj = nn.Sequential(
            nn.Conv2d(clip_dim, hd, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(hd), hd),
            nn.SiLU(inplace=True),
        )
        self.cls_proj = nn.Sequential(
            nn.Linear(clip_dim, hd, bias=False),
            nn.LayerNorm(hd),
            nn.SiLU(inplace=True),
        )

        # --- Bidirectional cross attention (spatial tokens only) ---
        self.sam_to_clip = nn.MultiheadAttention(hd, heads, dropout=dropout, batch_first=True)
        self.clip_to_sam = nn.MultiheadAttention(hd, heads, dropout=dropout, batch_first=True)
        self.sam_norm1 = nn.LayerNorm(hd)
        self.clip_norm1 = nn.LayerNorm(hd)

        self.sam_ffn = FeedForward(hd, expansion=2, dropout=dropout)
        self.clip_ffn = FeedForward(hd, expansion=2, dropout=dropout)
        self.sam_norm2 = nn.LayerNorm(hd)
        self.clip_norm2 = nn.LayerNorm(hd)

        # --- Spatial gate ---
        gate_mid = max(64, hd // 2)
        self.gate = nn.Sequential(
            nn.Conv2d(hd * 2, gate_mid, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(gate_mid), gate_mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_mid, hd, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.gate[-2].bias)  # start gate ~0.5

        self.fuse = nn.Sequential(ConvGNAct(hd, hd, 3, 1), ConvGNAct(hd, hd, 3, 1))

        # --- RGB detail branch (pretrained) ---
        self.detail = PretrainedDetailStem(hd, trainable=detail_trainable, pretrained=detail_pretrained)

        # --- P3/P4/P5 heads ---
        self.p3_refine = nn.Sequential(ConvGNAct(hd, hd, 3, 1), ConvGNAct(hd, hd, 3, 1))
        self.down4 = ConvGNAct(hd, hd, 3, 2)
        self.p4_refine = nn.Sequential(ConvGNAct(hd, hd, 3, 1), ConvGNAct(hd, hd, 3, 1))
        self.down5 = ConvGNAct(hd, hd, 3, 2)
        self.p5_refine = nn.Sequential(ConvGNAct(hd, hd, 3, 1), ConvGNAct(hd, hd, 3, 1))

    @staticmethod
    def _resize(x, size):
        h, w = x.shape[-2:]
        th, tw = size
        if h == th and w == tw:
            return x
        if th <= h and tw <= w:
            return F.adaptive_avg_pool2d(x, output_size=(th, tw))
        return F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)

    def forward(self, d):
        img = d["img"]
        sam = d["mobile_sam"]
        clip_patches = d["clip_patches"]
        clip_cls = d["clip_cls"]

        # --- Project to shared dim ---
        sam_map = self.sam_proj(sam)
        B, _, Hs, Ws = sam_map.shape
        sam_tokens = sam_map.flatten(2).transpose(1, 2)

        clip_map0 = self.clip_patch_proj(clip_patches)
        _, _, Hc, Wc = clip_map0.shape
        clip_tokens = clip_map0.flatten(2).transpose(1, 2)

        # --- Bidirectional cross-attention (sequence lengths may differ) ---
        sam_attn, _ = self.sam_to_clip(query=sam_tokens, key=clip_tokens, value=clip_tokens, need_weights=False)
        sam_tokens = self.sam_norm1(sam_tokens + sam_attn)

        clip_attn, _ = self.clip_to_sam(query=clip_tokens, key=sam_tokens, value=sam_tokens, need_weights=False)
        clip_tokens = self.clip_norm1(clip_tokens + clip_attn)

        sam_tokens = self.sam_norm2(sam_tokens + self.sam_ffn(sam_tokens))
        clip_tokens = self.clip_norm2(clip_tokens + self.clip_ffn(clip_tokens))

        sam_map = sam_tokens.transpose(1, 2).reshape(B, self.hd, Hs, Ws)
        clip_map = clip_tokens.transpose(1, 2).reshape(B, self.hd, Hc, Wc)
        clip_map = self._resize(clip_map, (Hs, Ws))

        # --- Gated fusion + global CLIP context ---
        gate = self.gate(torch.cat([sam_map, clip_map], dim=1))
        fused = gate * clip_map + (1.0 - gate) * sam_map

        global_context = self.cls_proj(clip_cls).unsqueeze(-1).unsqueeze(-1)
        fused = self.fuse(fused + global_context)

        # --- RGB detail (pretrained, stride 8) drives P3 ---
        detail = self.detail(img)
        p3 = self.p3_refine(self._resize(fused, detail.shape[-2:]) + detail)

        p4 = self.down4(p3)
        p4 = self.p4_refine(p4 + self._resize(fused, p4.shape[-2:]))

        p5 = self.down5(p4)
        p5 = self.p5_refine(p5 + self._resize(fused, p5.shape[-2:]))

        return [p3, p4, p5]


# ============================================================
# Pair Fusion (generic two-encoder gated fusion, any pairing)
# ============================================================

class PairFusion(nn.Module):
    """
    Generic two-encoder gated fusion - same bidirectional
    cross-attention + spatial gate design as GatedFusion, but not
    hardcoded to MobileSAM+CLIP, so it can pair ConvNeXt with
    either of them (or, in principle, any two spatial-map encoders).

    `key1` is the reference resolution: `key2`'s map is resized to
    match it before gating. Pick whichever branch has the *smaller*
    native grid as key1, so key2 gets downsampled (clean average
    pooling) into it rather than key1 getting upsampled (blurry)
    into key2's size.

    If one branch has a global/CLS-style vector (CLIP does;
    ConvNeXt/MobileSAM don't), pass cls_key + cls_dim and it's added
    as an additive global context term, same as in GatedFusion.
    Leave both as None for a pairing where neither branch has one.
    """

    def __init__(
        self,
        hd=256,
        c1=384,
        c2=256,
        key1="convnext",
        key2="mobile_sam",
        heads=8,
        dropout=0.1,
        cls_key=None,
        cls_dim=None,
        detail_trainable=True,
        detail_pretrained=True,
    ):
        super().__init__()

        if hd % heads != 0:
            raise ValueError(f"hd={hd} must be divisible by heads={heads}")

        self.hd = hd
        self.key1 = key1
        self.key2 = key2
        self.cls_key = cls_key

        self.proj1 = nn.Sequential(
            nn.Conv2d(c1, hd, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(hd), hd),
            nn.SiLU(inplace=True),
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(c2, hd, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(hd), hd),
            nn.SiLU(inplace=True),
        )

        if cls_key is not None:
            if cls_dim is None:
                raise ValueError("cls_dim is required when cls_key is set")
            self.cls_proj = nn.Sequential(
                nn.Linear(cls_dim, hd, bias=False),
                nn.LayerNorm(hd),
                nn.SiLU(inplace=True),
            )
        else:
            self.cls_proj = None

        self.a_to_b = nn.MultiheadAttention(hd, heads, dropout=dropout, batch_first=True)
        self.b_to_a = nn.MultiheadAttention(hd, heads, dropout=dropout, batch_first=True)
        self.norm_a1 = nn.LayerNorm(hd)
        self.norm_b1 = nn.LayerNorm(hd)

        self.ffn_a = FeedForward(hd, expansion=2, dropout=dropout)
        self.ffn_b = FeedForward(hd, expansion=2, dropout=dropout)
        self.norm_a2 = nn.LayerNorm(hd)
        self.norm_b2 = nn.LayerNorm(hd)

        gate_mid = max(64, hd // 2)
        self.gate = nn.Sequential(
            nn.Conv2d(hd * 2, gate_mid, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(gate_mid), gate_mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_mid, hd, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.gate[-2].bias)  # start gate ~0.5

        self.fuse = nn.Sequential(ConvGNAct(hd, hd, 3, 1), ConvGNAct(hd, hd, 3, 1))
        self.detail = PretrainedDetailStem(hd, trainable=detail_trainable, pretrained=detail_pretrained)

        self.p3_refine = nn.Sequential(ConvGNAct(hd, hd, 3, 1), ConvGNAct(hd, hd, 3, 1))
        self.down4 = ConvGNAct(hd, hd, 3, 2)
        self.p4_refine = nn.Sequential(ConvGNAct(hd, hd, 3, 1), ConvGNAct(hd, hd, 3, 1))
        self.down5 = ConvGNAct(hd, hd, 3, 2)
        self.p5_refine = nn.Sequential(ConvGNAct(hd, hd, 3, 1), ConvGNAct(hd, hd, 3, 1))

    @staticmethod
    def _resize(x, size):
        h, w = x.shape[-2:]
        th, tw = size
        if h == th and w == tw:
            return x
        if th <= h and tw <= w:
            return F.adaptive_avg_pool2d(x, output_size=(th, tw))
        return F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)

    def forward(self, d):
        img = d["img"]
        map1 = d[self.key1]
        map2 = d[self.key2]

        proj1 = self.proj1(map1)
        B, _, H1, W1 = proj1.shape
        tok1 = proj1.flatten(2).transpose(1, 2)

        proj2 = self.proj2(map2)
        _, _, H2, W2 = proj2.shape
        tok2 = proj2.flatten(2).transpose(1, 2)

        attn_a, _ = self.a_to_b(query=tok1, key=tok2, value=tok2, need_weights=False)
        tok1 = self.norm_a1(tok1 + attn_a)

        attn_b, _ = self.b_to_a(query=tok2, key=tok1, value=tok1, need_weights=False)
        tok2 = self.norm_b1(tok2 + attn_b)

        tok1 = self.norm_a2(tok1 + self.ffn_a(tok1))
        tok2 = self.norm_b2(tok2 + self.ffn_b(tok2))

        map1f = tok1.transpose(1, 2).reshape(B, self.hd, H1, W1)
        map2f = tok2.transpose(1, 2).reshape(B, self.hd, H2, W2)
        map2f = self._resize(map2f, (H1, W1))

        gate = self.gate(torch.cat([map1f, map2f], dim=1))
        fused = gate * map2f + (1.0 - gate) * map1f

        if self.cls_key is not None:
            global_context = self.cls_proj(d[self.cls_key]).unsqueeze(-1).unsqueeze(-1)
            fused = fused + global_context

        fused = self.fuse(fused)

        detail = self.detail(img)
        p3 = self.p3_refine(self._resize(fused, detail.shape[-2:]) + detail)

        p4 = self.down4(p3)
        p4 = self.p4_refine(p4 + self._resize(fused, p4.shape[-2:]))

        p5 = self.down5(p4)
        p5 = self.p5_refine(p5 + self._resize(fused, p5.shape[-2:]))

        return [p3, p4, p5]


# ============================================================
# Single Encoder Feature Pyramid (ablation helper)
# ============================================================

class SingleEncoderPyramid(nn.Module):
    """
    Converts a single encoder's features into P3/P4/P5, for ablations.

    PATCHED: now optionally uses the same PretrainedDetailStem that
    GatedFusion/PairFusion use, so single-encoder ablations can also
    get a fine-detail P3 signal from the raw RGB image rather than
    relying purely on cascaded downsampling from one coarse map.

    ConvNeXt is excluded by default (it's a CNN and already preserves
    fine spatial detail natively via its own conv layers, so it
    doesn't have the problem the detail branch exists to fix). Remove
    the `self.input_type != "convnext"` condition in forward() below
    if you want ConvNeXt-only to use it too.
    """

    def __init__(self, in_channels, out_channels=192, input_type="mobile_sam",
                 detail_trainable=True, detail_pretrained=True):
        super().__init__()
        self.input_type = input_type
        self.out_channels = out_channels

        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

        # NEW: same reusable detail branch GatedFusion/PairFusion already use
        self.detail = PretrainedDetailStem(out_channels, trainable=detail_trainable,
                                            pretrained=detail_pretrained)

        self.p3 = nn.Sequential(ConvGNAct(out_channels, out_channels, 3, 1))
        self.p4 = nn.Sequential(ConvGNAct(out_channels, out_channels, 3, 2))
        self.p5 = nn.Sequential(ConvGNAct(out_channels, out_channels, 3, 2))

    @staticmethod
    def _resize(x, size):
        h, w = x.shape[-2:]
        th, tw = size
        if h == th and w == tw:
            return x
        if th <= h and tw <= w:
            return F.adaptive_avg_pool2d(x, output_size=(th, tw))
        return F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)

    def forward(self, x):
        img = None
        if isinstance(x, dict):
            img = x["img"]  # NEW: keep the raw image before narrowing to one key
            if self.input_type == "mobile_sam":
                x = x["mobile_sam"]
            elif self.input_type == "clip":
                x = x["clip_patches"]
            elif self.input_type == "convnext":
                x = x["convnext"]

        if x.ndim == 3:
            B, N, C = x.shape
            if int(math.sqrt(N - 1)) ** 2 == N - 1:
                x = x[:, 1:]  # drop CLS if present
                N -= 1
            grid = int(math.sqrt(N))
            if grid * grid != N:
                raise RuntimeError(f"Cannot convert {N} tokens into a square spatial grid.")
            x = x.transpose(1, 2).reshape(B, C, grid, grid)

        x = self.project(x)

        # NEW: detail branch drives P3's resolution, same pattern as GatedFusion.
        # Excluded for ConvNeXt by default - see class docstring.
        if img is not None and self.input_type != "convnext":
            detail = self.detail(img)
            p3 = self.p3(self._resize(x, detail.shape[-2:]) + detail)
        else:
            p3 = self.p3(x)

        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return [p3, p4, p5]


# ============================================================
# Differential learning-rate helper
# ============================================================

def get_param_groups(model, pretrained_lr=1e-5, new_lr=1e-4, weight_decay=0.05):
    """
    Split model parameters into two optimizer groups:

        params = get_param_groups(model, pretrained_lr=1e-5, new_lr=1e-4)
        optimizer = torch.optim.AdamW(params, weight_decay=0.05)

    Pretrained encoders (MobileSAM, CLIP, the ResNet detail stem)
    should move slowly; the new fusion/gate/projection/detection-head
    modules are randomly initialized and need a much higher LR.
    Training everything at one LR is a common reason a hybrid
    backbone trains slower than a from-scratch one.

    NOTE: if you're calling `model.train(...)` on an Ultralytics
    `RTDETR` object directly, Ultralytics builds its own optimizer
    internally and this won't be wired in automatically - you'd need
    to either subclass the trainer and override `build_optimizer`,
    or use a custom training loop that calls this directly.
    """
    pretrained_keywords = ("sam_encoder.encoder", "clip_encoder.vm", "detail.backbone")

    pretrained_params, new_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (pretrained_params if any(k in name for k in pretrained_keywords) else new_params).append(p)

    groups = []
    if pretrained_params:
        groups.append({"params": pretrained_params, "lr": pretrained_lr, "weight_decay": weight_decay})
    if new_params:
        groups.append({"params": new_params, "lr": new_lr, "weight_decay": weight_decay})

    print(f"[ParamGroups] pretrained: {len(pretrained_params)} tensors @ lr={pretrained_lr} "
          f"| new: {len(new_params)} tensors @ lr={new_lr}")
    return groups


# ============================================================
# Ultralytics registration
# ============================================================

try:
    import ultralytics.nn.tasks as _tasks

    _tasks.MobileSAMEncoder = MobileSAMEncoder
    _tasks.CLIPEncoder = CLIPEncoder
    _tasks.ConvNeXtEncoder = ConvNeXtEncoder
    _tasks.GatedFusion = GatedFusion
    _tasks.PairFusion = PairFusion
    _tasks.SingleEncoderPyramid = SingleEncoderPyramid

    print("[HybridModules] Registered: MobileSAMEncoder, CLIPEncoder, ConvNeXtEncoder, "
          "GatedFusion, PairFusion, SingleEncoderPyramid")
except Exception as e:
    print(f"[HybridModules] Registration failed: {e}")


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":
    print("\nTesting hybrid modules (v2, patched)...\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    x = torch.rand(1, 3, 640, 640, device=device)

    # --- Original hybrid model sanity check (unchanged) ---
    sam_encoder = MobileSAMEncoder(ckpt=None, train_mode="partial", train_blocks=2).to(device)
    clip_encoder = CLIPEncoder(variant="base16", train_mode="partial", train_blocks=4, input_res=336).to(device)
    fusion = GatedFusion(hd=192, clip_dim=768, heads=6, dropout=0.1,
                          detail_trainable=True, detail_pretrained=True).to(device)

    if sam_encoder.train_mode == "frozen" and clip_encoder.train_mode == "frozen":
        sam_encoder.eval()
        clip_encoder.eval()
    fusion.train()

    d = sam_encoder(x)
    d = clip_encoder(d)
    outputs = fusion(d)

    print("\nHybrid model output shapes:")
    for i, y in enumerate(outputs):
        print(f"P{i + 3}: {tuple(y.shape)}")

    # --- NEW: MobileSAM-only + detail branch sanity check ---
    print("\n--- Testing patched SingleEncoderPyramid (MobileSAM-only + detail) ---")
    sam_only = MobileSAMEncoder(ckpt=None, train_mode="full", train_blocks=0).to(device)
    pyramid = SingleEncoderPyramid(in_channels=256, out_channels=256, input_type="mobile_sam",
                                    detail_trainable=True, detail_pretrained=True).to(device)
    sam_only.train()
    pyramid.train()

    d2 = sam_only(x)
    outputs2 = pyramid(d2)

    print("\nMobileSAM-only (+detail) output shapes:")
    for i, y in enumerate(outputs2):
        print(f"P{i + 3}: {tuple(y.shape)}")

    print("\nTrainable parameters:")
    _print_trainable_status("MobileSAM (standalone)", sam_only)
    _print_trainable_status("SingleEncoderPyramid (incl. detail stem)", pyramid)

    print("\nTest complete.")