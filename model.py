import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
import numpy as np
from torchvision.io import read_video
from torchvision import transforms
import os
import torch.hub


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    if embed_dim == 0:
        return np.zeros((pos.size, 0))
    if embed_dim % 2 != 0:
        embed_dim = (embed_dim // 2) * 2
    if embed_dim == 0:
        return np.zeros((pos.size, 0))
    omega = np.arange(embed_dim // 2, dtype=float) / (embed_dim / 2.0)
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_3d_sincos_pos_embed(
    embed_dim, grid_size_s, grid_size_t, cls_token=False, uniform_power=False
):
    grid_t, grid_h, grid_w = np.meshgrid(
        np.arange(grid_size_t, dtype=float),
        np.arange(grid_size_s, dtype=float),
        np.arange(grid_size_s, dtype=float),
        indexing="ij",
    )
    if not uniform_power:
        dim_t_target, dim_s_each_target = embed_dim // 2, embed_dim // 4
        dim_t, dim_s_each = (dim_t_target // 2) * 2, (dim_s_each_target // 2) * 2
        rem = embed_dim - (dim_t + 2 * dim_s_each)
        if rem > 0:
            dim_t += (rem // 2) * 2
        rem = embed_dim - (dim_t + 2 * dim_s_each)
        if rem > 0 and dim_s_each * 2 + dim_t < embed_dim:
            dim_s_each += (rem // 4) * 2
            rem = embed_dim - (dim_t + 2 * dim_s_each)
            if rem > 0:
                dim_t += (rem // 2) * 2
        parts = [
            get_1d_sincos_pos_embed_from_grid(d, g.flatten())
            for d, g in zip([dim_t, dim_s_each, dim_s_each], [grid_t, grid_h, grid_w])
            if d > 0
        ]
    else:
        comp_dim = (int(np.ceil(embed_dim / 3.0)) // 2) * 2
        parts = [
            get_1d_sincos_pos_embed_from_grid(comp_dim, g.flatten())
            for g in [grid_t, grid_h, grid_w]
            if comp_dim > 0
        ]
    pos_embed = np.concatenate(parts, axis=1) if parts else np.zeros((grid_t.size, 0))
    if pos_embed.shape[1] < embed_dim:
        pos_embed = np.concatenate(
            [pos_embed, np.zeros((pos_embed.shape[0], embed_dim - pos_embed.shape[1]))],
            axis=1,
        )
    elif pos_embed.shape[1] > embed_dim:
        pos_embed = pos_embed[:, :embed_dim]
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features, hidden_features = (
            out_features or in_features,
            hidden_features or in_features,
        )
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
    ):
        super().__init__()
        self.num_heads, head_dim = num_heads, dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_prob, self.proj = attn_drop, nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = True

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop_prob if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, use_sdpa
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=16, tubelet_size=2, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class AttentivePooler(nn.Module):
    def __init__(
        self,
        num_queries=1,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        use_sdpa=True,
    ):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    use_sdpa=use_sdpa,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.cross_attention = Attention(
            dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, use_sdpa=use_sdpa
        )
        self.norm_q = norm_layer(embed_dim)
        self.norm_x = norm_layer(embed_dim)
        # Initialization logic removed

    def forward(self, x_patches):
        B = x_patches.shape[0]
        queries = self.query_tokens.expand(B, -1, -1)
        q_norm = self.norm_q(queries)
        x_norm = self.norm_x(x_patches)
        attn_weights = F.softmax(
            (q_norm @ x_norm.transpose(-2, -1)) / math.sqrt(q_norm.size(-1)), dim=-1
        )
        out = attn_weights @ x_norm
        for blk in self.blocks:
            out = blk(out)
        return out


PRETRAINED_MODELS = {
    "vith16_384": {
        "url": "https://dl.fbaipublicfiles.com/jepa/vith16-384/vith16-384.pth.tar",
        "config": {
            "img_size": 384,
            "patch_size": 16,
            "num_frames": 16,
            "tubelet_size": 2,
            "embed_dim": 1280,
            "depth": 32,
            "num_heads": 16,
            "mlp_ratio": 4.0,
            "uniform_power": True,
            "use_sdpa": True,
        },
        "checkpoint_key": "target_encoder",
    }
}


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=384,
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        in_chans=3,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_std=0.02,
        uniform_power=True,
        use_sdpa=True,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.img_size_init, self.patch_size = img_size, patch_size
        self.num_frames_init, self.tubelet_size = num_frames, tubelet_size
        self.patch_embed = PatchEmbed3D(patch_size, tubelet_size, in_chans, embed_dim)
        self.grid_size_s_init = self.img_size_init // patch_size
        self.grid_size_t_init = self.num_frames_init // tubelet_size
        self.num_patches_init = self.grid_size_t_init * self.grid_size_s_init**2
        self.uniform_power = uniform_power
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches_init, embed_dim), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    qk_scale,
                    drop_rate,
                    attn_drop_rate,
                    norm_layer=norm_layer,
                    use_sdpa=use_sdpa,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # Initialization logic removed

    def interpolate_pos_encoding(self, video_shape, current_pe):
        T_in, H_in, W_in = video_shape[-3:]
        T_curr_p, H_curr_p, W_curr_p = (
            T_in // self.tubelet_size,
            H_in // self.patch_size,
            W_in // self.patch_size,
        )
        N_curr_total_p = T_curr_p * H_curr_p * W_curr_p
        if (
            N_curr_total_p == self.num_patches_init
            and T_curr_p == self.grid_size_t_init
            and H_curr_p == self.grid_size_s_init
            and W_curr_p == self.grid_size_s_init
        ):
            return current_pe
        pe_r = current_pe.reshape(
            1,
            self.grid_size_t_init,
            self.grid_size_s_init,
            self.grid_size_s_init,
            self.embed_dim,
        ).permute(0, 4, 1, 2, 3)
        pe_i = F.interpolate(
            pe_r,
            size=(T_curr_p, H_curr_p, W_curr_p),
            mode="trilinear",
            align_corners=False,
        )
        return pe_i.permute(0, 2, 3, 4, 1).reshape(1, N_curr_total_p, self.embed_dim)

    def forward(self, video_tensor, attn_mask=None):
        interp_pe = self.interpolate_pos_encoding(video_tensor.shape, self.pos_embed)
        x = self.patch_embed(video_tensor) + interp_pe
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        return self.norm(x)

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        if model_name_or_path in PRETRAINED_MODELS:
            info = PRETRAINED_MODELS[model_name_or_path]
            cfg, url, key = info["config"], info["url"], info["checkpoint_key"]
            print(f"Loading pretrained '{model_name_or_path}' from {url}")
            model_kwargs = {**cfg, **kwargs}
            model = cls(**model_kwargs)
            ckpt = torch.hub.load_state_dict_from_url(
                url, map_location="cpu", progress=True
            )
            sd = ckpt.get(key, ckpt.get("encoder", ckpt.get("state_dict", ckpt)))
            csd = {
                k.replace("module.", "").replace("backbone.", ""): v
                for k, v in sd.items()
            }
            msg = model.load_state_dict(csd, strict=False)
            print(f"State_dict loading: {msg}")
        elif os.path.exists(model_name_or_path):
            print(
                f"Loading from local: {model_name_or_path}. Ensure config via kwargs."
            )
            model = cls(**kwargs)
            ckpt = torch.load(model_name_or_path, map_location="cpu")
            sd = ckpt.get("target_encoder", ckpt.get("encoder", ckpt))
            csd = {
                k.replace("module.", "").replace("backbone.", ""): v
                for k, v in sd.items()
            }
            msg = model.load_state_dict(csd, strict=False)
            print(f"State_dict loading (local): {msg}")
        else:
            raise ValueError(f"'{model_name_or_path}' not recognized or found.")
        model.eval()
        return model


def preprocess_video(
    video_path, num_frames_to_sample=16, frame_sample_rate=4, crop_size=384
):
    try:
        vframes, _, info = read_video(video_path, pts_unit="sec", output_format="TCHW")
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return None
    total_frames = vframes.shape[0]
    if total_frames == 0:
        print(f"Video {video_path} has 0 frames.")
        return None
    indices = []
    required_original_span = (num_frames_to_sample - 1) * frame_sample_rate + 1
    if total_frames >= required_original_span:
        start_idx_original_video = (total_frames - required_original_span) // 2
        for i in range(num_frames_to_sample):
            indices.append(start_idx_original_video + i * frame_sample_rate)
    else:
        print(
            f"Warning: Video is shorter ({total_frames} frames) than required span ({required_original_span}). Adapting sampling."
        )
        last_valid_original_idx = 0
        for i in range(num_frames_to_sample):
            original_idx_ideal = i * frame_sample_rate
            actual_original_idx = min(original_idx_ideal, total_frames - 1)
            indices.append(actual_original_idx)
            last_valid_original_idx = actual_original_idx
    selected_vframes = vframes[indices]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    frame_transform = transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize([crop_size, crop_size], antialias=True),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    transformed_frames_list = [frame_transform(frame) for frame in selected_vframes]
    transformed_frames_tensor = torch.stack(transformed_frames_list)
    final_video_tensor = transformed_frames_tensor.permute(1, 0, 2, 3)
    return final_video_tensor.unsqueeze(0)


if __name__ == "__main__":
    pretrained_model_name = "vith16_384"
    print(f"Attempting to load pretrained model: {pretrained_model_name}")
    try:
        model = VisionTransformer.from_pretrained(pretrained_model_name)
    except Exception as e:
        print(f"Failed to load pretrained model: {e}")
        print("Exiting.")
        exit()
    model.eval()
    print("Pretrained model loaded successfully and set to evaluation mode.")
    dummy_video_path = "dummy_video_for_vjepa.mp4"
    try:
        dummy_video_data = torch.randint(0, 255, (70, 3, 240, 320), dtype=torch.uint8)
        from torchvision.io import write_video

        write_video(dummy_video_path, dummy_video_data.permute(0, 2, 3, 1), fps=30)
        print(f"Created dummy video: {dummy_video_path}")
        video_to_process = dummy_video_path
    except Exception as e:
        print(
            f"Could not create dummy video: {e}. Trying to use a user-provided MP4 path."
        )
        user_video_path = input(
            "Enter path to an MP4 video file (or press Enter to skip): "
        ).strip()
        if user_video_path and os.path.exists(user_video_path):
            video_to_process = user_video_path
            print(f"Using user-provided video: {video_to_process}")
        else:
            print("No valid video path provided. Exiting.")
            video_to_process = None
            exit()
    if video_to_process:
        model_config = PRETRAINED_MODELS[pretrained_model_name]["config"]
        video_tensor = preprocess_video(
            video_to_process,
            num_frames_to_sample=model_config["num_frames"],
            frame_sample_rate=4,
            crop_size=model_config["img_size"],
        )
        if video_tensor is not None:
            print(f"Preprocessed video tensor shape: {video_tensor.shape}")
            print("Performing inference with pretrained model...")
            with torch.no_grad():
                output_features = model(video_tensor)
            print(f"Model output features shape: {output_features.shape}")
        else:
            print(f"Could not process video: {video_to_process}")
    if video_to_process == dummy_video_path and os.path.exists(dummy_video_path):
        os.remove(dummy_video_path)
        print(f"Removed dummy video: {dummy_video_path}")
