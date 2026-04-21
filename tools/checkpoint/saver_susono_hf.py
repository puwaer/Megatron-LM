"""
キュー → Susono HuggingFace チェックポイント セーバープラグイン

Megatron の tools/checkpoint/convert.py フレームワーク用。
キューから受け取った重みを HuggingFace safetensors 形式で保存する。
既存の susono_convert.py の変換ロジックをプラグイン化したもの。

使い方:
  python convert.py \\
      --model-type GPT \\
      --loader susono_mcore \\
      --saver susono_hf \\
      --load-dir /path/to/iter_NNNNNNN \\
      --save-dir /path/to/output_hf \\
      --susono-config /path/to/config.json \\
      [--susono-tokenizer-dir /path/to/tokenizer_dir]
"""

import json
import os
import shutil

import torch


# ──────────────────────────────────────────────────────────────────────────────
# プラグインインターフェース
# ──────────────────────────────────────────────────────────────────────────────

def add_arguments(parser):
    group = parser.add_argument_group(title='Susono HF saver')
    group.add_argument('--susono-tokenizer-dir', type=str, default=None,
                       help='トークナイザーファイルのあるディレクトリ (出力先にコピーされる)')
    group.add_argument('--susono-config', type=str, default=None,
                       help='config.json テンプレートのパス')


def save_checkpoint(queue, args):
    """キューから重みを受け取り、HF safetensors 形式で保存する"""
    _save_checkpoint_impl(queue, args)


# ──────────────────────────────────────────────────────────────────────────────
# 実装
# ──────────────────────────────────────────────────────────────────────────────

def _save_checkpoint_impl(queue, args):
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # ── メタデータ受信 ────────────────────────────────────────────────────
    md = queue.get()
    if md == 'exit':
        print('[saver_susono_hf] ローダーがエラーで終了しました。')
        return

    num_layers         = md.num_layers
    num_attn_heads     = md.num_attention_heads
    num_kv_heads       = md.num_key_value_heads
    head_dim           = md.head_dim
    full_attn_interval = md.full_attention_interval
    engram_layer_ids   = md.engram_layer_ids

    full_attn_layers  = {i for i in range(num_layers) if (i + 1) % full_attn_interval == 0}
    engram_by_layer   = {layer_idx: j for j, layer_idx in enumerate(engram_layer_ids)}

    hf = {}

    # stream_proj はメタデータに埋め込まれている
    if getattr(md, 'stream_proj_weight', None) is not None:
        hf['model.stream_proj.weight'] = md.stream_proj_weight

    def _check(msg, expected_name):
        if getattr(args, 'checking', True) and isinstance(msg, dict):
            actual = msg.get('name', '')
            if actual != expected_name:
                raise ValueError(
                    f'キュープロトコルエラー: "{expected_name}" を期待しましたが "{actual}" を受信しました。'
                )

    # ── 埋め込み ──────────────────────────────────────────────────────────
    msg = queue.get()
    _check(msg, 'embeddings')
    emb_w = msg['word embeddings']
    hf['model.embed_tokens.weight'] = emb_w

    # ── 各層 ──────────────────────────────────────────────────────────────
    for i in range(num_layers):
        msg = queue.get()
        _check(msg, f'transformer layer {i}')
        dst     = f'model.layers.{i}'
        is_full = i in full_attn_layers

        # MHC-Lite パラメータ
        for p in ['norm.weight', 'static_alpha', 'dynamic_alpha_fn', 'pre_branch_scale',
                  'residual_scale', 'static_beta', 'dynamic_beta_fn', 'h_post_scale']:
            key = f'mhc.{p}'
            if key in msg:
                hf[f'model.mhc_modules.{i}.{p}'] = msg[key]

        if is_full:
            # --- Full Attention 層 ---
            # Megatron: layernorm_zero_centered_gamma=True のため重みは 0 初期化済み (1+w 形式)
            # HF SusonoRMSNorm も同一の 0 初期化 (1+w 形式) → オフセット不要
            hf[f'{dst}.input_layernorm.weight']          = msg['input norm weight']
            hf[f'{dst}.post_attention_layernorm.weight'] = msg['post norm weight']

            # QKV 分割 (GQA インターリーブを解除)
            # attention_output_gate=True の場合、mcore linear_qkv per-group layout は
            # [q_heads_per_group, gate_heads_per_group, k_head, v_head]
            # → HF 側は q/gate を per-head interleave して q_proj.weight (2x 出力) に結合、
            #   k_proj / v_proj / o_proj は通常通り
            qkv_w   = msg['qkv weight']
            hidden  = qkv_w.shape[1]
            num_q_per_kv = num_attn_heads // num_kv_heads
            total_heads_per_group = qkv_w.shape[0] // (num_kv_heads * head_dim)
            has_gate = total_heads_per_group == (2 * num_q_per_kv + 2)
            if has_gate:
                qkv_r = qkv_w.view(num_kv_heads, 2 * num_q_per_kv + 2, head_dim, hidden)
                q    = qkv_r[:, :num_q_per_kv]                    # [kv, q_per_kv, head_dim, hidden]
                gate = qkv_r[:, num_q_per_kv:2 * num_q_per_kv]
                k_w  = qkv_r[:, 2 * num_q_per_kv:2 * num_q_per_kv + 1]
                v    = qkv_r[:, 2 * num_q_per_kv + 1:]
                # HF per-head layout: [q_head(head_dim), gate_head(head_dim)] interleaved
                q_and_gate = torch.stack([q, gate], dim=2)  # [kv, q_per_kv, 2, head_dim, hidden]
                hf[f'{dst}.self_attn.q_proj.weight'] = (
                    q_and_gate.reshape(num_attn_heads * head_dim * 2, hidden).contiguous()
                )
            else:
                qkv_r = qkv_w.view(num_kv_heads, num_q_per_kv + 2, head_dim, hidden)
                q   = qkv_r[:, :num_q_per_kv]
                k_w = qkv_r[:, num_q_per_kv:num_q_per_kv + 1]
                v   = qkv_r[:, num_q_per_kv + 1:]
                hf[f'{dst}.self_attn.q_proj.weight'] = (
                    q.reshape(num_attn_heads * head_dim, hidden).contiguous()
                )
            hf[f'{dst}.self_attn.k_proj.weight'] = (
                k_w.reshape(num_kv_heads * head_dim, hidden).contiguous()
            )
            hf[f'{dst}.self_attn.v_proj.weight'] = (
                v.reshape(num_kv_heads * head_dim, hidden).contiguous()
            )
            hf[f'{dst}.self_attn.o_proj.weight'] = msg['dense weight']
            # QK LayerNorm (optional)
            if 'q layernorm weight' in msg:
                hf[f'{dst}.self_attn.q_norm.weight'] = msg['q layernorm weight']
            if 'k layernorm weight' in msg:
                hf[f'{dst}.self_attn.k_norm.weight'] = msg['k layernorm weight']

            # MLP: Dense (SwiGLU split) or MoE (mlp.* passthrough).
            if 'mlp l0 weight' in msg:
                # Dense SwiGLU: split linear_fc1 → gate_proj + up_proj
                fc1  = msg['mlp l0 weight']
                gate_m, up_m = torch.chunk(fc1, 2, dim=0)
                hf[f'{dst}.mlp.gate_proj.weight'] = gate_m.contiguous()
                hf[f'{dst}.mlp.up_proj.weight']   = up_m.contiguous()
                hf[f'{dst}.mlp.down_proj.weight']  = msg['mlp l1 weight']
            else:
                # MoE: pass through all mlp.* keys as-is.
                for key, val in msg.items():
                    if key.startswith('mlp.'):
                        hf[f'{dst}.{key}'] = val

        else:
            # --- Linear Attention 層 ---
            hf[f'{dst}.input_layernorm.weight']          = msg['input norm weight']
            hf[f'{dst}.post_attention_layernorm.weight'] = msg['post norm weight']

            for key, val in msg.items():
                if key.startswith('linear_attn.') or key.startswith('mlp.'):
                    hf[f'{dst}.{key}'] = val

        # Engram パラメータ
        if i in engram_by_layer:
            j = engram_by_layer[i]
            for key, val in msg.items():
                if key.startswith('engram.'):
                    subkey = key[len('engram.'):]
                    hf[f'model.engram_modules.{j}.{subkey}'] = val

    # ── 最終層ノルム ──────────────────────────────────────────────────────
    msg = queue.get()
    _check(msg, 'final layer norm')
    # Megatron: layernorm_zero_centered_gamma=True のため重みは 0 初期化済み (1+w 形式)
    # HF SusonoRMSNorm も同一の 0 初期化 (1+w 形式) → オフセット不要
    hf['model.norm.weight'] = msg['weight']

    # ── 出力層 ────────────────────────────────────────────────────────────
    msg = queue.get()
    if msg != 'done' and isinstance(msg, dict) and msg.get('name') == 'output layer':
        hf['lm_head.weight'] = msg['weight']
        msg = queue.get()
    elif getattr(md, 'tie_word_embeddings', False):
        hf['lm_head.weight'] = emb_w.clone()

    if msg != 'done':
        raise RuntimeError(f'キュープロトコルエラー: "done" を期待しましたが {msg!r} を受信しました。')

    print(f'[saver_susono_hf] state_dict キー数: {len(hf)}')

    # ── BF16 へキャスト ───────────────────────────────────────────────────
    hf_bf16 = {}
    for k, v in hf.items():
        if v.is_floating_point():
            hf_bf16[k] = v.to(torch.bfloat16).contiguous()
        else:
            hf_bf16[k] = v.contiguous()

    # ── safetensors 保存 ──────────────────────────────────────────────────
    _save_safetensors(hf_bf16, save_dir)

    # ── config.json 書き込み ──────────────────────────────────────────────
    _write_config(save_dir, md, args)

    # ── トークナイザーファイルをコピー ────────────────────────────────────
    tokenizer_dir = getattr(args, 'susono_tokenizer_dir', None)
    if tokenizer_dir:
        _copy_tokenizer(tokenizer_dir, save_dir)

    print(f'[saver_susono_hf] 保存完了: {save_dir}')


# ──────────────────────────────────────────────────────────────────────────────
# ヘルパー関数
# ──────────────────────────────────────────────────────────────────────────────

def _save_safetensors(state_dict: dict, save_dir: str) -> None:
    """safetensors 形式で保存する (5 GB を超える場合はシャーディング)"""
    SHARD_SIZE = 5 * 1024 ** 3

    def _tensor_bytes(t):
        return t.nelement() * t.element_size()

    total_bytes = sum(_tensor_bytes(v) for v in state_dict.values())

    try:
        from safetensors.torch import save_file

        if total_bytes <= SHARD_SIZE:
            path = os.path.join(save_dir, 'model.safetensors')
            save_file(state_dict, path)
            print(f'[saver_susono_hf] 単一ファイルで保存: {path}')
        else:
            shards, cur_shard, cur_bytes = [], {}, 0
            for key, tensor in state_dict.items():
                tb = _tensor_bytes(tensor)
                if cur_shard and cur_bytes + tb > SHARD_SIZE:
                    shards.append(cur_shard)
                    cur_shard, cur_bytes = {}, 0
                cur_shard[key] = tensor
                cur_bytes += tb
            if cur_shard:
                shards.append(cur_shard)

            n = len(shards)
            weight_map = {}
            for idx, shard in enumerate(shards, 1):
                fname = f'model-{idx:05d}-of-{n:05d}.safetensors'
                fpath = os.path.join(save_dir, fname)
                save_file(shard, fpath)
                for key in shard:
                    weight_map[key] = fname
            print(f'[saver_susono_hf] {n} シャードで保存')

            index = {'metadata': {'total_size': total_bytes}, 'weight_map': weight_map}
            idx_path = os.path.join(save_dir, 'model.safetensors.index.json')
            with open(idx_path, 'w') as f:
                json.dump(index, f, indent=2, ensure_ascii=False)

    except ImportError:
        pt_path = os.path.join(save_dir, 'pytorch_model.bin')
        torch.save(state_dict, pt_path)
        print(f'[saver_susono_hf] pytorch_model.bin で保存 (safetensors 未インストール): {pt_path}')


def _write_config(save_dir: str, md, args) -> None:
    """config.json を生成する"""
    # テンプレート config.json の読み込み
    cfg = {}
    config_path = getattr(args, 'susono_config', None)
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
    elif hasattr(md, 'hf_config') and md.hf_config:
        cfg = dict(md.hf_config)

    # メタデータの値で上書き
    cfg.update({
        'num_hidden_layers':       md.num_layers,
        'full_attention_interval': md.full_attention_interval,
        'mhc_num_streams':         md.mhc_num_streams,
    })

    out_path = os.path.join(save_dir, 'config.json')
    with open(out_path, 'w') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f'[saver_susono_hf] config.json 書き込み完了: {out_path}')


def _copy_tokenizer(src_dir: str, dst_dir: str) -> None:
    """トークナイザー関連ファイルを出力ディレクトリにコピーする"""
    files = [
        'tokenizer.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'vocab.json',
        'merges.txt',
    ]
    for fname in files:
        src = os.path.join(src_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst_dir)
            print(f'[saver_susono_hf] コピー: {fname}')
