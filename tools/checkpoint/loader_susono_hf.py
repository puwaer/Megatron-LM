"""
Susono HuggingFace チェックポイント → キュー ローダープラグイン

Megatron の tools/checkpoint/convert.py フレームワーク用。
HuggingFace safetensors 形式から重みを読み込み、キューに送信する。

使い方:
  python convert.py \\
      --model-type GPT \\
      --loader susono_hf \\
      --saver susono_mcore \\
      --load-dir /path/to/hf_checkpoint \\
      --save-dir /path/to/output_mcore \\
      [--susono-config /path/to/config.json]
"""

import glob
import json
import os
import types

import torch


# ──────────────────────────────────────────────────────────────────────────────
# プラグインインターフェース
# ──────────────────────────────────────────────────────────────────────────────

def add_arguments(parser):
    group = parser.add_argument_group(title='Susono HF loader')
    group.add_argument('--susono-config', type=str, default=None,
                       help='config.json のパス。省略時は <load-dir>/config.json を使用')


def load_checkpoint(queue, args):
    """HF チェックポイントを読み込み、キューに送信する"""
    try:
        _load_checkpoint_impl(queue, args)
    except Exception as e:
        queue.put('exit')
        raise e


# ──────────────────────────────────────────────────────────────────────────────
# 実装
# ──────────────────────────────────────────────────────────────────────────────

def _load_checkpoint_impl(queue, args):
    load_dir = args.load_dir

    # config.json 読み込み
    config_path = getattr(args, 'susono_config', None) or os.path.join(load_dir, 'config.json')
    with open(config_path) as f:
        cfg = json.load(f)

    print(f'[loader_susono_hf] HF チェックポイント読み込み中: {load_dir}')
    hf = _load_hf_weights(load_dir)

    # アーキテクチャパラメータ
    num_layers        = cfg['num_hidden_layers']
    num_attn_heads    = cfg['num_attention_heads']
    head_dim          = cfg['head_dim']
    num_kv_heads      = cfg['num_key_value_heads']
    full_attn_interval = cfg['full_attention_interval']
    mhc_num_streams   = cfg.get('mhc_num_streams', 4)
    engram_layer_ids  = cfg.get('engram_layer_ids', [])
    hidden_size       = cfg['hidden_size']

    full_attn_layers  = {i for i in range(num_layers) if (i + 1) % full_attn_interval == 0}
    engram_by_layer   = {layer_idx: j for j, layer_idx in enumerate(engram_layer_ids)}

    # ── メタデータ送信 ──────────────────────────────────────────────────────
    md = types.SimpleNamespace(
        model_type='GPT',
        num_layers=num_layers,
        hidden_size=hidden_size,
        seq_length=cfg.get('max_position_embeddings', 4096),
        num_attention_heads=num_attn_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        full_attention_interval=full_attn_interval,
        mhc_num_streams=mhc_num_streams,
        engram_layer_ids=list(engram_layer_ids),
        num_experts=cfg.get('num_experts', 0),
        params_dtype=torch.bfloat16,
        max_position_embeddings=cfg.get('max_position_embeddings', 4096),
        tokenizer_type='GPTSentencePieceTokenizer',
        iteration=0,
        true_vocab_size=cfg.get('vocab_size', 151680),
        make_vocab_size_divisible_by=128,
        consumed_train_samples=0,
        consumed_valid_samples=0,
        stream_proj_weight=hf.get('model.stream_proj.weight'),
        tie_word_embeddings=cfg.get('tie_word_embeddings', False),
        hf_config=cfg,
    )
    queue.put(md)
    print(f'[loader_susono_hf] メタデータ送信完了 (num_layers={num_layers})')

    # ── 埋め込み ──────────────────────────────────────────────────────────
    queue.put({
        'name': 'embeddings',
        'word embeddings': hf['model.embed_tokens.weight'],
    })

    # ── 各層 ──────────────────────────────────────────────────────────────
    for i in range(num_layers):
        hf_prefix  = f'model.layers.{i}.'
        mhc_prefix = f'model.mhc_modules.{i}.'
        is_full    = i in full_attn_layers

        msg = {'name': f'transformer layer {i}'}

        # mHC モジュール
        for p in ['norm.weight', 'static_alpha', 'dynamic_alpha_fn', 'pre_branch_scale',
                  'residual_scale', 'static_beta', 'dynamic_beta_fn', 'h_post_scale']:
            k = f'{mhc_prefix}{p}'
            if k in hf:
                msg[f'mhc.{p}'] = hf[k]

        if is_full:
            # --- Full Attention 層 ---
            # SusonoRMSNorm: x*(1+w) → Megatron 標準 RMSNorm: x*w → +1.0 オフセット
            input_ln = hf[f'{hf_prefix}input_layernorm.weight'] + 1.0
            msg['input norm weight']       = input_ln
            msg['qkv layer norm weight']   = input_ln.clone()
            msg['post norm weight']        = hf[f'{hf_prefix}post_attention_layernorm.weight'] + 1.0

            # QKV インターリーブ (GQA: [kv_heads, q_per_kv+2, head_dim, hidden])
            # attention_output_gate=True の場合、q_proj は shape [heads*head_dim*2, hidden]
            # per-head layout: [q_head(head_dim), gate_head(head_dim)] が head 毎に連続
            # → Megatron の linear_qkv layout [q_heads_per_group, gate_heads_per_group, k_head, v_head]
            # に並び替える
            q_hf = hf[f'{hf_prefix}self_attn.q_proj.weight']
            k_w = hf[f'{hf_prefix}self_attn.k_proj.weight']
            v   = hf[f'{hf_prefix}self_attn.v_proj.weight']
            num_q_per_kv = num_attn_heads // num_kv_heads

            has_gate = q_hf.shape[0] == num_attn_heads * head_dim * 2
            if has_gate:
                # HF q_proj: [num_kv_heads, num_q_per_kv, 2, head_dim, hidden] (per-head q/gate interleave)
                q_and_gate = q_hf.view(num_kv_heads, num_q_per_kv, 2, head_dim, hidden_size)
                q    = q_and_gate[:, :, 0]   # [num_kv_heads, num_q_per_kv, head_dim, hidden]
                gate = q_and_gate[:, :, 1]
            else:
                q = q_hf.view(num_kv_heads, num_q_per_kv, head_dim, hidden_size)
                gate = None
            k_w = k_w.view(num_kv_heads, 1, head_dim, hidden_size)
            v   = v.view(num_kv_heads, 1, head_dim, hidden_size)
            if has_gate:
                # Megatron layout per group: [q, gate, k, v]
                msg['qkv weight'] = torch.cat([q, gate, k_w, v], dim=1).view(-1, hidden_size)
            else:
                msg['qkv weight'] = torch.cat([q, k_w, v], dim=1).view(-1, hidden_size)
            msg['dense weight'] = hf[f'{hf_prefix}self_attn.o_proj.weight']

            # QK LayerNorm (Qwen3-Next style per-head RMSNorm)
            q_norm_key = f'{hf_prefix}self_attn.q_norm.weight'
            k_norm_key = f'{hf_prefix}self_attn.k_norm.weight'
            if q_norm_key in hf:
                msg['q layernorm weight'] = hf[q_norm_key]
            if k_norm_key in hf:
                msg['k layernorm weight'] = hf[k_norm_key]

            # MLP can be either Dense (SwiGLU) or MoE (Qwen3-Next full equivalence).
            # Detect by checking whether the Dense-specific gate_proj key exists.
            if f'{hf_prefix}mlp.gate_proj.weight' in hf:
                # Dense SwiGLU MLP: merge gate_proj + up_proj → linear_fc1
                gate_m = hf[f'{hf_prefix}mlp.gate_proj.weight']
                up_m   = hf[f'{hf_prefix}mlp.up_proj.weight']
                msg['mlp l0 weight'] = torch.cat([gate_m, up_m], dim=0)
                msg['mlp l1 weight'] = hf[f'{hf_prefix}mlp.down_proj.weight']
            else:
                # MoE MLP: pass through all mlp.* keys as-is (Susono mcore uses
                # HF-compatible naming for MoE parameters).
                for hf_key, tensor in hf.items():
                    if hf_key.startswith(f'{hf_prefix}mlp.'):
                        subkey = hf_key[len(f'{hf_prefix}mlp.'):]
                        msg[f'mlp.{subkey}'] = tensor

        else:
            # --- Linear Attention 層 ---
            # 線形アテンション層の LayerNorm はオフセットなし
            msg['input norm weight'] = hf[f'{hf_prefix}input_layernorm.weight']
            msg['post norm weight']  = hf[f'{hf_prefix}post_attention_layernorm.weight']

            # GatedDeltaNet / MoE キーを全てコピー
            for hf_key, tensor in hf.items():
                if hf_key.startswith(f'{hf_prefix}linear_attn.'):
                    subkey = hf_key[len(f'{hf_prefix}linear_attn.'):]
                    msg[f'linear_attn.{subkey}'] = tensor
                elif hf_key.startswith(f'{hf_prefix}mlp.'):
                    subkey = hf_key[len(f'{hf_prefix}mlp.'):]
                    msg[f'mlp.{subkey}'] = tensor

        # Engram モジュール
        if i in engram_by_layer:
            j = engram_by_layer[i]
            engram_prefix = f'model.engram_modules.{j}.'
            for hf_key, tensor in hf.items():
                if hf_key.startswith(engram_prefix):
                    subkey = hf_key[len(engram_prefix):]
                    msg[f'engram.{subkey}'] = tensor

        queue.put(msg)

    print(f'[loader_susono_hf] 全 {num_layers} 層の送信完了')

    # ── 最終層ノルム ──────────────────────────────────────────────────────
    queue.put({
        'name': 'final layer norm',
        'weight': hf['model.norm.weight'] + 1.0,
    })

    # ── 出力層 ────────────────────────────────────────────────────────────
    if not md.tie_word_embeddings and 'lm_head.weight' in hf:
        queue.put({
            'name': 'output layer',
            'weight': hf['lm_head.weight'],
        })

    queue.put('done')
    print('[loader_susono_hf] 完了')


def _load_hf_weights(hf_dir: str) -> dict:
    """safetensors または pytorch_model.bin から重みを読み込む"""
    try:
        from safetensors.torch import load_file
        st_files = sorted(glob.glob(os.path.join(hf_dir, '*.safetensors')))
        if st_files:
            state_dict = {}
            for f in st_files:
                state_dict.update(load_file(f))
            return state_dict
    except ImportError:
        pass

    pt_path = os.path.join(hf_dir, 'pytorch_model.bin')
    if os.path.exists(pt_path):
        return torch.load(pt_path, map_location='cpu')

    raise FileNotFoundError(
        f'safetensors も pytorch_model.bin も見つかりません: {hf_dir}'
    )
