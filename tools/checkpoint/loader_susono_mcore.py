"""
Susono Megatron mcore distcp チェックポイント → キュー ローダープラグイン

Megatron の tools/checkpoint/convert.py フレームワーク用。
Megatron mcore 分散チェックポイント (distcp) 形式から重みを読み込み、
キューに送信する。既存の susono_convert.py のロードロジックを流用。

使い方:
  python convert.py \\
      --model-type GPT \\
      --loader susono_mcore \\
      --saver susono_hf \\
      --load-dir /path/to/iter_NNNNNNN \\
      --save-dir /path/to/output_hf \\
      --susono-num-layers 24 \\
      --susono-full-attn-interval 4 \\
      --susono-mhc-num-streams 4 \\
      [--susono-config /path/to/config.json]
"""

import json
import os
import re
import types
import warnings

import torch


# ──────────────────────────────────────────────────────────────────────────────
# プラグインインターフェース
# ──────────────────────────────────────────────────────────────────────────────

def add_arguments(parser):
    group = parser.add_argument_group(title='Susono mcore loader')
    group.add_argument('--susono-num-layers', type=int, default=None,
                       help='トランスフォーマー層数 (省略時はチェックポイントから自動推定)')
    group.add_argument('--susono-full-attn-interval', type=int, default=4,
                       help='Full-Attention 間隔 (デフォルト: 4)')
    group.add_argument('--susono-mhc-num-streams', type=int, default=4,
                       help='mHC ストリーム数 (デフォルト: 4)')
    group.add_argument('--susono-num-attn-heads', type=int, default=8,
                       help='Full-Attention ヘッド数 (デフォルト: 8)')
    group.add_argument('--susono-num-kv-heads', type=int, default=2,
                       help='KV ヘッド数 GQA (デフォルト: 2)')
    group.add_argument('--susono-head-dim', type=int, default=256,
                       help='アテンションヘッド次元 (デフォルト: 256)')
    group.add_argument('--susono-config', type=str, default=None,
                       help='config.json のパス (パラメータを上書きするために使用)')


def load_checkpoint(queue, args):
    """mcore distcp チェックポイントを読み込み、キューに送信する"""
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

    # config.json (省略可)
    cfg = {}
    config_path = getattr(args, 'susono_config', None)
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)

    print(f'[loader_susono_mcore] distcp 読み込み中: {load_dir}')
    megatron = _load_distcp(load_dir)

    # アーキテクチャパラメータの解決 (引数 > config.json > 自動推定)
    num_layers         = getattr(args, 'susono_num_layers', None)     or cfg.get('num_hidden_layers')     or _infer_num_layers(megatron)
    full_attn_interval = getattr(args, 'susono_full_attn_interval', 4) or cfg.get('full_attention_interval', 4)
    mhc_num_streams    = getattr(args, 'susono_mhc_num_streams', 4)   or cfg.get('mhc_num_streams', 4)
    num_attn_heads     = getattr(args, 'susono_num_attn_heads', 8)     or cfg.get('num_attention_heads', 8)
    num_kv_heads       = getattr(args, 'susono_num_kv_heads', 2)       or cfg.get('num_key_value_heads', 2)
    head_dim           = getattr(args, 'susono_head_dim', 256)         or cfg.get('head_dim', 256)
    hidden_size        = cfg.get('hidden_size') or megatron['embedding.word_embeddings.weight'].shape[1]
    engram_layer_ids   = cfg.get('engram_layer_ids') or _detect_engram_layers(megatron)

    full_attn_layers   = {i for i in range(num_layers) if (i + 1) % full_attn_interval == 0}

    print(f'[loader_susono_mcore] num_layers={num_layers}, full_attn_interval={full_attn_interval}, '
          f'engram_layer_ids={engram_layer_ids}')

    # stream_proj キーを検索 (PP 分割の場合は pp 番号付き)
    sp_matches = [
        (int(m.group(1)), k)
        for k in megatron
        for m in [re.match(r'^decoder\.stream_proj_pp(\d+)\.weight$', k)]
        if m
    ]
    if sp_matches:
        _, sp_key = max(sp_matches, key=lambda x: x[0])
        stream_proj = megatron[sp_key]
        # 全 PP キーを消費済みとしてマーク (後の未使用キーチェックのため)
    else:
        stream_proj = megatron.get('decoder.stream_proj.weight')

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
        iteration=_detect_iteration(load_dir),
        true_vocab_size=cfg.get('vocab_size') or megatron['embedding.word_embeddings.weight'].shape[0],
        make_vocab_size_divisible_by=128,
        consumed_train_samples=0,
        consumed_valid_samples=0,
        stream_proj_weight=stream_proj,
        tie_word_embeddings=cfg.get('tie_word_embeddings', False),
        hf_config=cfg,
    )
    queue.put(md)

    # ── 埋め込み ──────────────────────────────────────────────────────────
    queue.put({
        'name': 'embeddings',
        'word embeddings': megatron['embedding.word_embeddings.weight'],
    })

    # ── 各層 ──────────────────────────────────────────────────────────────
    for i in range(num_layers):
        src    = f'decoder.layers.{i}'
        is_full = i in full_attn_layers
        msg    = {'name': f'transformer layer {i}'}

        # mHC パラメータ
        for p in ['norm.weight', 'static_alpha', 'dynamic_alpha_fn', 'pre_branch_scale',
                  'residual_scale', 'static_beta', 'dynamic_beta_fn', 'h_post_scale']:
            k = f'{src}.mhc.{p}'
            if k in megatron:
                msg[f'mhc.{p}'] = megatron[k]

        if is_full:
            msg['input norm weight']      = megatron[f'{src}.input_layernorm.weight']
            msg['qkv weight']             = megatron[f'{src}.self_attention.linear_qkv.weight']
            msg['qkv layer norm weight']  = megatron.get(
                f'{src}.self_attention.linear_qkv.layer_norm_weight',
                megatron[f'{src}.input_layernorm.weight'],  # フォールバック
            )
            msg['dense weight']   = megatron[f'{src}.self_attention.linear_proj.weight']
            msg['post norm weight'] = megatron[f'{src}.pre_mlp_layernorm.weight']
            msg['mlp l0 weight']  = megatron[f'{src}.mlp.linear_fc1.weight']
            msg['mlp l1 weight']  = megatron[f'{src}.mlp.linear_fc2.weight']
        else:
            # B-6: GDN では TELayerNormColumnParallelLinear により
            #   input_layernorm.weight + in_proj_qkvz.weight + in_proj_ba.weight
            # が以下の 2 キーに統合されている:
            #   linear_attn.input_ln_proj.layer_norm_weight      [D]
            #   linear_attn.input_ln_proj.weight                 [proj_qkvz + proj_ba, D]
            # 下流 (saver_susono_hf など) は旧レイアウトを期待するため、ここで
            # 逆変換 (split + rename) して msg に入れる。
            fused_ln = f'{src}.linear_attn.input_ln_proj.layer_norm_weight'
            fused_w = f'{src}.linear_attn.input_ln_proj.weight'
            if fused_ln in megatron and fused_w in megatron:
                # B-6 fused format
                msg['input norm weight'] = megatron[fused_ln]
                w = megatron[fused_w]
                # num_v_heads * 2 = proj_ba; 残りが proj_qkvz
                num_v_heads = int(cfg.get('linear_num_value_heads', 32))
                proj_ba = num_v_heads * 2
                proj_qkvz = w.shape[0] - proj_ba
                msg['linear_attn.in_proj_qkvz.weight'] = w[:proj_qkvz].contiguous()
                msg['linear_attn.in_proj_ba.weight'] = w[proj_qkvz:].contiguous()
                fused_skip_keys = {fused_ln, fused_w}
            else:
                # 旧 (B-6 以前) レイアウト
                msg['input norm weight'] = megatron[f'{src}.input_layernorm.weight']
                fused_skip_keys = set()

            msg['post norm weight'] = megatron[f'{src}.post_attention_layernorm.weight']

            # Susono は HF-compatible な key 名を mcore 側でもそのまま使用しているため、
            # generic pass-through で OK (リネーム不要)。
            for meg_key, val in megatron.items():
                if meg_key in fused_skip_keys:
                    continue
                if meg_key.startswith(f'{src}.linear_attn.'):
                    subkey = meg_key[len(f'{src}.linear_attn.'):]
                    msg[f'linear_attn.{subkey}'] = val
                elif meg_key.startswith(f'{src}.mlp.'):
                    subkey = meg_key[len(f'{src}.mlp.'):]
                    msg[f'mlp.{subkey}'] = val

        # Engram パラメータ
        for meg_key, val in megatron.items():
            if meg_key.startswith(f'{src}.engram.'):
                subkey = meg_key[len(f'{src}.engram.'):]
                msg[f'engram.{subkey}'] = val

        queue.put(msg)

    print(f'[loader_susono_mcore] 全 {num_layers} 層の送信完了')

    # ── 最終層ノルム ──────────────────────────────────────────────────────
    if 'decoder.final_layernorm.weight' in megatron:
        queue.put({
            'name': 'final layer norm',
            'weight': megatron['decoder.final_layernorm.weight'],
        })

    # ── 出力層 ────────────────────────────────────────────────────────────
    if 'output_layer.weight' in megatron and not md.tie_word_embeddings:
        queue.put({
            'name': 'output layer',
            'weight': megatron['output_layer.weight'],
        })

    queue.put('done')
    print('[loader_susono_mcore] 完了')


# ──────────────────────────────────────────────────────────────────────────────
# ヘルパー関数
# ──────────────────────────────────────────────────────────────────────────────

def _load_distcp(checkpoint_dir: str) -> dict:
    """distcp 形式のチェックポイントを単一プロセスで読み込む (no_dist=True)"""
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    reader   = FileSystemReader(checkpoint_dir)
    metadata = reader.read_metadata()

    model_keys = [
        k for k, m in metadata.state_dict_metadata.items()
        if not k.startswith('optimizer')
        and not k.startswith('rng')
        and '_extra_state' not in k
        and isinstance(m, TensorStorageMetadata)
    ]

    state_dict = {}
    for key in model_keys:
        meta = metadata.state_dict_metadata[key]
        state_dict[key] = torch.zeros(list(meta.size), dtype=meta.properties.dtype)

    from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
    dcp_load(state_dict=state_dict, storage_reader=reader, no_dist=True)

    # model.module. プレフィックスを除去
    prefix = 'model.module.'
    if any(k.startswith(prefix) for k in state_dict):
        state_dict = {
            (k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in state_dict.items()
        }

    # FP8 → BF16 キャスト
    fp8_keys = [
        k for k, v in state_dict.items()
        if hasattr(v, 'dtype') and 'float8' in str(v.dtype).lower()
    ]
    if fp8_keys:
        warnings.warn(
            f'[WARNING] {len(fp8_keys)} 個の FP8 パラメータを BF16 にキャストします。'
            '精度が低下する場合があります。',
            stacklevel=2,
        )
        for k in fp8_keys:
            state_dict[k] = state_dict[k].to(torch.bfloat16)

    return state_dict


def _infer_num_layers(megatron: dict) -> int:
    """state_dict のキーから層数を推定する"""
    indices = set()
    for k in megatron:
        m = re.match(r'decoder\.layers\.(\d+)\.', k)
        if m:
            indices.add(int(m.group(1)))
    return max(indices) + 1 if indices else 0


def _detect_engram_layers(megatron: dict) -> list:
    """engram キーを持つ層のインデックスを検出する"""
    indices = set()
    for k in megatron:
        m = re.search(r'decoder\.layers\.(\d+)\.engram', k)
        if m:
            indices.add(int(m.group(1)))
    return sorted(indices)


def _detect_iteration(load_dir: str) -> int:
    """ディレクトリ名または latest_checkpointed_iteration.txt からイテレーション番号を取得する"""
    basename = os.path.basename(os.path.normpath(load_dir))
    m = re.match(r'iter_(\d+)$', basename)
    if m:
        return int(m.group(1))
    iter_file = os.path.join(os.path.dirname(load_dir), 'latest_checkpointed_iteration.txt')
    if os.path.exists(iter_file):
        with open(iter_file) as f:
            return int(f.read().strip())
    return 0
