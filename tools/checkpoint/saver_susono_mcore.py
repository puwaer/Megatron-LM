"""
キュー → Susono Megatron mcore distcp チェックポイント セーバープラグイン

Megatron の tools/checkpoint/convert.py フレームワーク用。
キューから受け取った重みを Megatron mcore 分散チェックポイント形式で保存する。

【旧 hf_to_susono.py との主な違い】
  - mp.spawn + DTensor + dist.init_process_group("gloo") を廃止
  - dcp_save(no_dist=True) による単一プロセス保存を採用
  - TP=1/PP=1/EP=1 の flat チェックポイントとして保存
  - Megatron は任意の TP/PP/EP 設定でロード時に再分割できる

使い方:
  python convert.py \\
      --model-type GPT \\
      --loader susono_hf \\
      --saver susono_mcore \\
      --load-dir /path/to/hf_checkpoint \\
      --save-dir /path/to/output_mcore_base \\
      [--susono-tp 1] [--susono-pp 1] [--susono-ep 1] [--susono-iteration 0]
"""

import io
import json
import os
import types

import torch


# ──────────────────────────────────────────────────────────────────────────────
# プラグインインターフェース
# ──────────────────────────────────────────────────────────────────────────────

def add_arguments(parser):
    group = parser.add_argument_group(title='Susono mcore saver')
    group.add_argument('--susono-tp', type=int, default=1,
                       help='テンソル並列サイズ (メタデータのみ記録。保存は常に TP=1)')
    group.add_argument('--susono-pp', type=int, default=1,
                       help='パイプライン並列サイズ (メタデータのみ記録)')
    group.add_argument('--susono-ep', type=int, default=1,
                       help='エキスパート並列サイズ (メタデータのみ記録)')
    group.add_argument('--susono-iteration', type=int, default=0,
                       help='チェックポイントのイテレーション番号')


def save_checkpoint(queue, args):
    """キューから重みを受け取り、mcore distcp 形式で保存する"""
    _save_checkpoint_impl(queue, args)


# ──────────────────────────────────────────────────────────────────────────────
# 実装
# ──────────────────────────────────────────────────────────────────────────────

def _save_checkpoint_impl(queue, args):
    # ── メタデータ受信 ────────────────────────────────────────────────────
    md = queue.get()
    if md == 'exit':
        print('[saver_susono_mcore] ローダーがエラーで終了しました。')
        return

    num_layers         = md.num_layers
    full_attn_interval = md.full_attention_interval
    engram_layer_ids   = md.engram_layer_ids
    full_attn_layers   = {i for i in range(num_layers) if (i + 1) % full_attn_interval == 0}

    base_dir  = args.save_dir
    iteration = getattr(args, 'susono_iteration', 0)
    iter_dir  = os.path.join(base_dir, f'iter_{iteration:07d}')
    os.makedirs(iter_dir, exist_ok=True)
    print(f'[saver_susono_mcore] 保存先: {iter_dir}')

    state_dict = {}

    # stream_proj はメタデータに埋め込まれている
    if getattr(md, 'stream_proj_weight', None) is not None:
        state_dict['decoder.stream_proj_pp0.weight'] = md.stream_proj_weight

    def _check(msg, expected_name):
        if getattr(args, 'checking', True) and isinstance(msg, dict):
            actual = msg.get('name', '')
            if actual != expected_name:
                raise ValueError(f'キュープロトコルエラー: "{expected_name}" を期待しましたが "{actual}" を受信しました。')

    # ── 埋め込み ──────────────────────────────────────────────────────────
    msg = queue.get()
    _check(msg, 'embeddings')
    state_dict['embedding.word_embeddings.weight'] = msg['word embeddings']

    # ── 各層 ──────────────────────────────────────────────────────────────
    for i in range(num_layers):
        msg = queue.get()
        _check(msg, f'transformer layer {i}')
        pfx = f'decoder.layers.{i}'

        # MHC-Lite パラメータ
        for p in ['norm.weight', 'static_alpha', 'dynamic_alpha_fn', 'pre_branch_scale',
                  'residual_scale', 'static_beta', 'dynamic_beta_fn', 'h_post_scale']:
            key = f'mhc.{p}'
            if key in msg:
                state_dict[f'{pfx}.mhc.{p}'] = msg[key]

        if i in full_attn_layers:
            # Full Attention 層
            state_dict[f'{pfx}.input_layernorm.weight']                           = msg['input norm weight']
            state_dict[f'{pfx}.self_attention.linear_qkv.weight']                 = msg['qkv weight']
            state_dict[f'{pfx}.self_attention.linear_qkv.layer_norm_weight']      = msg['qkv layer norm weight']
            state_dict[f'{pfx}.self_attention.linear_proj.weight']                = msg['dense weight']
            state_dict[f'{pfx}.pre_mlp_layernorm.weight']                         = msg['post norm weight']
            state_dict[f'{pfx}.mlp.linear_fc1.weight']                            = msg['mlp l0 weight']
            state_dict[f'{pfx}.mlp.linear_fc2.weight']                            = msg['mlp l1 weight']
        else:
            # Linear Attention 層
            state_dict[f'{pfx}.input_layernorm.weight']          = msg['input norm weight']
            state_dict[f'{pfx}.post_attention_layernorm.weight'] = msg['post norm weight']

            # GatedDeltaNet / MoE キーを全転送
            for key, val in msg.items():
                if key.startswith('linear_attn.') or key.startswith('mlp.'):
                    state_dict[f'{pfx}.{key}'] = val

        # Engram パラメータ
        for key, val in msg.items():
            if key.startswith('engram.'):
                state_dict[f'{pfx}.{key}'] = val

    # ── 最終層ノルム ──────────────────────────────────────────────────────
    msg = queue.get()
    _check(msg, 'final layer norm')
    state_dict['decoder.final_layernorm.weight'] = msg['weight']

    # ── 出力層 (存在すれば) ───────────────────────────────────────────────
    msg = queue.get()
    if msg != 'done' and isinstance(msg, dict) and msg.get('name') == 'output layer':
        state_dict['output_layer.weight'] = msg['weight']
        msg = queue.get()

    if msg != 'done':
        raise RuntimeError(f'キュープロトコルエラー: "done" を期待しましたが {msg!r} を受信しました。')

    # ── Transformer Engine _extra_state (TE FP8 初期化状態) ──────────────────
    # distcp は ShardedObject.unique_key = {key}/shard_{offset}_{shape} 形式で管理する。
    # TE 未初期化状態は get_extra_state() == None → [None] を bytes にシリアライズして保存。
    # global_offset=(0,), global_shape=(1,) → suffix は /shard_0_1 (PP=1 単一シャード)。
    buf = io.BytesIO()
    torch.save([None], buf)
    _te_es = buf.getvalue()

    for i in full_attn_layers:
        pfx = f'decoder.layers.{i}'
        for sub in ['input_layernorm', 'self_attention.linear_qkv',
                    'self_attention.linear_proj', 'pre_mlp_layernorm',
                    'mlp.linear_fc1', 'mlp.linear_fc2']:
            state_dict[f'{pfx}.{sub}._extra_state/shard_0_1'] = io.BytesIO(_te_es)

    state_dict['decoder.final_layernorm._extra_state/shard_0_1'] = io.BytesIO(_te_es)

    print(f'[saver_susono_mcore] state_dict キー数: {len(state_dict)}')

    # ── distcp 保存 ───────────────────────────────────────────────────────
    _save_distcp(state_dict, iter_dir)

    # ── サイドカーファイル作成 ─────────────────────────────────────────────
    _write_sidecars(base_dir, iter_dir, md, args)

    print(f'[saver_susono_mcore] 保存完了: {iter_dir}')


def _save_distcp(state_dict: dict, save_dir: str) -> None:
    """
    単一プロセスで distcp 形式に保存する。
    no_dist=True を使用することで mp.spawn / dist.init_process_group が不要。
    """
    from torch.distributed.checkpoint import FileSystemWriter

    print(f'[saver_susono_mcore] distcp 保存中 (no_dist=True): {save_dir}')

    try:
        # PyTorch >= 2.1
        from torch.distributed.checkpoint.state_dict_saver import save as dcp_save
        dcp_save(state_dict, storage_writer=FileSystemWriter(save_dir), no_dist=True)
    except TypeError:
        # no_dist 引数が未サポートの旧バージョン向けフォールバック
        import torch.distributed.checkpoint as dcp
        dcp.save(state_dict, checkpoint_id=save_dir)


def _write_sidecars(base_dir: str, iter_dir: str, md, args) -> None:
    """Megatron-LM が読み込むためのメタデータファイルを作成する"""
    tp        = getattr(args, 'susono_tp', 1)
    pp        = getattr(args, 'susono_pp', 1)
    ep        = getattr(args, 'susono_ep', 1)
    iteration = getattr(args, 'susono_iteration', 0)

    # metadata.json (iter ディレクトリ内)
    metadata = {
        'sharded_backend': 'torch_dist',
        'sharded_backend_version': 1,
        'common_backend': 'torch',
        'common_backend_version': 1,
    }
    with open(os.path.join(iter_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)

    # common.pt (iter ディレクトリ内)
    ckpt_args = types.SimpleNamespace(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        expert_model_parallel_size=ep,
    )
    common_state = {
        'args': ckpt_args,
        'checkpoint_version': 3.0,
        'iteration': iteration,
        'num_floating_point_operations_so_far': 0,
    }
    torch.save(common_state, os.path.join(iter_dir, 'common.pt'))

    # latest_checkpointed_iteration.txt (ベースディレクトリ直下)
    with open(os.path.join(base_dir, 'latest_checkpointed_iteration.txt'), 'w') as f:
        f.write(str(iteration))

    print(f'[saver_susono_mcore] サイドカーファイル作成完了 (TP={tp}, PP={pp}, EP={ep}, iter={iteration})')
