# /// script
# requires-python = ">=3.12"
# ///
"""VLA-JEPA Qwen context export/probe.

This lane keeps tokenizer/chat-template/image preprocessing and the Qwen3-VL
vision tower on the host. The exported graph receives the already-conditioned
Qwen language-model inputs and returns:

  embodied_action_tokens [B, 32, 2048]

That is the exact conditioning contract the verified `action_denoise_step`
graph consumes.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch


QWEN_HIDDEN = 2048
OUT_NAME = "qwen_context.pt2"
CONTRACT_NAME = "vla-jepa-qwen-contract.json"


def _load_config(config_json: Path):
    from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig

    cfg = VLAJEPAConfig()
    raw = json.loads(config_json.read_text())
    valid = {f.name for f in dataclasses.fields(VLAJEPAConfig)}
    for key, value in raw.items():
        if key in valid and not isinstance(value, (dict, list)):
            setattr(cfg, key, value)
    return cfg, raw


def _image_keys(raw: dict) -> list[str]:
    feats = raw.get("input_features") or {}
    return [k for k, v in feats.items() if v.get("type") == "VISUAL"]


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    return torch.bfloat16


def _load_qwen_local(cfg):
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    src = snapshot_download(cfg.qwen_model_name, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        src,
        torch_dtype=_dtype(cfg.torch_dtype),
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(src, local_files_only=True)
    processor.tokenizer.padding_side = cfg.tokenizer_padding_side
    model.config.hidden_size = model.config.text_config.hidden_size
    return model.eval(), processor


def _expand_tokenizer(cfg, model, processor):
    max_action_tokens = cfg.chunk_size * 4
    tokenizer = processor.tokenizer
    action_tokens = []
    for idx in range(max_action_tokens):
        token = cfg.special_action_token.format(idx)
        action_tokens.append(token)
        if token not in tokenizer.get_vocab():
            tokenizer.add_tokens([token], special_tokens=True)

    if cfg.embodied_action_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([cfg.embodied_action_token], special_tokens=True)
    embodied_action_token_id = tokenizer.convert_tokens_to_ids(cfg.embodied_action_token)
    if model.get_input_embeddings().weight.size(0) < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    return action_tokens, embodied_action_token_id


def _build_prompt(cfg, action_tokens):
    num_action_prompt_steps = cfg.num_video_frames // cfg.jepa_tubelet_size - 1
    replace_prompt = "".join(
        token * cfg.num_action_tokens_per_timestep for token in action_tokens[:num_action_prompt_steps]
    )
    embodied_prompt = cfg.embodied_action_token * cfg.num_embodied_action_tokens_per_instruction
    return replace_prompt, embodied_prompt


def _build_inputs_local(cfg, processor, images, instruction: str, action_prompt: str, embodied_prompt: str):
    prompt = cfg.prompt_template.format(
        instruction=instruction,
        actions=action_prompt,
        e_actions=embodied_prompt,
    )
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    messages = [[{"role": "user", "content": content}]]
    batch_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        processor_kwargs={
            "padding": True,
            "return_tensors": "pt",
            "do_rescale": False,
        },
    )
    return batch_inputs


class QwenContextWrapper(torch.nn.Module):
    def __init__(self, qwen_model):
        super().__init__()
        self.qwen_model = qwen_model

    def forward(
        self,
        inputs_embeds,
        attention_mask,
        position_ids,
        embodied_positions,
    ):
        captured = []

        def _hook(_module, _inputs, output):
            captured.append(output[0] if isinstance(output, tuple) else output)

        last_layer = self.qwen_model.model.language_model.layers[-1]
        handle = last_layer.register_forward_hook(_hook)
        try:
            self.qwen_model.model.language_model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=None,
                visual_pos_masks=None,
                deepstack_visual_embeds=None,
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True,
            )
        finally:
            handle.remove()

        last_hidden = captured[0]
        gather_index = embodied_positions.unsqueeze(-1).expand(-1, -1, last_hidden.shape[-1])
        return torch.gather(last_hidden, 1, gather_index)


def _build_language_inputs(model, sample):
    with torch.no_grad():
        inputs_embeds = model.model.get_input_embeddings()(sample["input_ids"])
        image_outputs = model.model.get_image_features(
            sample["pixel_values"],
            sample["image_grid_thw"],
            return_dict=True,
        )
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
            inputs_embeds.device,
            inputs_embeds.dtype,
        )
        image_mask, _ = model.model.get_placeholder_mask(
            sample["input_ids"],
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        position_ids = model.model.compute_3d_position_ids(
            input_ids=sample["input_ids"],
            image_grid_thw=sample["image_grid_thw"],
            video_grid_thw=None,
            inputs_embeds=inputs_embeds,
            attention_mask=sample["attention_mask"],
            past_key_values=None,
            mm_token_type_ids=sample["mm_token_type_ids"],
        )
    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": sample["attention_mask"],
        "position_ids": position_ids,
        "embodied_positions": sample["embodied_positions"],
    }


def _build_sample(config_json: Path, instruction: str):
    cfg, raw = _load_config(config_json)
    qwen_model, processor = _load_qwen_local(cfg)
    action_tokens, embodied_action_token_id = _expand_tokenizer(cfg, qwen_model, processor)
    action_prompt, embodied_prompt = _build_prompt(cfg, action_tokens)
    views = len(_image_keys(raw))
    img_size = cfg.resize_images_to[0] if cfg.resize_images_to is not None else 224
    images = [torch.zeros(3, img_size, img_size, dtype=torch.float32) for _ in range(views)]
    qwen_inputs = _build_inputs_local(cfg, processor, images, instruction, action_prompt, embodied_prompt)
    input_ids = torch.as_tensor(qwen_inputs["input_ids"], dtype=torch.long)
    attention_mask = torch.as_tensor(qwen_inputs["attention_mask"], dtype=torch.long)
    mm_token_type_ids = torch.as_tensor(qwen_inputs["mm_token_type_ids"], dtype=torch.long)
    embodied = (input_ids == embodied_action_token_id).nonzero(as_tuple=False)[:, 1].view(
        input_ids.shape[0], -1
    )
    sample = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mm_token_type_ids": mm_token_type_ids,
        "pixel_values": qwen_inputs["pixel_values"],
        "image_grid_thw": qwen_inputs["image_grid_thw"],
        "embodied_positions": embodied.to(torch.long),
    }
    return qwen_model, sample, cfg, raw


def cmd_probe(args) -> None:
    model, sample, cfg, raw = _build_sample(args.config_json, args.instruction)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    info = {
        "image_keys": _image_keys(raw),
        "num_views": len(_image_keys(raw)),
        "input_ids_shape": list(sample["input_ids"].shape),
        "attention_mask_shape": list(sample["attention_mask"].shape),
        "mm_token_type_ids_shape": list(sample["mm_token_type_ids"].shape),
        "pixel_values_shape": list(sample["pixel_values"].shape),
        "image_grid_thw_shape": list(sample["image_grid_thw"].shape),
        "embodied_positions_shape": list(sample["embodied_positions"].shape),
        "embodied_positions": sample["embodied_positions"].tolist(),
        "num_embodied_tokens": int(cfg.num_embodied_action_tokens_per_instruction),
        "hidden_size": int(model.config.hidden_size),
    }
    (out / "vla-jepa-qwen-probe.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    print(f"ok: wrote {out/'vla-jepa-qwen-probe.json'}")


def cmd_export(args) -> None:
    model, sample, cfg, raw = _build_sample(args.config_json, args.instruction)
    language_inputs = _build_language_inputs(model, sample)
    wrapper = QwenContextWrapper(model).eval()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        ep = torch.export.export(
            wrapper,
            args=(
                language_inputs["inputs_embeds"],
                language_inputs["attention_mask"],
                language_inputs["position_ids"],
                language_inputs["embodied_positions"],
            ),
            strict=False,
        )
    torch.export.save(ep, str(out / OUT_NAME))
    contract = {
        "entrypoint": "qwen_context",
        "host_components": [
            "tokenizer",
            "chat_template",
            "vision_tower",
            "image_placeholder_scatter",
            "position_ids",
        ],
        "image_keys": _image_keys(raw),
        "num_views": len(_image_keys(raw)),
        "hidden_size": int(model.config.hidden_size),
        "num_embodied_tokens": int(cfg.num_embodied_action_tokens_per_instruction),
        "inputs_embeds_shape": list(language_inputs["inputs_embeds"].shape),
        "attention_mask_shape": list(sample["attention_mask"].shape),
        "position_ids_shape": list(language_inputs["position_ids"].shape),
        "embodied_positions_shape": list(sample["embodied_positions"].shape),
    }
    (out / CONTRACT_NAME).write_text(json.dumps(contract, indent=2) + "\n")
    print(f"ok: wrote {out/OUT_NAME}")
    print(f"ok: wrote {out/CONTRACT_NAME}")


def cmd_lower(args) -> None:
    import torch
    from coreai_torch import TorchConverter, get_decomp_table

    out = args.out
    ep = torch.export.load(str(out / OUT_NAME))
    ep = ep.run_decompositions(get_decomp_table())
    conv = TorchConverter()
    conv.add_exported_program(
        ep,
        input_names=["inputs_embeds", "attention_mask", "position_ids", "embodied_positions"],
        output_names=["embodied_action_tokens"],
        entrypoint_name="qwen_context",
    )
    prog = conv.to_coreai()
    prog.optimize()
    aimodel = out / "qwen_context.aimodel"
    prog.save_asset(aimodel)
    print(f"ok: lowered qwen_context -> {aimodel}")


def main() -> None:
    ap = argparse.ArgumentParser(description="VLA-JEPA qwen_context export/probe")
    ap.add_argument("phase", nargs="?", default="export", choices=["probe", "export"])
    ap.add_argument("--lower", action="store_true")
    ap.add_argument("--config-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--instruction", default="Pick and place the object.")
    args = ap.parse_args()
    if args.lower:
        cmd_lower(args)
    elif args.phase == "probe":
        cmd_probe(args)
    else:
        cmd_export(args)


if __name__ == "__main__":
    main()
