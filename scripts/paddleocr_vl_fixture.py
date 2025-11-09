#!/usr/bin/env python3

"""Capture PaddleOCR-VL reference tensors for parity tests.

This script loads the official PaddleOCR-VL checkpoints via the Transformers
implementation, runs a single prompt + image pair through the SigLIP vision
stack and Ernie decoder, and saves the intermediate tensors we need to validate
the Candle port. The resulting `.npz` file contains:

* tokenizer outputs (`input_ids`, `attention_mask`)
* vision metadata (`image_grid_thw`, `siglip_position_ids`)
* SigLIP encoder outputs (pre- and post-projector)
* fused prompt embeddings after image injection
* decoder hidden states plus the final-token logits
* rotary deltas computed by the Python reference

Metadata about the prompt/image/model paths is written alongside the tensor
archive so we can regenerate the fixture deterministically in the future.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


DEFAULT_PROMPT = (
    "User: <image> Summarize the key fields in this document.\nAssistant:"
)
DEFAULT_IMAGE = "baselines/fixtures/paddleocr_vl/fixture_image.png"


def _ensure_prompt(prompt: str | None, prompt_file: str | None) -> str:
    if prompt_file:
        return Path(prompt_file).read_text().strip()
    if prompt:
        return prompt.strip()
    return DEFAULT_PROMPT


def _placeholder_prompt(raw: str) -> str:
    return raw.replace(
        "<image>", "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>"
    )


def _prepare_image(path: Path) -> Image.Image:
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _grid_list(grid: torch.Tensor) -> List[Tuple[int, int, int]]:
    return [tuple(int(v) for v in row.tolist()) for row in grid]


def _stack_projector_outputs(
    embeds: Union[torch.Tensor, Sequence[torch.Tensor]]
) -> torch.Tensor:
    if isinstance(embeds, torch.Tensor):
        return embeds
    if len(embeds) == 1:
        return embeds[0]
    return torch.cat(list(embeds), dim=0)


def capture_fixture(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = getattr(torch, args.torch_dtype)

    processor = AutoProcessor.from_pretrained(
        args.model_dir, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, torch_dtype=dtype
    )
    model.eval().to(device)

    prompt = _ensure_prompt(args.prompt, args.prompt_file)
    prompt_with_tokens = _placeholder_prompt(prompt)
    image_paths = args.images or [DEFAULT_IMAGE]
    loaded_images = [_prepare_image(Path(path)) for path in image_paths]
    slot_count = prompt.count("<image>")
    if slot_count != len(loaded_images):
        raise RuntimeError(
            f"prompt contains {slot_count} <image> slots but {len(loaded_images)} images were provided"
        )
    image_input: Union[Image.Image, List[Image.Image]]
    if len(loaded_images) == 1:
        image_input = loaded_images[0]
    else:
        image_input = loaded_images

    processor_kwargs = {}
    if args.max_pixels is not None:
        processor_kwargs["max_pixels"] = args.max_pixels
    if args.min_pixels is not None:
        processor_kwargs["min_pixels"] = args.min_pixels
    batch = processor(
        images=image_input,
        text=prompt_with_tokens,
        return_tensors="pt",
        **processor_kwargs,
    )
    if "pixel_values" not in batch:
        raise RuntimeError("processor did not return pixel values for the image")

    model_inputs = {
        key: (value.to(device) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]
    image_grid_thw = model_inputs.get("image_grid_thw")
    if image_grid_thw is None or len(image_grid_thw) == 0:
        raise RuntimeError("processor did not provide image_grid_thw metadata")
    if not isinstance(image_grid_thw, torch.Tensor):
        image_grid_thw = torch.tensor(image_grid_thw, dtype=torch.long, device=device)
    model_inputs["image_grid_thw"] = image_grid_thw
    grid_list = _grid_list(image_grid_thw.cpu())

    pixel_values_for_encoder = pixel_values
    if pixel_values_for_encoder.ndim == 4:
        pixel_values_for_encoder = pixel_values_for_encoder.unsqueeze(0)
    elif pixel_values_for_encoder.ndim != 5:
        raise RuntimeError(
            f"expected pixel_values with rank 4 or 5, got {pixel_values.ndim}"
        )
    pixel_values_for_encoder = pixel_values_for_encoder.to(model.visual.dtype)

    siglip_position_ids: List[torch.Tensor] = []
    sample_indices: List[torch.Tensor] = []
    cu_seqlens = [0]
    for idx, thw in enumerate(grid_list):
        numel = int(np.prod(thw))
        spatial = int(np.prod(thw[1:]))
        image_position_ids = (
            torch.arange(numel, device=device, dtype=torch.long) % spatial
        )
        siglip_position_ids.append(image_position_ids)
        sample_indices.append(
            torch.full((numel,), idx, dtype=torch.long, device=device)
        )
        cu_seqlens.append(cu_seqlens[-1] + numel)

    siglip_position_ids = torch.cat(siglip_position_ids, dim=0)
    sample_indices = torch.cat(sample_indices, dim=0)
    cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    with torch.no_grad():
        token_embeds = model.model.embed_tokens(input_ids)
        vision_outputs = model.visual(
            pixel_values=pixel_values_for_encoder,
            output_hidden_states=args.dump_vision_hidden,
            image_grid_thw=grid_list,
            position_ids=siglip_position_ids,
            vision_return_embed_list=True,
            interpolate_pos_encoding=True,
            sample_indices=sample_indices,
            cu_seqlens=cu_seqlens_tensor,
            return_pooler_output=False,
            use_rope=True,
            window_size=-1,
        )
        siglip_hidden_raw = vision_outputs.last_hidden_state
        projected_embeds = model.mlp_AR(siglip_hidden_raw, grid_list)
        projector_concat = _stack_projector_outputs(projected_embeds)
        siglip_hidden_flat = _stack_projector_outputs(siglip_hidden_raw)

        image_token_mask = (input_ids == model.config.image_token_id).unsqueeze(-1)
        if projector_concat.shape[0] != int(image_token_mask.sum()):
            raise RuntimeError(
                "image embeddings do not align with placeholder token count"
            )
        fused_embeddings = token_embeds.masked_scatter(
            image_token_mask.expand_as(token_embeds),
            projector_concat.to(token_embeds.dtype),
        )

        position_ids, rope_deltas = model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )

        decoder_outputs = model.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            inputs_embeds=fused_embeddings,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        decoder_hidden = decoder_outputs.last_hidden_state
        logits = model.lm_head(decoder_hidden)
        next_token_logits = logits[:, -1, :]

    pixel_values_dump = None
    siglip_hidden_states = None
    if args.dump_vision_hidden:
        hidden_states = vision_outputs.hidden_states
        if hidden_states is None or len(hidden_states) == 0:
            raise RuntimeError("vision model did not return hidden states")
        stacked_states = []
        for state in hidden_states:
            stacked_states.append(
                state.squeeze(0).detach().cpu().numpy().astype(np.float32)
            )
        siglip_hidden_states = np.stack(stacked_states, axis=0)
        pixel_values_dump = (
            pixel_values_for_encoder.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stream_logits = np.zeros(
        (0, int(next_token_logits.shape[-1])), dtype=np.float32
    )
    stream_generated = np.zeros(
        (int(input_ids.shape[0]), 0), dtype=np.int64
    )
    if args.stream_steps > 0:
        with torch.no_grad():
            generation = model.generate(
                **model_inputs,
                max_new_tokens=args.stream_steps,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        scores = generation.scores or []
        if scores:
            stream_logits = np.stack(
                [
                    score[0].detach().float().cpu().numpy().astype(np.float32)
                    for score in scores
                ],
                axis=0,
            )
            stream_generated = (
                generation.sequences[:, -len(scores) :]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64)
            )

    archive_kwargs = dict(
        input_ids=input_ids.cpu().numpy(),
        attention_mask=attention_mask.cpu().numpy(),
        position_ids=position_ids.cpu().numpy(),
        rope_deltas=rope_deltas.cpu().numpy(),
        image_grid_thw=image_grid_thw.cpu().numpy(),
        siglip_position_ids=siglip_position_ids.cpu().numpy(),
        siglip_hidden=siglip_hidden_flat.cpu().numpy().astype(np.float32),
        projector_embeddings=projector_concat.cpu().numpy().astype(np.float32),
        fused_embeddings=fused_embeddings.cpu().numpy().astype(np.float32),
        next_token_logits=next_token_logits.cpu().numpy().astype(np.float32),
        stream_generated_ids=stream_generated,
        stream_logits=stream_logits,
    )
    if siglip_hidden_states is not None:
        archive_kwargs["siglip_hidden_states"] = siglip_hidden_states
    if pixel_values_dump is not None:
        archive_kwargs["pixel_values_for_encoder"] = pixel_values_dump

    np.savez_compressed(output_path, **archive_kwargs)

    resolved_images = [str(Path(path).resolve()) for path in image_paths]
    meta = {
        "prompt": prompt,
        "prompt_with_placeholders": prompt_with_tokens,
        "image": resolved_images[0],
        "images": resolved_images,
        "model_dir": str(Path(args.model_dir).resolve()),
        "output": str(output_path),
        "seq_len": int(input_ids.shape[1]),
        "vision_tokens": int(projector_concat.shape[0]),
        "vocab_size": int(next_token_logits.shape[-1]),
        "stream_steps": int(stream_generated.shape[1]),
    }
    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    print(
        f"Fixture saved to {output_path} (seq={meta['seq_len']}, "
        f"vision_tokens={meta['vision_tokens']})"
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture PaddleOCR-VL tensors for parity testing"
    )
    parser.add_argument(
        "--model-dir",
        default="PaddleOCR-VL",
        help="Path to the PaddleOCR-VL Hugging Face export",
    )
    parser.add_argument(
        "--image",
        dest="images",
        action="append",
        default=None,
        help="Image to feed into the SigLIP encoder (repeat for multi-image prompts)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Inline prompt text (use '<image>' as the placeholder)",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Optional file containing the prompt text",
    )
    parser.add_argument(
        "--output",
        default="baselines/fixtures/paddleocr_vl/sample_doc.npz",
        help="Destination for the captured tensors",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to run the reference model on",
    )
    parser.add_argument(
        "--torch-dtype",
        default="float32",
        help="Torch dtype to load the reference weights with",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Optional cap on the processed image area before patch extraction",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help="Optional floor on the processed image area",
    )
    parser.add_argument(
        "--stream-steps",
        type=int,
        default=0,
        help="Number of greedy decode steps to capture streaming logits for",
    )
    parser.add_argument(
        "--dump-vision-hidden",
        action="store_true",
        help="If set, store SigLIP embedding + per-layer hidden states for debugging",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    capture_fixture(args)


if __name__ == "__main__":
    main()
