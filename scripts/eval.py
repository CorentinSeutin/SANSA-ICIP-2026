from weights.local_weights import PROJECT_ROOT, local_weight, load_lisa_model

import argparse, sys, torch, transformers

from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader

LISA_ROOT = PROJECT_ROOT.parent / "LISA"
sys.path.insert(0, str(LISA_ROOT))

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import SANSADataset, collate_fn
from utils.utils import (
    AverageMeter,
    Summary,
    dict_to_cuda,
    intersectionAndUnionGPU,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a LISA model")
    parser.add_argument("model_name", help="Training dataset name, or 'base'")
    parser.add_argument("test_dataset_name")
    parsed_args = parser.parse_args()

    is_base_model = parsed_args.model_name == "base"
    model_path = (
        local_weight("lisa")
        if is_base_model
        else PROJECT_ROOT / "runs" / parsed_args.model_name / "model"
    )
    result_group = (
        "base_model"
        if is_base_model
        else f"trained_on_{parsed_args.model_name}"
    )

    return argparse.Namespace(
        batch_size=1,
        bce_loss_weight=0.25,
        ce_loss_weight=0.1,
        conv_type="llava_v1",
        data_dir=PROJECT_ROOT / "data" / "for_lisa" / parsed_args.test_dataset_name / "test",
        dice_loss_weight=1.0,
        image_size=1024,
        model_max_length=2048,
        model_path=model_path,
        out_dim=256,
        precision="fp16",
        save_dir=PROJECT_ROOT / "eval_results" / result_group / f"tested_on_{parsed_args.test_dataset_name}",
        use_mm_start_end=True,
        vision_pretrained=local_weight("sam"),
        vision_tower=local_weight("clip"),
        workers=2,
    )


@torch.no_grad()
def evaluate(model, dataloader):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model.eval()

    for input_dict in tqdm(dataloader, desc="[INFO] Evaluating"):
        input_dict = dict_to_cuda(input_dict)
        input_dict["images"] = input_dict["images"].half()
        input_dict["images_clip"] = input_dict["images_clip"].half()
        output_dict = model(**input_dict)

        pred_masks = output_dict["pred_masks"]
        masks = output_dict["gt_masks"][0].int()
        predictions = (pred_masks[0] > 0).int()

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask, prediction in zip(masks, predictions):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                prediction.contiguous().clone(),
                mask.contiguous(),
                2,
                ignore_index=255,
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0

        intersection_meter.update(intersection.cpu().numpy())
        union_meter.update(union.cpu().numpy())
        acc_iou_meter.update(
            acc_iou.cpu().numpy() / masks.shape[0],
            n=masks.shape[0],
        )

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    return acc_iou_meter.avg[1], iou_class[1]


def main():
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LISA evaluation; no CUDA GPU is available.")
    torch.cuda.set_device(0)
    print(f"[INFO][DEVICE] GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    model = load_lisa_model(
        LISAForCausalLM,
        args.model_path,
        bce_loss_weight=args.bce_loss_weight,
        ce_loss_weight=args.ce_loss_weight,
        dice_loss_weight=args.dice_loss_weight,
        low_cpu_mem_usage=True,
        out_dim=args.out_dim,
        seg_token_idx=tokenizer("[SEG]", add_special_tokens=False).input_ids[0],
        torch_dtype=torch.float16,
        train_mask_decoder=False,
        use_mm_start_end=args.use_mm_start_end,
        vision_pretrained=args.vision_pretrained,
        vision_tower=args.vision_tower,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.get_model().initialize_vision_modules(model.get_model().config)
    model.resize_token_embeddings(len(tokenizer))
    model = model.cuda().eval()

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    dataset = SANSADataset(
        args.data_dir,
        tokenizer,
        args.vision_tower,
        precision=args.precision,
        image_size=args.image_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=0,
        ),
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False,
    )

    giou, ciou = evaluate(model, dataloader)
    results = f"gIoU: {giou:.4f}\ncIoU: {ciou:.4f}\n"
    (args.save_dir / "results.txt").write_text(results, encoding="utf-8")
    print(f"[INFO] {results.strip()}")


if __name__ == "__main__":
    main()
