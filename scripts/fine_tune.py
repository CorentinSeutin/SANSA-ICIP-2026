from weights.local_weights import local_weight, load_lisa_model

import argparse, os, sys, deepspeed, torch, transformers

from pathlib import Path
from tqdm import tqdm
from functools import partial
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

repo_root = Path(__file__).resolve().parents[2] / "LISA"
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import SANSADataset, collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)

LOSS_NAMES = (
    "loss",
    "ce_loss",
    "mask_bce_loss",
    "mask_dice_loss",
    "mask_loss",
)


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Model Training")
    parser.add_argument("dataset_name")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parsed_args = parser.parse_args(args)

    dataset_name = parsed_args.dataset_name
    return argparse.Namespace(
        batch_size=2,
        bce_loss_weight=0.25,
        beta1=0.9,
        beta2=0.95,
        ce_loss_weight=0.1,
        conv_type="llava_v1",
        dataset_dir=str(project_root / "data" / "for_lisa" / dataset_name),
        dice_loss_weight=1.0,
        epochs=200,
        grad_accumulation_steps=8,
        image_size=1024,
        local_rank=parsed_args.local_rank,
        log_dir=str(project_root / "runs" / dataset_name),
        lora_alpha=16,
        lora_dropout=0.05,
        lora_r=8,
        lora_target_modules="q_proj,v_proj,k_proj,o_proj",
        lr=1e-4,
        model_max_length=2048,
        out_dim=256,
        patience=5,
        precision="fp16",
        steps_per_epoch=170,
        use_mm_start_end=True,
        version=local_weight("lisa"),
        vision_pretrained=local_weight("sam"),
        vision_tower=local_weight("clip"),
        workers=4,
    )


def train(train_loader, model, epoch, scheduler, writer, args):
    loss_meters = {
        name: AverageMeter(name, ":.4f")
        for name in LOSS_NAMES
    }

    model.train()

    for input_dict in tqdm(train_loader, total=len(train_loader), desc=f"[INFO] Training [{epoch}/{args.epochs}]"):
        input_dict = dict_to_cuda(input_dict)

        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        output_dict = model(**input_dict)

        batch_size = input_dict["images"].size(0)
        for name, meter in loss_meters.items():
            meter.update(output_dict[name].item(), batch_size)

        model.backward(output_dict["loss"])
        model.step()

    if args.local_rank == 0:
        for name, meter in loss_meters.items():
            writer.add_scalar(f"train/{name}", meter.avg, epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)


def validate(val_loader, model, epoch, writer, args):
    loss_meters = {
        name: AverageMeter(name, ":.4f")
        for name in LOSS_NAMES
    }
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model.eval()

    for input_dict in tqdm(val_loader, total=len(val_loader), desc=f"[INFO] Validating [{epoch}/{args.epochs}]"):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        with torch.no_grad():
            output_dict = model(**input_dict)

        batch_size = input_dict["images"].size(0)
        for name, meter in loss_meters.items():
            meter.update(output_dict[name].item(), batch_size)

        pred_masks = output_dict["pred_masks"]
        # Process every image in the validation batch.  Validation uses the
        # training batch size to avoid LISA's singleton-batch shape issue.
        for sample_masks, sample_preds in zip(output_dict["gt_masks"], pred_masks):
            masks_list = sample_masks.int()
            output_list = (sample_preds > 0).int()
            intersection, union, acc_iou = 0.0, 0.0, 0.0
            for mask_i, output_i in zip(masks_list, output_list):
                intersection_i, union_i, _ = intersectionAndUnionGPU(
                    output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
                )
                intersection += intersection_i
                union += union_i
                acc_iou += intersection_i / (union_i + 1e-5)
                acc_iou[union_i == 0] += 1.0  # no-object target
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
            intersection_meter.update(intersection), union_meter.update(
                union
            ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    if args.local_rank == 0:
        val_metrics = {
            name: meter.avg
            for name, meter in loss_meters.items()
        }
        val_metrics.update(giou=giou, ciou=ciou)

        for name, value in val_metrics.items():
            writer.add_scalar(f"val/{name}", value, epoch)

        print(f"[INFO] gIoU: {giou:.4f}, cIoU: {ciou:.4f}")

    return giou, ciou


def main(args):
    args = parse_args(args)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LISA training; no CUDA GPU is available.")
    torch.cuda.set_device(args.local_rank)
    print(f"[INFO][DEVICE] GPU: {torch.cuda.get_device_name(args.local_rank)}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": False,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = load_lisa_model(
        LISAForCausalLM, args.version, torch_dtype=torch_dtype,
        low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    model.get_model().initialize_lisa_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    # llava_v1
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        # model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True

    train_dataset = SANSADataset(
        Path(args.dataset_dir) / "train",
        tokenizer,
        args.vision_tower,
        precision=args.precision,
        image_size=args.image_size,
    )

    val_dataset = SANSADataset(
        Path(args.dataset_dir) / "val",
        tokenizer,
        args.vision_tower,
        precision=args.precision,
        image_size=args.image_size,
    )
    print(
        f"[INFO] Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
    )

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "torch_adam": True,
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 3,
            "reduce_bucket_size": "auto",

            # Keep ZeRO-3 without the quantizer CUDA extension.  Enabling this
            # option JIT-compiles `quantizer`, which requires a newer C++ toolchain.
            "zero_quantized_weights": False,
            "zero_hpz_partition_size": 8,
            "zero_quantized_gradients": False,

            "contiguous_gradients": True,
            "overlap_comm": True
        }
    }

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        )
    )

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    # validation dataset
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, shuffle=False, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        # LISA's validation forward currently squeezes hidden states for a
        # singleton batch, which makes its segmentation-token mask invalid.
        # Keep validation batches aligned with training batches instead.
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        sampler=val_sampler,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type, # llava_v1
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
    )

    best_score = -float("inf")
    patience = args.patience
    counter = 0

    # Training
    for epoch in range(args.epochs):
        train(train_loader, model_engine, epoch, scheduler, writer, args)
        giou, _ = validate(val_loader, model_engine, epoch, writer, args)
            
        if giou > best_score:
            best_score = giou
            
            counter = 0  # reset

            save_dir = os.path.join(args.log_dir, "deepspeed_checkpoint")
            os.makedirs(save_dir, exist_ok=True)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)
        else:
            print(f"[INFO] No improvement [{counter}/{patience}]")
            counter += 1

        if counter >= patience:
            print("[INFO] Early stopping triggered. No improvement in gIoU for {} epochs.".format(patience))
            break        


if __name__ == "__main__":
    main(sys.argv[1:])
