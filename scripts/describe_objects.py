from weights.local_weights import local_weight

import argparse, json, torch

import torchvision.transforms as T

from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    LogitsProcessor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
TEMPERATURE = 0.1
TOP_P = 0.9
MIN_NEW_TOKENS = 3
MAX_NEW_TOKENS = 120
OTHER_LOGIT = -0.5

# --- Restricted vocabulary ---

ALLOWED_COLORS = {
    # basic
    "black","white","gray","grey","silver","dark","light","beige","brown","blue","green","red",
    "yellow","orange","pink","purple","cream","ivory","gold","bronze","copper","multicolored",
    # extra tones
    "turquoise","teal","navy","indigo","violet","maroon","burgundy","cyan","magenta",
    "lime","olive","khaki","tan","peach","coral","salmon","amber","mustard",
}

ALLOWED_TEXTURE = {
    # surface/material cues
    "glossy","rough","smooth","grainy","polished","shiny","metallic","metallic-like",
    "fabric","fabric-like","rubber-like","plastic-like","wooden","wood-like","stone-like",
    "leathery","fuzzy","soft","coarse","velvety","silky","woolly","bumpy","wrinkled",
    "translucent","transparent","opaque","porous","glassy","paper-like",
}

ALLOWED_SHAPE = {
    # geometric
    "rectangular","rounded","circular","oval","square","triangular","hexagonal","polygonal",
    # proportions
    "tapered","slender","wide","tall","narrow","short","elongated","flattened","bulky",
    # outline/style
    "sharp","flat","curved","straight","irregular","angular","symmetrical","asymmetrical",
    "outline","silhouette","proportions","edges","corners","surface","form","contour","profile",
    "layered","stacked","clustered","pointed","domed","arched","cylindrical","spherical","conical",
}

ALLOWED_PATTERNS = {
    "pattern","striped","dotted","spotted","checkered","plaid","zigzag","swirled","marbled",
    "gradient","faded","blended","uniform","irregular","repeated","grid","lattice","textured",
}

ALLOWED_LIGHTING = {
    "reflective","shadowed","bright","dim","highlighted","contrasted",
    "glossy","matte","subtle","faint","strong","soft","harsh","muted","radiant",
}

ALLOWED_CONNECTORS = {
    "a","an","the","with","and","in","of","on","at","to","from",
    "slightly","subtle","mostly","partly","entirely","overall",
    "appears","seems","resembles","shows","has","features","presents","displays","contains",
    "that","which","where","as","while","although",
    "across","along","around","throughout","towards","near","adjacent","between",
    "area","region","section","part","overall","whole","entire","surface","side","center","middle","top","bottom",
    "very","somewhat","highly","deep","rich","pale","darkened","lightened",
    ",",".",";","-","contrast",
}

ALLOWED_WORDS = (
    ALLOWED_COLORS
    | ALLOWED_TEXTURE
    | ALLOWED_SHAPE
    | ALLOWED_PATTERNS
    | ALLOWED_LIGHTING
    | ALLOWED_CONNECTORS
)


# --- Instruction Prompts (IPs) given to the Description VLM ---

DISP_IP =    (
                "Ignore any black/empty background. Describe ONLY the visible object using strictly visual cues.\n"
                "Use low-level attributes: colors/tones, surface qualities, overall outline/proportions and simple patterns.\n"
            )

EXSP_IP =    (
                "You are a helpful and precise low-level visual assistant.\n"
                "Your goal is to describe the object in the image without using any high-level semantic or conceptual information.\n"
                "To describe the object, you can only use geometric, shape, and other visual properties, such as: shape, geometry, texture, surface patterns, materials, colors, brightness, dimensions.\n"
                "You can choose between these properties, or add additional ones, according to the most dominant ones in the images but never name the object or subparts of the objet in any way.\n"
                "Any semantic information or any knowledge about what the object is or its function is prohibited. You cannot name the object class and any of its parts/components. Do not describe the background.\n\n"
                "Here are some examples of expected words to use (RIGHT)/not to use (WRONG), problematical words are written in capital letters:\n\n"
                "- Tomato:\n"
                "\tWRONG: 'The object is a TOMATO...'\n"
                "\nWRONG: 'The object is a VEGETABLE that CAN BE EATEN...'\n"
                "\nRIGHT: 'The object is a circular and red, with a smooth surface and a green pattern on top. It has a green rectangular subpart on top. The surface appears to be highly reflective. The overall shape of the object is consistent and symmetrical.'\n\n"
                "- Green mug:\n"
                "\tWRONG: 'The object is a MUG...'\n"
                "\tWRONG: 'The object CAN BE USED TO DRINK COFFEE OR TEA...'\n"
                "\tRIGHT: 'The object is cylindrical with an open top and a curved part, with a smooth green texture and is asymmetrical.'\n\n"
                "- Bike:\n"
                "\tWRONG: 'The object has WHEELS and PEDALS...'\n"
                "\tWRONG: 'The object CAN BE USED BY A PERSON TO MOVE...'\n"
                "\tRIGHT: 'The object is a symmetric and complex arrangement of multiple elongated tubular shapes, with metallic texture. The object also has two large circular parts on each side, with repeated straight lines inside arranged in a radial pattern.'\n\n"
                "- Person:\n"
                "\tWRONG: 'The object has a white SHIRT and a blue SHORT...'\n"
                "\tWRONG: 'The PERSON IS WEARING...'\n"
                "\tRIGHT: 'The object is white and blue, elongated, cylindrical. It has a yellow and white pattern on the top.'\n"
            )


# Base code copied directly from InternVL2_5-8B's
# Huggingface page on 30/08/2026: https://huggingface.co/OpenGVLab/InternVL2_5-8B 
  
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# --- Logits processor ---

def build_allowed_token_ids(tokenizer, words):
    ids = set()
    for w in words:
        for variant in (w, " " + w):
            toks = tokenizer.encode(variant, add_special_tokens=False)
            ids.update(toks)

    for s in [" ", ",", ".", ";", "\n"]:
        ids.update(tokenizer.encode(s, add_special_tokens=False))

    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)

    return ids


class AllowedTokensLogitsProcessor(LogitsProcessor):
    """
    Keep original logits for allowed tokens; set all others to a constant penalty.
    This discourages out-of-lexicon words.
    """
    def __init__(self, allowed_token_ids, other_logit=-0.5):
        self.allowed_ids = None
        self.other_logit = other_logit
        self._allowed_tensor = None
        self._allowed_sorted = sorted(list(allowed_token_ids))

    def __call__(self, input_ids, scores):
        if self._allowed_tensor is None or self._allowed_tensor.device != scores.device:
            self._allowed_tensor = torch.tensor(self._allowed_sorted, dtype=torch.long, device=scores.device)

        # fill with penalty, then copy original scores for allowed tokens
        new_scores = torch.full_like(scores, self.other_logit)
        new_scores.index_copy_(
            dim=1,
            index=self._allowed_tensor,
            source=scores.index_select(1, self._allowed_tensor)
        )
        return new_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=("train", "val"))
    ap.add_argument("dataset_name")
    args = ap.parse_args()

    split = args.split
    dataset_name = args.dataset_name

    objects_input_path = DATA_ROOT / "descriptions" / f"{split}_objects_sampled.json"
    objects_out_path = DATA_ROOT / "descriptions" / dataset_name / f"{split}_objects_described.json"

    objects_out_path.parent.mkdir(parents=True, exist_ok=True)

    images_dir = DATA_ROOT / "obj-cropped" / split

    if objects_out_path.is_file():
        print(f"[INFO][{split.upper()}][{dataset_name}][SKIP] Output already exists at: {objects_out_path}")
        return
    print(f"[INFO][{split.upper()}][{dataset_name}] Starting describe_objects.py")

    if not objects_input_path.is_file():
        raise FileNotFoundError(f"[INFO][{split.upper()}][{dataset_name}][ERROR] Input dataset not found: {objects_input_path}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"[INFO][{split.upper()}][{dataset_name}][ERROR] Cropped objects directory not found: {images_dir}")

    with objects_input_path.open("r", encoding="utf-8") as f:
        objects = json.load(f)

    if "DISP" in dataset_name:
        print(f"[INFO][{split.upper()}][{dataset_name}][MODE] Dictionary-based Segmentation Prompts (DISP).")
        question = f"<image>\n{DISP_IP}"
    elif "Baseline" in dataset_name:
        print(f"[INFO][{split.upper()}][{dataset_name}][MODE] Baseline: No description generated needed.")
        pass
    else:
        print(f"[INFO][{split.upper()}][{dataset_name}][MODE] Example-based Segmentation Prompts (EXSP).")
        question = f"<image>\n{EXSP_IP}"

    if "Baseline" in dataset_name:
        pass
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to run the InternVL model; no CUDA GPU is available.")
        device = torch.device("cuda", torch.cuda.current_device())
        print(f"[INFO][{split.upper()}][{dataset_name}][DEVICE] GPU: {torch.cuda.get_device_name(device)}")
        model_path = local_weight("internvl")
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True,
        ).to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True, 
            use_fast=False, 
            local_files_only=True,
        )

        allowed_ids = build_allowed_token_ids(tokenizer, ALLOWED_WORDS)
        logits_processors = [AllowedTokensLogitsProcessor(allowed_ids, other_logit=OTHER_LOGIT)]

        if "DISP" in dataset_name:
            description_config = dict(
                max_new_tokens=MAX_NEW_TOKENS, 
                do_sample=True, 
                logits_processor=logits_processors, 
                temperature=TEMPERATURE, 
                min_new_tokens=MIN_NEW_TOKENS,
                top_p=TOP_P,
            )
        elif "EXSP" in dataset_name:
            description_config = dict(
                max_new_tokens=MAX_NEW_TOKENS, 
                do_sample=True, 
                temperature=TEMPERATURE, 
                min_new_tokens=MIN_NEW_TOKENS,
                top_p=TOP_P,
            )

        if "LLMJ" in dataset_name:
            print(f"[INFO][{split.upper()}][{dataset_name}][SUBMODE] Large Language Model as a Judge (LLMJ).")
            verification_config = dict(
                max_new_tokens=1, 
                do_sample=True, 
                temperature=TEMPERATURE, 
                min_new_tokens=1,
                top_p=TOP_P,
            )

    # For each object, generate a description using the Description VLM (InternVL2_5-8B) or a baseline prompt
    for obj in tqdm(objects, desc=f"[INFO][{split.upper()}][{dataset_name}] Describing objects with the Description VLM"):
        obj_filename = obj.get("obj_filename")

        # Get the vision module’s dtype & device, then cast inputs to match
        if "Baseline" not in dataset_name:
            try:
                vision_param = next(p for p in model.vision_model.parameters() if p.requires_grad)
            except StopIteration:
                # Fallback: any model param
                vision_param = next(model.parameters())

        obj_path = images_dir / obj_filename
        obj_category = obj.get("category")
        
        if "Baseline" not in dataset_name:
            try:
                pixel_values = load_image(obj_path, max_num=12).to(device=vision_param.device, dtype=vision_param.dtype)
            except Exception as e:
                continue

            with torch.inference_mode():
                answer = model.chat(
                    tokenizer,
                    pixel_values,
                    question,
                    description_config,
                )

                obj["description"] = answer

                if "LLMJ" in dataset_name:
                    LLMJ_question = (
                        f"TESTED_SENTENCE: '{answer}'"
                        "Does TESTED_SENTENCE contain any semantic information, 'YES' or 'NO' ?\n"
                        "For example, you have to answer 'YES' for the following sentences:\n"
                        "\t(1): 'The object is a clock, with black boundaries and branches.'\n"
                        "\t(2): 'The image shows a grey or silver airplane flying in the sky. The airplane has multiple rivets on the sides.'\n"
                        "\t(3): 'The object is an animal, black and brown, fluffy texture. Its claws are big and the dog is opening his mouth.'\n"
                        "\t(4): 'The image shows a person, oval shape, white and blue.'\n"
                        "\t(5): 'The object has a golden-brown crust. The surface has a glossy appearance, likely indicating a baked or toasted material.'\n"
                        "\t(6): 'The object in the image is a large aircraft, specifically a Royal Air Force (RAF) aircraft.'\n"
                        "\t(7): 'The object is oval, green and yellow, with a smooth texture, without pattern, likely made of vegetables, with slight reflection and is medium-sized.'\n"
                        "\t(8): 'The object is oval, red and white, with a smooth texture, without pattern, likely made of dough or pastry, with slight reflection.'\n"            
                        "\t(9): 'The object is rectangular, red and white, with a smooth texture, without pattern, likely made of dough or pastry, with slight reflection.'\n"
                        "\t(10): 'The image shows a person. The person is wearing a black shirt. The person has a white beard. The person is smiling.'\n"
                        "Describing the object, sub-object, its class or sub-class are semantic information contrary to "
                        "shape, geometry, texture, surface patterns, materials, colors, "
                        "brightness, shadows, reflections, and any measurable or mathematical properties (" 
                        "curvature, dimensions) information that you have NOT to consider as semantic information.\n"
                        "For example, you have to answer 'NO' for the following sentences:\n"
                        "\t(1): 'The object is circular, red and white, with a smooth texture, without pattern, likely metal or plastic, with slight reflection and is large.'\n"
                        "\t(2): 'The image shows a grey or silver cylindrical object. The object is elongated and the texture is like metal.'\n"
                        "\t(3): 'The object is oval, black and brown, fluffy texture. No reflection.'\n"
                        "\t(4): 'The object is oval, white, smooth texture, without pattern, likely ceramic or porcelain, with slight reflection and is large.'\n"
                        "\t(5): 'The object is rectangular, purple, with a smooth texture, without pattern, likely metal or plastic, with slight reflection and is large.'\n"
                        "\t(6): 'The object is rectangular, brown and yellow, with a smooth texture, has a striped pattern, likely made of leather or fabric, with slight reflection and is medium-sized.'\n"
                        "\t(7): 'The object is rectangular, with a surface pattern of green, red, and yellow colors. The texture appears to be rough, and there is a slight reflection. The object is medium-sized.'\n"
                        "\t(8): 'The object is elongated, white, with a fluffy texture, without pattern, likely made of wool or fur, with slight reflection and is medium-sized.'\n"
                        "\t(9): 'The object is oval, yellow and green, with a smooth texture, without pattern, likely made of plastic or metal, with slight reflection.'\n"
                        "\t(10): '(1): 'The object is elongated, silver and grey, with a smooth texture, without pattern, likely metal, with slight reflection.''\n"
                        "\t(11): 'The object is rectangular, brown, with a smooth texture, has a striped pattern, likely made of leather or fabric, with slight reflection.'\n"
                        "\t(12): 'The object is cylindrical, blue, with a smooth texture, decorated with white star patterns, likely metal or plastic, with slight reflection.'\n"
                        "\t(13): 'The object is circular, white and green, with a smooth texture, without pattern, likely ceramic or porcelain, with slight reflection.'\n"
                        "\t(14): 'The object is cylindrical, transparent, with a smooth texture, without pattern, likely glass, with slight reflection.'\n"
                        "\t(15): 'The object is elongated, blue and grey, with a smooth texture, without pattern, likely metal, with slight reflection.'\n"
                        "\t(16): 'The object is irregularly shaped, brown and black, with a fluffy texture. No reflection.'\n"
                        "\t(17): 'The object is cylindrical, transparent, with a smooth texture, without pattern, likely made of plastic, with slight reflection.'\n"
                        "\t(18): 'The object is elongated, yellow, with a smooth texture, without pattern, likely made of a soft material, with slight reflection.'\n"
                    )
                    LLMJ_answer = model.chat(
                        tokenizer,
                        pixel_values,
                        LLMJ_question,
                        verification_config,
                    )

                    obj["verification"] = LLMJ_answer
        else: # Baseline case
            obj["description"] = f"Can you segment the {obj_category} in this image?"
    
    with objects_out_path.open("w", encoding="utf-8") as f:
        json.dump(objects, f, ensure_ascii=False, indent=2)

    print(f"[INFO][{split.upper()}][{dataset_name}] Saved {len(objects)} described objects in: {objects_out_path}")
    print(f"[INFO][{split.upper()}][{dataset_name}] Finished describe_objects.py")

if __name__ == "__main__":
    main()
