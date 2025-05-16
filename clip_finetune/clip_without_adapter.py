import torch
from open_clip import create_model_and_transforms
from peft import LoraConfig, get_peft_model

def get_model_without_adapter(model_size="b"):
    model_name_map =  {
            'b': 'ViT-B-16',
            'l': 'ViT-L-14',
        }
    pretrained_map = {
        'b':'laion2b_s34b_b88k',
        'l':'laion2b_s32b_b82k',
    }
    model_name = model_name_map[model_size]
    pretrained = pretrained_map[model_size]

    pretrained_model, preprocess_train, preprocess_val = create_model_and_transforms(model_name, pretrained=pretrained)
    pretrained_model = pretrained_model.visual
    return pretrained_model, preprocess_train, preprocess_val


def get_model_with_lora(model_size="b"):
    model, preprocess_train, preprocess_val = get_model_without_adapter(model_size)
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["attn"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model, preprocess_train, preprocess_val

