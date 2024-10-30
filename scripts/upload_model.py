import argparse

from huggingface_hub import login as huggingface_login

from cde.lib.utils import analyze_utils

# huggingface_login()


def main():
    parser = argparse.ArgumentParser(description="Alias Converter")

    parser.add_argument("alias", type=str, help="Local alias")
    parser.add_argument("hf_alias", type=str, help="Model alias for HuggingFace")

    args = parser.parse_args()

    model = analyze_utils.load_model_from_alias(args.alias)
    
    model.config.__class__.register_for_auto_class()
    model.__class__.register_for_auto_class("AutoModel")
    model.push_to_hub(args.hf_alias)


if __name__ == "__main__":
    main()
