from nemo.core.classes.modelPT import ModelPT


def main():
    model = ModelPT.load_for_inference("gpt_infer.yaml")

    output = model.generate_text(["Deep learning is"], end_strings=["."], tokens_to_generate=50)
    print(output)


if __name__ == "__main__":
    main()
