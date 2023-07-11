from nemo.core.classes.modelPT import ModelPT

def main():
    model = ModelPT.auto_load("/models/gpt2b", trainer_args={"devices": 1, "num_nodes": 1, "accelerator": "gpu", "logger": False, "precision": 16})

    output = model.generate_text(["Deep learning is"], end_strings=["."])
    print(output)

if __name__ == "__main__":
    main()
