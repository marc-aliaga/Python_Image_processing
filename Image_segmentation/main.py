
import os
from segformer import segment_image


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    input_dir = os.path.join(base_dir, "imagenes")
    output_dir = os.path.join(input_dir, "masks")

    mask_paths = segment_image(input_dir, output_dir)

    print(f"\nProceso terminado. Máscaras generadas: {len(mask_paths)}")

    for mask_path in mask_paths:
        print(mask_path)


if __name__ == "__main__":
    main()
