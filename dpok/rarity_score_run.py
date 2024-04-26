import argparse
from dpok_nft.ImageReward.models.RarityScore import RarityScore
import os

def main(vit_weights_path, generated_images_path):
    rarity_model = RarityScore(vit_weights_path)
    generated_images = [os.path.join(generated_images_path, filename) 
                              for filename in os.listdir(generated_images_path) 
                              if filename.endswith(('.png'))]


    for image_path in generated_images:
        rarity_model.compute_reward(image_path=image_path)
    
    rarity_model.normalize_reward()

    mean_rarity_score = rarity_model.calculate_mean_reward()
    print("Mean Rarity Score: ", mean_rarity_score)


if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Compute RarityScore for generated images.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--vit_weights", required=True, help="Path to the pretrained ViT model weights file.")
    parser.add_argument("--generated_images_path", nargs="+", required=True, help="Paths to the generated images.")
    args = parser.parse_args()

    main(args.vit_weights, args.generated_images)

# Example usage:
# python rarity_score_run.py --vit_weights path/to/pretrained/vit/model/weights --generated_images_path path/to/generated/images/folder