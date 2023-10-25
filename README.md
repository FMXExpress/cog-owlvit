# Cog-OWL-ViT

This is an implementation of Google's [OWL-ViT (v1)]([https://github.com/facebookresearch/nougat](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)) as a [Cog](https://github.com/replicate/cog) model. OWL-ViT uses a CLIP backbone to perform text-guided and open-vocabulary object detection. To use the model, simply input the image you'd like to query and enter the objects you would like to query as comma-separated text. For more details, see this [Replicate model](https://replicate.com/alaradirik/owlvit-base-patch32).

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of OWL-ViT to [Replicate](https://replicate.com).

## Basic Usage

To run a prediction:
```bash
cog predict -i image=@data/astronaut.png -i query="human face, rocket, star-spangled banner, nasa badge"
```

To build the cog image and launch the API on your local:
```bash
cog run -p 5000 python -m cog.server.http
```

## References
```
@article{minderer2022simple,
  title={Simple Open-Vocabulary Object Detection with Vision Transformers},
  author={Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, Xiao Wang, Xiaohua Zhai, Thomas Kipf, Neil Houlsby},
  journal={ECCV},
  year={2022},
}
```
