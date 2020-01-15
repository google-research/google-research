# Codebase for "ProtoAttend: Attention-Based Prototypical Learning."

Authors: Sercan O. Arik and Tomas Pfister

Paper: Sercan O. Arik and Tomas Pfister, "ProtoAttend: Attention-Based Prototypical Learning"
Link: https://arxiv.org/abs/1902.06292

We propose a novel inherently interpretable machine learning method that bases decisions on few relevant examples that we call prototypes. Our method, ProtoAttend, can be integrated into a wide range of neural network architectures including pre-trained models. It utilizes an attention mechanism that relates the encoded representations to samples in order to determine prototypes. The resulting model outperforms state of the art in three high impact problems without sacrificing accuracy of the original model: (1) it enables high-quality interpretability that outputs samples most relevant to the decision-making (i.e. a sample-based interpretability method); (2) it achieves state of the art confidence estimation by quantifying the mismatch across prototype labels; and (3) it obtains state of the art in distribution mismatch detection. All this can be achieved with minimal additional test time and a practically viable training time computational cost.

This codebase exemplifies the ProtoAttend training and evaluation pipeline for Fashion-MNIST dataset, using ResNet as the image encoder model.

To run the training pipeline, simply use `python3 main_protoattend.py`. The results and visualizations will be ported to Tensorboard.

To modify the experiment to other datasets and models:
- Implement data batching and preprocessing functions (modify `input_data.py` and data iterators like `iter_train` etc.).
- Integrate the encoder model function suitable for the data type (modify `cnn_encoder` in `model.py`).
- Reoptimize the learning hyperparameters for the new dataset.
