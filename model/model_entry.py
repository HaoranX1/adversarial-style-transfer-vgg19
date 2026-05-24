from __future__ import annotations

from model.vgg19_wrapper import VGG19Classifier, VGG19FeatureExtractor


def build_models(args):
    """构建固定 VGG19 分类器和特征提取器。"""
    feature_layers = sorted(set(args.style_layers + args.content_layers))
    classifier = VGG19Classifier().to(args.device)
    extractor = VGG19FeatureExtractor(feature_layers).to(args.device)
    classifier.eval()
    extractor.eval()
    return classifier, extractor
