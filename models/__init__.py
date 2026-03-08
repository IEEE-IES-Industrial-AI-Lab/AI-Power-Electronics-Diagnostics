from models.cnn_waveform_classifier import CNN1DWaveformClassifier
from models.spectrogram_cnn import SpectrogramCNN
from models.transformer_signal import TransformerSignalClassifier
from models.lstm_classifier import BiLSTMClassifier
from models.autoencoder_anomaly import AutoencoderAnomalyDetector

MODEL_REGISTRY = {
    "cnn_waveform": CNN1DWaveformClassifier,
    "spectrogram_cnn": SpectrogramCNN,
    "transformer": TransformerSignalClassifier,
    "bilstm": BiLSTMClassifier,
    "autoencoder": AutoencoderAnomalyDetector,
}

__all__ = [
    "CNN1DWaveformClassifier",
    "SpectrogramCNN",
    "TransformerSignalClassifier",
    "BiLSTMClassifier",
    "AutoencoderAnomalyDetector",
    "MODEL_REGISTRY",
]
