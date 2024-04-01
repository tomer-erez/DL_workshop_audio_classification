import librosa
import numpy as np



def extract_mfcc(audio_file, pad_mode='constant', pad_value=0):
    """
    Extracts MFCC features from an audio file.
    Parameters:
    audio_file (str): Path to the audio file
    pad_mode (str): Mode for padding the MFCCs. Default is 'constant'.
    pad_value (int): Value to pad the MFCCs with. Default is 0.
    Returns:
    np.ndarray: Array of MFCC features
    """

    n_fft=1024
    hop_length=1024
    max_length = 318
    num_mfcc=30

    audio, sr = librosa.load(audio_file,sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    # Pad or truncate mfccs to a fixed length
    if mfccs.shape[1] < max_length:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode=pad_mode, constant_values=pad_value)
    else:
        mfccs = mfccs[:, :max_length]

    return mfccs

def predict_wrapper(audio_model,audio_file):
    """
    Predicts the class of an audio file.
    Parameters:
    audio_file (str): Path to the audio file
    Returns:
    str: Predicted class
    """
    
    # Extract MFCC features
    mfccs = extract_mfcc(audio_file)
    # Perform inference
    prediction = audio_model.predict(mfccs)
    # Get the predicted class
    predicted_class = np.argmax(prediction)
    return predicted_class
