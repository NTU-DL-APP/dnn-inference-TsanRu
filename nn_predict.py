import numpy as np
import json

# === Activation functions ===
def relu(x):
    """
    實現 Rectified Linear Unit (ReLU) 函數。
    它會回傳 x 和 0 之間的較大值。
    """
    return np.maximum(0, x)

def softmax(x):
    """
    實現 SoftMax 函數。
    為了數值穩定性（避免 overflow），先將每個樣本的數值減去其最大值。
    """
    # 確保輸入是二維陣列以處理單一樣本和批次樣本
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    # 減去最大值以提高數值穩定性
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    
    # 計算機率分佈
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# === Flatten ===
def flatten(x):
    """
    將輸入資料展平。
    """
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    """
    實現全連接層（Dense layer）的計算。
    """
    return x @ W + b

# Infer TensorFlow h5 model using numpy
# Support only Dense, Flatten, relu, softmax now
def nn_forward_h5(model_arch, weights, data):
    """
    使用 NumPy 進行神經網路的前向傳播。
    """
    x = data
    for layer in model_arch:
        lname = layer['name']
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            if cfg.get("activation") == "relu":
                x = relu(x)
            elif cfg.get("activation") == "softmax":
                x = softmax(x)

    return x


# You are free to replace nn_forward_h5() with your own implementation 
def nn_inference(model_arch, weights, data):
    """
    神經網路推論的主函數。
    """
    return nn_forward_h5(model_arch, weights, data)