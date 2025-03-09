import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
from sklearn.preprocessing import OneHotEncoder
import time
from models.pcn_model import PCN, eval_model 

from scripts.preprocess import preprocess
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.sparse  

file_path = os.path.join('data', 'spam.csv')  
X_train, X_test, y_train, y_test = preprocess(file_path)

dkey = jax.random.PRNGKey(1234)


hidden_dim1 = 128
hidden_dim2 = 64
hidden_dim3 = 32
input_dim = 5000   
output_dim = 2    


model = PCN(dkey = dkey, in_dim=input_dim, out_dim=output_dim, hid1_dim=hidden_dim1,
                  hid2_dim=hidden_dim2, hid3_dim = hidden_dim3)




n_iter = 1 
mb_size = 32 
train_acc_set = [] 
dev_acc_set = []  

best_acc = 0 
patience = 5 
wait = 0  

def batch_generator(X, y, mb_size, key):
    num_samples = X.shape[0]
    indices = jax.random.permutation(key, num_samples)

   
    y_array = y.to_numpy() if isinstance(y, pd.Series) else y

   
    if len(y_array.shape) == 1:  
        encoder = OneHotEncoder(sparse_output=False)
        y_array = encoder.fit_transform(y_array.reshape(-1, 1))

    for i in range(0, num_samples, mb_size):
        batch_indices = indices[i:i + mb_size]

        
        if scipy.sparse.issparse(X):
            Xb = X[batch_indices].toarray()
        else: 
            Xb = X[batch_indices] 

        Yb = y_array[batch_indices]

   
        print(f"Xb shape: {Xb.shape}, Yb shape: {Yb.shape}")

        yield Xb, Yb


for epoch in range(n_iter):
    epoch_start_time = time.time()

    dkey, subkey = jax.random.split(dkey)  
    batch_gen = batch_generator(X_train, y_train, mb_size, subkey)
    
    for Xb, Yb in batch_gen:
        print(f"Xb shape: {Xb.shape}, Yb shape: {Yb.shape}")
        model.train(Xb, Yb) 

  
    accuracy = eval_model(model, X_test, y_test)
    train_acc_set.append(accuracy)
    dev_acc_set.append(accuracy)
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch+1}/{n_iter}, Train Acc: {accuracy:.4f}, Dev Acc: {accuracy:.4f}, Epoch Time: {epoch_time:.2f} seconds")

    if accuracy > best_acc:
        best_acc = accuracy
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

try:
    model.save_to_disk(params_only=True)
except Exception as e:
    print(f"Error saving model: {e}")

# Plotting the learning curve
plt.figure(figsize=(8, 6))
plt.plot(train_acc_set, label='Training Accuracy', marker='o')
plt.plot(dev_acc_set, label='Validation Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.ylim(0, 1)  
plt.savefig('exp/learning_curve.png', dpi=300)
plt.show()