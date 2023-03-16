import pandas as pd
import numpy as np
def get_results(gen ="targets.csv", root= "data-bin/raw", mode="valid", model_name='gcn'):
    preds_df = pd.read_csv(gen, header=0).sort_values('gdb_idx',ascending=True).reset_index()
    true_df = pd.read_csv(root+'/'+mode+'/'+mode+'_target.csv', header=0).sort_values('gdb_idx', ascending=True).reset_index()
    #print(preds['gdb_idx'][0:10], '\n', true['gdb_idx'][0:10])
    
    #print(sum(preds['gdb_idx'].eq(true['gdb_idx'])), len(preds))
    preds = preds_df.values[:,1:]
    true = true_df.values[:,1:]
    mse = get_mse(preds, true)
    mae = get_mae(preds, true)
    #print(preds[0], '\n', true[0], '\n')
    print(f"Model name: {model_name}, MSE: {mse}, MAE: {mae}")

def get_mse(preds, true):
    return np.mean(((true-preds)**2))

def get_mae(preds, true):
    return np.mean(np.abs(preds-true))

