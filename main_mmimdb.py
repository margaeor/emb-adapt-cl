
from sklearn.metrics import f1_score
import pandas as pd 
import random 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np 
from sklearn.decomposition import PCA
import torch
# This loads our contrastive projector class
from contrastive_projector import ContrastiveProjector

random.seed(42)
np.random.seed(42)
torch.manual_seed(42) # Note: this may make performance slower. Remove if necessary.

# Define the possible models and projections that can be used in an experiment
name_to_model_map = {
    'CART': DecisionTreeClassifier(),
    'RF': RandomForestClassifier(),
    'GBM': GradientBoostingClassifier(),
    'MLP': MLPClassifier(),
    'XGB': XGBClassifier(),
    'LR': LogisticRegression(),
    'SVC': SVC(kernel='rbf', C=1, gamma=1)
}

projection_map = {
    'contrastive': lambda: ContrastiveProjector(768, 128, temp=0.1, batch_size=128, num_epochs=10, verbose=0, lr=0.001),
    'pca': lambda: PCA(n_components=128)
}


def load_data_mmimdb(k=5, sources=['image', 'text']):
    """
    Load and preprocess the MM-IMDb dataset embeddings from specified sources.
    
    Parameters:
        k (int): Number of folds for cross-validation. Default is 5.
        sources (list): List of sources to load embeddings from. Default is ['image', 'text'].
    
    Returns:
        pd.DataFrame: A DataFrame containing the combined embeddings from the specified sources,
                      with an additional 'fold' column for cross-validation.
    """
    
    dfs = []
    for source in sources:
        prefix = ''
        if source == 'image':
            prefix = 'ViT'
        if source == 'text':
            prefix = 'bert'
        
        # Load the embeddings for the current source
        df = pd.read_pickle(f'mmimdb/{prefix}_emb_mmimdb.pickle')
        df['source'] = source
        df = df.rename(columns={'genre_index': 'label', 'id': 'ID', 'embedding': 'emb'})
        dfs.append(df)
    
    # Combine the dataframes from all sources
    df_all = pd.concat(dfs)
    
    # Pivot the table to have separate columns for each source's embeddings
    df_all = df_all.pivot_table(index='ID', values=['label', 'emb'], columns='source', aggfunc='first')
    df_all.columns = ['_'.join([str(i) for i in col]) for col in df_all.columns]
    df_all = df_all.dropna()
    
    # Ensure that the labels are consistent across sources
    if len(sources) > 0:
        for src in sources[1:]:
            assert (df_all[f'label_{src}'] != df_all[f'label_{src}']).sum() == 0
            del df_all[f'label_{src}']
    
    # Rename the label column and reset the index
    df_all = df_all.rename(columns={f'label_{sources[0]}': 'label'}).reset_index()
    
    # Assign random folds for cross-validation
    folds = [random.randint(0, k-1) for _ in range(0, df_all.shape[0])]
    df_all['fold'] = folds
    
    return df_all



def run_experiment(df, fold, projection, multi_proj, model_name, sources=['image', 'text']):
    """
    Run an experiment with the given projection, model and parameters.
    
    Parameters:
        df (pd.DataFrame): 
            DataFrame containing the data with embeddings and labels.
        fold (int): 
            The fold id to use for splitting the data into train and test sets.
        projection (str): 
            The type of projection to use ('none', 'contrastive', or 'pca').
        multi_proj (bool): 
            Whether to use multiple projections for each source.
        model_name (str): 
            The name of the model to use for classification (models defined in `name_to_model_map`).
        sources (list of str, optional): 
            List of sources to use for embeddings. Default is ['image', 'text'].

    Prints:
        The F1 score of the model on the test set.
    """

    # Split the data into training and testing sets based on the fold
    train = df[df['fold'] != fold]
    test = df[df['fold'] == fold]

    if projection == 'none':
        # No projection, concatenate embeddings from all sources
        X_train = np.concatenate([np.stack(train[f'emb_{src}'].values) for src in sources], axis=1)
        X_test = np.concatenate([np.stack(test[f'emb_{src}'].values) for src in sources], axis=1)
    elif multi_proj:
        # Use multiple projections for each source
        X_train, X_test = [], []

        for src in sources:
            proj = projection_map[projection]()
            X_train_sub, y_train_sub = np.stack(train[f'emb_{src}'].values), train['label'].values
            X_test_sub, y_test_sub = np.stack(test[f'emb_{src}'].values), test['label'].values

            try:
                proj.fit(X_train_sub, y_train_sub)
            except:
                proj.fit(X_train_sub)

            X_train.append(proj.transform(X_train_sub))
            X_test.append(proj.transform(X_test_sub))

        X_train = np.concatenate(X_train, axis=1)
        X_test = np.concatenate(X_test, axis=1)
    else:
        # Use a single projection for concatenated embeddings
        proj = projection_map[projection]()
        X_train = np.concatenate([np.stack(train[f'emb_{src}'].values) for src in sources], axis=1)
        X_test = np.concatenate([np.stack(test[f'emb_{src}'].values) for src in sources], axis=1)
        y_train = train['label'].values
        y_test = test['label'].values

        try:
            proj.fit(X_train, y_train)
        except:
            proj.fit(X_train)

        X_train = proj.transform(X_train)
        X_test = proj.transform(X_test)

    # Get labels for training and testing sets
    y_train, y_test = train['label'].values, test['label'].values

    # Train the model and make predictions
    model = name_to_model_map[model_name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate and print the F1 score
    f1 = f1_score(y_test, y_pred)
    print(f"Projection Type: {projection}, Multi-Proj: {multi_proj}, Model: {model_name}, F1 score: {f1}")



if __name__ == '__main__':

    df = load_data_mmimdb()

    model = 'CART'
    run_experiment(df, 0, 'none', False, model)
    run_experiment(df, 0, 'pca', True, model)
    run_experiment(df, 0, 'contrastive', True, model)

    #print(df.shape) 
