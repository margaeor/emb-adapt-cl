from sklearn.base import BaseEstimator, TransformerMixin
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression


class ContrastiveLoss(nn.Module):
    """
    ContrastiveLoss is the custom loss we use for the Constrastive Learning.
    Args:
        T (float, optional): Temperature parameter to scale the logits. Default is 0.1.
    """
    def __init__(self, T=0.1):
        super(ContrastiveLoss, self).__init__()
        self.bce_loss = torch.nn.BCELoss()
        self.T = T

    def forward(self, emb1, emb2, label):

        logits = torch.sum(emb1 * emb2, axis=1)/self.T
        return F.binary_cross_entropy_with_logits(logits, label)


class ProjectionHead(nn.Module):
    """
    A neural network module for projection head used in contrastive learning.
    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int, optional): Dimension of the output features. Default is 256.
        verbose (int, optional): Verbosity level. Default is 0.
    """
    def __init__(self, input_dim, output_dim=256, verbose=0):
        super(ProjectionHead, self).__init__()
        hidden_dim = min(input_dim//2, 16)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.verbose = verbose

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def contrastive_train(self, X, y, T=0.1, batch_size=128, num_epochs=10, lr=0.001):
        """
        Trains the model using a contrastive loss function.
        
        Parameters:
        -----------
        X : array-like
            The input data.
        y : array-like
            The labels corresponding to the input data.
        T : float, optional, default=0.1
            Temperature parameter for the contrastive loss.
        batch_size : int, optional, default=128
            The number of samples per minibatch.
        num_epochs : int, optional, default=10
            The number of epochs to train the model.
        lr : float, optional, default=0.001
            Learning rate for the optimizer.
        
        """
        
        # Initialize the loss function and optimizer
        criterion = ContrastiveLoss(T)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Split the data into training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create data loaders for training and validation sets
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        val_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training loop
        for epoch in range(num_epochs):
            total_train_loss, total_val_loss = 0.0, 0.0
            train_pair_labels, val_pair_labels = [], []
            train_similarities, val_similarities = [], []

            # Train the model
            iterator = tqdm(train_loader, total=len(train_loader)) if self.verbose > 0 else train_loader
            for batch_X, batch_y in iterator:
                optimizer.zero_grad()
                x1, x2, pair_labels = self.create_pairs_in_batch(batch_X, batch_y)
                output1, output2 = self(x1), self(x2)
                loss = criterion(output1, output2, pair_labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                similarities = torch.nn.functional.cosine_similarity(output1, output2)
                train_pair_labels.extend(pair_labels.cpu().detach().numpy())
                train_similarities.extend(similarities.cpu().detach().numpy())

            # Validate the model
            iterator = tqdm(val_loader, total=len(val_loader)) if self.verbose > 0 else val_loader
            for batch_X, batch_y in iterator:
                x1, x2, pair_labels = self.create_pairs_in_batch(batch_X, batch_y)
                output1, output2 = self(x1), self(x2)
                with torch.no_grad():
                    loss = criterion(output1, output2, pair_labels)
                    total_val_loss += loss.item()
                    similarities = torch.nn.functional.cosine_similarity(output1, output2)
                    val_pair_labels.extend(pair_labels.cpu().detach().numpy())
                    val_similarities.extend(similarities.cpu().detach().numpy())

            # Calculate AUC scores
            train_auc = roc_auc_score(train_pair_labels, train_similarities)
            val_auc = roc_auc_score(val_pair_labels, val_similarities)

            # Print training and validation losses and AUC scores
            if self.verbose > 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_train_loss / len(train_loader)}, "
                      f"Val Loss: {total_val_loss / len(val_loader)}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

    def create_pairs_in_batch(self, X, y):
        """Creates pairs of embeddings and their labels (1 for same class, 0 for different classes) within a batch."""
        # X = torch.tensor(X)
        # y = torch.tensor(y)

        # Get all combinations of indices (i, j) where i < j
        batch_size = len(X)
        idx = torch.triu_indices(batch_size, batch_size, offset=1)

        # Create pairs of embeddings
        pairs_1 = X[idx[0]]
        pairs_2 = X[idx[1]]

        # Create labels for the pairs
        labels = (y[idx[0]] == y[idx[1]]).float()

        return pairs_1, pairs_2, labels



class ContrastiveProjector(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformation class that applies a contrastive learning projection head.
    Parameters
    ----------
    input_dim : int
        The dimensionality of the input features.
    output_dim : int, optional (default=128)
        The dimensionality of the output features after projection.
    temp : float, optional (default=0.1)
        The temperature parameter for contrastive learning.
    batch_size : int, optional (default=128)
        The number of samples per batch during training.
    num_epochs : int, optional (default=10)
        The number of epochs to train the model.
    verbose : int, optional (default=0)
        The verbosity level of the training process.
    lr : float, optional (default=0.001)
        The learning rate for the optimizer.
    Methods
    -------
    fit(X, y)
        Train the model using contrastive learning on X and y.
    transform(X)
        Apply the projection head's forward method to X.
    """

    def __init__(self, input_dim, output_dim=128, temp=0.1, batch_size=128, num_epochs=10, verbose=0, lr=0.001):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.lr = lr
        self.model = ProjectionHead(input_dim, output_dim, verbose)
        self.temp = temp
        
    def fit(self, X, y):
        """Train the model using contrastive learning on X and y."""
        self.model.contrastive_train(X, y, batch_size=self.batch_size, num_epochs=self.num_epochs, T=self.temp,
                                     lr=self.lr)
        return self

    def transform(self, X):
        """Apply the projection head's forward method to X."""
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            return self.model.forward(X_tensor).numpy()



# Sample usage of the ContrastiveProjector class
if __name__ == "__main__":

    # Generate a dummy binary classification dataset
    X, y = make_classification(n_samples=4000, n_features=400, n_informative=350, n_redundant=50, random_state=42)
    
    # Standardize the dataset
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the ContrastiveProjector
    projector = ContrastiveProjector(input_dim=X.shape[1], output_dim=128, num_epochs=20, verbose=1)
    projector.fit(X_train, y_train)
    
    # Transform the data using the trained projector
    X_train_proj = projector.transform(X_train)
    X_test_proj = projector.transform(X_test)
    
    # Train a logistic regression model on the projected data
    clf = LogisticRegression()
    clf.fit(X_train_proj, y_train)
    
    # Make predictions and evaluate the F1 score
    y_pred = clf.predict(X_test_proj)
    f1 = f1_score(y_test, y_pred)
    
    
    # Baseline logistic regression without projection
    clf_baseline = LogisticRegression()
    clf_baseline.fit(X_train, y_train)
    y_pred_baseline = clf_baseline.predict(X_test)
    f1_baseline = f1_score(y_test, y_pred_baseline)
    print(f"Baseline F1 Score: {f1_baseline:.4f}")
    print(f"F1 Score after contrastive projection: {f1:.4f}")