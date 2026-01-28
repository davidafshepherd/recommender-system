import csv
import numpy as np
import time


def load_train_set(train_csv):
    """Loads the train set from the train CSV file into 3 numpy arrays.
    
    Args:
        train_csv (str): Path to the train CSV file.
    
    Returns:
        tuple: 3 numpy arrays containing:
            - user_ids: Array of user IDs
            - item_ids: Array of item IDs
            - ratings: Array of corresponding ratings
    """

    # Log the start of loading 
    print("Loading train set...")

    # Initialize empty lists to store the train set
    user_ids = []
    item_ids = []
    ratings = []
    
    # Read the train CSV file and process each row
    with open(train_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # Convert each column to the appropriate numeric type
            user_ids.append(int(row[0]))      # Convert UserID to integer
            item_ids.append(int(row[1]))      # Convert ItemID to integer
            ratings.append(float(row[2]))     # Convert Rating to float
    
    # Convert the lists to numpy arrays for efficient computation
    user_ids = np.array(user_ids)
    item_ids = np.array(item_ids)
    ratings = np.array(ratings)
    
    # Log the end of loading
    print(f"Finished loading {len(ratings)} ratings")

    return user_ids, item_ids, ratings




class MatrixFactorization:
    """Implements a matrix factorization model, where the user-item rating matrix is
    decomposed into 2 lower-dimensional matrices: user factors and item factors. The
    model also includes bias terms to account for user and item rating tendencies.
    
    Attributes:
        num_factors (int): Number of latent factors for the decomposition
        max_user_id (int): Maximum user ID in the dataset
        max_item_id (int): Maximum item ID in the dataset
        user_factors (numpy.ndarray): Matrix of user factors
        item_factors (numpy.ndarray): Matrix of item factors
        user_bias (numpy.ndarray): Array of user bias terms
        item_bias (numpy.ndarray): Array of item bias terms
        global_bias (float): Global bias term for the model
    """

    def __init__(self, max_user_id, max_item_id, num_factors):
        """Initializes the matrix factorization model with random factors.
        
        Args:
            max_user_id (int): Maximum user ID in the dataset
            max_item_id (int): Maximum item ID in the dataset
            num_factors (int): Number of latent factors for the decomposition
        """

        # Store the number of latent factors and the maximum user and item IDs
        self.num_factors = num_factors
        self.max_user_id = max_user_id
        self.max_item_id = max_item_id

        # Initialize user and item factors with small random values from a normal distribution with mean 0 and std 0.1
        self.user_factors = np.random.normal(0, 0.1, (max_user_id + 1, num_factors))
        self.item_factors = np.random.normal(0, 0.1, (max_item_id + 1, num_factors))

        # Initialize bias terms to zero
        self.user_bias = np.zeros(max_user_id + 1)      # User-specific bias terms
        self.item_bias = np.zeros(max_item_id + 1)      # Item-specific bias terms
        self.global_bias = 0.0                          # Global bias term


    def predict(self, user_ids, item_ids):
        """Predicts ratings for given user-item pairs. The prediction is computed as:
        rating = dot_product(user_factors, item_factors) + user_bias + item_bias + global_bias
        
        Args:
            user_ids (numpy.ndarray): Array of user IDs to predict for
            item_ids (numpy.ndarray): Array of item IDs to predict for
            
        Returns:
            numpy.ndarray: Array of predicted ratings for the given user-item pairs
        """

        # Get the latent factors and bias terms for the given user and item IDs
        user_factors = self.user_factors[user_ids]      # Shape: (batch_size, num_factors)
        item_factors = self.item_factors[item_ids]      # Shape: (batch_size, num_factors)
        user_bias = self.user_bias[user_ids]            # Shape: (batch_size,)
        item_bias = self.item_bias[item_ids]            # Shape: (batch_size,)

        # Compute predictions using the dot product of user and item factors plus bias terms
        predictions = np.sum(user_factors * item_factors, axis=1)       # Shape: (batch_size,)
        predictions += user_bias + item_bias + self.global_bias

        return predictions




def train_model(model, user_ids, item_ids, ratings, batch_size, num_epochs, learning_rate, regularization):
    """Trains a matrix factorization model using stochastic gradient descent (SGD). The
    function processes the train set (user IDs, item IDs and corresponding ratings) in
    batches and updates the model parameters (user factors, item factors and bias terms)
    to minimize the prediction error.
    
    Args:
        model (MatrixFactorization): The matrix factorization model to train
        user_ids (numpy.ndarray): Array of user IDs from the train set
        item_ids (numpy.ndarray): Array of item IDs from the train set
        ratings (numpy.ndarray): Array of ratings from the train set
        batch_size (int): Size of batches for training
        num_epochs (int): Number of complete passes through the train set
        learning_rate (float): Step size for gradient descent updates
        regularization (float): L2 regularization parameter to prevent overfitting
        
    Returns:
        MatrixFactorization: Trained model with optimized parameters
    """

    # Calculate the total number of ratings and the number of batches
    total_ratings = len(ratings)
    num_batches = (total_ratings + batch_size - 1) // batch_size

    # Log the start of training
    print(f"\nStarting training with {total_ratings} ratings...")
    
    # Training loop over epochs
    for epoch in range(num_epochs):
        total_loss = 0
        start_time = time.time()
        
        # Shuffle the train set to ensure random order in each epoch
        indices = np.random.permutation(total_ratings)
        user_ids_shuffled = user_ids[indices]
        item_ids_shuffled = item_ids[indices]
        ratings_shuffled = ratings[indices]
        
        # Process the train set in batches
        for batch_idx in range(num_batches):
            # Calculate batch boundaries
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_ratings)
            
            # Get current batch
            batch_users = user_ids_shuffled[start_idx:end_idx]
            batch_items = item_ids_shuffled[start_idx:end_idx]
            batch_ratings = ratings_shuffled[start_idx:end_idx]
            
            # Forward pass: compute predictions and errors for current batch
            predictions = model.predict(batch_users, batch_items)
            errors = batch_ratings - predictions
            
            # Get user and item factors for current batch
            user_factors = model.user_factors[batch_users]
            item_factors = model.item_factors[batch_items]
            
            # Reshape errors for broadcasting in gradient computation
            errors = errors.reshape(-1, 1)
            
            # Update model parameters using SGD: param += learning_rate * (gradient - regularization * param)
            model.user_factors[batch_users] += learning_rate * (errors * item_factors - regularization * user_factors)          # User factors update
            model.item_factors[batch_items] += learning_rate * (errors * user_factors - regularization * item_factors)          # Item factors update
            model.user_bias[batch_users] += learning_rate * (errors.ravel() - regularization * model.user_bias[batch_users])    # User bias terms update
            model.item_bias[batch_items] += learning_rate * (errors.ravel() - regularization * model.item_bias[batch_items])    # Item bias terms update
            model.global_bias += learning_rate * (np.sum(errors) - regularization * model.global_bias)                          # Global bias term update
            
            # Accumulate loss for this batch
            total_loss += np.sum(errors ** 2)
        
        # Compute epoch statistics (average loss and epoch time)
        avg_loss = total_loss / total_ratings
        epoch_time = time.time() - start_time

        # Log epoch statistics
        print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s')
    
    return model




def load_test_set(test_csv):
    """Loads the test set from the test CSV file into 3 numpy arrays.
    
    Args:
        test_csv (str): Path to the test CSV file.
        
    Returns:
        tuple: 3 numpy arrays containing:
            - user_ids: Array of user IDs
            - item_ids: Array of item IDs
            - timestamps: Array of corresponding timestamps
    """

    # Log the start of loading
    print("\nLoading test set...")
    
    # Initialize empty lists to store the test set
    user_ids = []
    item_ids = []
    timestamps = []
    
    # Read the test CSV file and process each row
    with open(test_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # Convert each column to the appropriate numeric type
            user_ids.append(int(row[0]))      # Convert UserID to integer
            item_ids.append(int(row[1]))      # Convert ItemID to integer
            timestamps.append(int(row[2]))    # Convert timestamp to integer
    
    # Convert the lists to numpy arrays for efficient computation
    user_ids = np.array(user_ids)
    item_ids = np.array(item_ids)
    timestamps = np.array(timestamps)

    # Log the end of loading
    print(f"Finished loading {len(user_ids)} ratings")

    return user_ids, item_ids, timestamps




def predict_and_save(model, user_ids, item_ids, timestamps, output_csv):
    """Makes predictions on the test set (user IDs, item IDs and corresponding timestamps)
     using a trained matrix factorization model and saves them to a CSV file.
    
    Args:
        model (MatrixFactorization): Trained matrix factorization model
        user_ids (numpy.ndarray): Array of user IDs from the test set
        item_ids (numpy.ndarray): Array of item IDs from the test set
        timestamps (numpy.ndarray): Array of timestamps from the test set
        output_csv (str): Path to save the predictions
    """

    # Log the start of predicting
    print("\nMaking predictions...")
    
    # Make predictions for all user-item pairs at once
    predictions = model.predict(user_ids, item_ids)
    
    # Process predictions
    processed_predictions = []
    for i in range(len(predictions)):
        # Clip and round prediction
        pred_rating = float(predictions[i])
        pred_rating = np.clip(pred_rating, 0.5, 5.0)    # Clip to valid range (0.5 to 5.0)
        pred_rating = round(pred_rating * 2) / 2                     # Round to nearest 0.5
        
        # Store prediction with timestamp
        processed_predictions.append((user_ids[i], item_ids[i], pred_rating, timestamps[i]))
    
    # Log the start of saving
    print("Saving predictions to file...")

    # Save predictions to a CSV file
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(processed_predictions)

    # Log the end of saving
    print(f"Saved {len(processed_predictions)} predictions to {output_csv}")




def main():
    # Load the train set (user IDs, item IDs and corresponding ratings)
    user_ids, item_ids, ratings = load_train_set('train_20M_withratings.csv')

    # Initialize a matrix factorization model
    model = MatrixFactorization(max_user_id=np.max(user_ids), max_item_id=np.max(item_ids), num_factors=100)
    
    # Train the model on the train set
    model = train_model(model, user_ids, item_ids, ratings, batch_size=1024, num_epochs=100, learning_rate=0.001, regularization=0.02)

    # Load the test set (user IDs, item IDs and corresponding timestamps)
    user_ids, item_ids, timestamps = load_test_set('test_20M_withoutratings.csv')

    # Make predictions on the test set using the trained model and save to a CSV file
    predict_and_save(model, user_ids, item_ids, timestamps, 'predictions.csv')




if __name__ == "__main__":
    main()