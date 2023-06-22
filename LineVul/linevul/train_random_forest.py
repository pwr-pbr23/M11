import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import matthews_corrcoef
import json
import time
from sklearn.model_selection import GridSearchCV


def save_array_to_file(array, file_path):
    np.save(file_path, array)
    print(f"Array saved to {file_path}")


def load_array_from_file(file_path):
    array = np.load(file_path)
    return array


if __name__ == '__main__':
    X_data = load_array_from_file("X_array.npy")
    y_data = load_array_from_file("y_array.npy")
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    n_estimators = 1000  # Number of decision trees in the forest
    criterion = 'gini'  # {“gini”, “entropy”, “log_loss”}
    max_depth = 150  # Maximum depth of each decision tree
    random_state = 42  # Random seed for reproducibility
    rf_model = RandomForestClassifier(criterion=criterion, n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    
    rf = RandomForestClassifier()

    param_grid = {
      'n_estimators': [1000, 1500, 2000],              # Number of trees
      'criterion': ['gini', 'entropy', 'log_loss'],
      'max_depth': [None, 5, 10],                    # Maximum depth of trees
      'min_samples_leaf': [1, 2, 4],                 # Minimum samples for a leaf node
      'max_features': ['auto', 'sqrt', 'log2']       # Maximum number of features to consider
    }

    # Perform a grid search to find the best parameters
    grid_search = GridSearchCV(rf, param_grid, cv=10, verbose=2, n_jobs=4)
    
    print('Fitting!')
    grid_search.fit(X_train, y_train)

    # Get the best parameters and the corresponding score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Train a Random Forest model with the best parameters
    print("TRAINING STARTED!")
    start_time = time.time()
    best_rf = RandomForestClassifier(**best_params)
    best_rf.fit(X_train, y_train)
    end_time = time.time()
    print("TRAINING ENDED!")

    #Evaluate
    y_preds = best_rf.predict(X_test)

    # Calculate metrics
    recall = recall_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds)
    f1 = f1_score(y_test, y_preds)
    mcc = matthews_corrcoef(y_test, y_preds)
    result = {
        "training_time": f'{end_time - start_time} seconds',
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_mcc": float(mcc),
    }
    print(result)
    with open('random_forest_results.json', 'w') as file:
        json.dump(result, file)
    print(f"Result saved to 'random_forest_results.json'")
