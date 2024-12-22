import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import PassiveAggressiveRegressor, LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
from typing import List, Any, Tuple, Dict
from abc import ABC, abstractmethod
import joblib
import os

# ----------------------------
# Feature Classes
# ----------------------------

class Feature(ABC):
    @abstractmethod
    def vector(self) -> List[float]:
        """Return the numerical representation of the feature."""
        pass

    @classmethod
    @abstractmethod
    def from_vector(cls, vector: List[float], **kwargs) -> 'Feature':
        """Reconstruct a Feature instance from its vector representation."""
        pass

    @abstractmethod
    def __str__(self):
        pass

class IntFeature(Feature):
    def __init__(self, value: int):
        self.value = value

    def vector(self) -> List[float]:
        return [float(self.value)]

    @classmethod
    def from_vector(cls, vector: List[float], **kwargs) -> 'IntFeature':
        if len(vector) != 1:
            raise ValueError("IntFeature vector must be of length 1.")
        return cls(int(vector[0]))

    def __str__(self):
        return f"IntFeature(value={self.value})"

class FloatFeature(Feature):
    def __init__(self, value: float):
        self.value = value

    def vector(self) -> List[float]:
        return [self.value]

    @classmethod
    def from_vector(cls, vector: List[float], **kwargs) -> 'FloatFeature':
        if len(vector) != 1:
            raise ValueError("FloatFeature vector must be of length 1.")
        return cls(vector[0])

    def __str__(self):
        return f"FloatFeature(value={self.value})"

class BinaryFeature(Feature):
    def __init__(self, value: bool):
        self.value = value

    def vector(self) -> List[float]:
        return [1.0 if self.value else 0.0]

    @classmethod
    def from_vector(cls, vector: List[float], **kwargs) -> 'BinaryFeature':
        if len(vector) != 1:
            raise ValueError("BinaryFeature vector must be of length 1.")
        return cls(bool(vector[0]))

    def __str__(self):
        return f"BinaryFeature(value={self.value})"

class OneHotFeature(Feature):
    _encoders: Dict[str, Dict[Any, int]] = {}
    _categories: Dict[str, List[Any]] = {}

    def __init__(self, feature_name: str, categories: List[Any], value: Any):
        """
        Initialize a OneHotFeature.
        :param feature_name: A unique name for the feature.
        :param categories: List of possible categories.
        :param value: The category value.
        """
        self.feature_name = feature_name
        self.categories = categories
        if value not in categories:
            raise ValueError(f"Value '{value}' not in categories {categories}")
        self.value = value

        # Initialize encoder for this feature_name if not already done
        if feature_name not in OneHotFeature._encoders:
            OneHotFeature._encoders[feature_name] = {cat: idx for idx, cat in enumerate(categories)}
            OneHotFeature._categories[feature_name] = categories
            self.num_categories = len(categories)
        else:
            # Ensure the categories match if encoder already exists
            existing_categories = OneHotFeature._categories[feature_name]
            if existing_categories != categories:
                raise ValueError(f"Categories for feature '{feature_name}' do not match existing encoder.")
            self.num_categories = len(categories)

    def vector(self) -> List[float]:
        encoding = [0.0] * self.num_categories
        index = OneHotFeature._encoders[self.feature_name][self.value]
        encoding[index] = 1.0
        return encoding

    @classmethod
    def from_vector(cls, feature_name: str, categories: List[Any], vector: List[float], **kwargs) -> 'OneHotFeature':
        if len(vector) != len(categories):
            raise ValueError("OneHotFeature vector length does not match number of categories.")
        value = categories[np.argmax(vector)]
        return cls(feature_name, categories, value)

    def __str__(self):
        return f"OneHotFeature(feature_name='{self.feature_name}', value='{self.value}')"

class YFeature(Feature):
    def __init__(self, value: float):
        self.value = value

    def vector(self) -> List[float]:
        return [self.value]

    @classmethod
    def from_vector(cls, vector: List[float], **kwargs) -> 'YFeature':
        if len(vector) != 1:
            raise ValueError("YFeature vector must be of length 1.")
        return cls(vector[0])

    def __str__(self):
        return f"YFeature(value={self.value})"

# ----------------------------
# Datastore Classes
# ----------------------------

class Datastore(ABC):
    @abstractmethod
    def add(self, feature_vector: List[float], target: float):
        pass

    @abstractmethod
    def get_all(self) -> Tuple[List[List[float]], List[float]]:
        pass

    @abstractmethod
    def find(self, feature_vector: List[float]) -> Any:
        pass

    @abstractmethod
    def save(self, filepath: str):
        pass

    @abstractmethod
    def load(self, filepath: str):
        pass

class InMemoryDatastore(Datastore):
    def __init__(self):
        self.features = []
        self.targets = []

    def add(self, feature_vector: List[float], target: float):
        self.features.append(feature_vector)
        self.targets.append(target)

    def get_all(self) -> Tuple[List[List[float]], List[float]]:
        return self.features, self.targets

    def find(self, feature_vector: List[float]) -> Any:
        for feat, targ in zip(self.features, self.targets):
            if np.allclose(feat, feature_vector):
                return targ
        return None

    def save(self, filepath: str):
        joblib.dump({'features': self.features, 'targets': self.targets}, filepath)

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.features = data['features']
        self.targets = data['targets']

# ----------------------------
# Model Persistence Classes
# ----------------------------

class ModelSaver:
    @staticmethod
    def save_model(model, filepath: str):
        joblib.dump(model, filepath)

class ModelLoader:
    @staticmethod
    def load_model(filepath: str):
        return joblib.load(filepath)

# ----------------------------
# Lake Class
# ----------------------------

class Lake:
    def __init__(self, 
                 model_type: str = 'regression',
                 batch_size: int = 10,
                 datastore: Datastore = None,
                 feature_classes: List[Feature] = None):
        """
        Initialize the Lake model.
        :param model_type: 'regression' or 'classification'.
        :param batch_size: Number of inserts before training.
        :param datastore: Instance of Datastore. If None, uses InMemoryDatastore.
        :param feature_classes: List of Feature instances defining the structure.
        """
        if feature_classes is None or len(feature_classes) < 2:
            raise ValueError("At least one feature and one target must be provided.")

        self.feature_classes = feature_classes[:-1]  # Input features
        self.target_class = feature_classes[-1]      # Target feature
        self.datastore = datastore if datastore else InMemoryDatastore()
        self.batch_size = batch_size
        self.buffer = []
        self.model_type = model_type

        # Initialize model based on type
        if self.model_type == 'regression':
            self.model = PassiveAggressiveRegressor(max_iter=10000, tol=1e-3)
        elif self.model_type == 'classification':
            self.model = LogisticRegression(max_iter=10000)
        else:
            raise ValueError("Unsupported model_type. Choose 'regression' or 'classification'.")

        self.initialized = False
        self.residuals = []

        # Identify OneHotFeatures to manage their encoders
        self.category_features = {}
        for feature in self.feature_classes:
            if isinstance(feature, OneHotFeature):
                self.category_features[feature.feature_name] = feature.categories

    def add(self, *args: Feature, force_train: bool = False):
        """
        Add a data point to the lake.
        :param args: Feature instances followed by YFeature.
        :param force_train: If True, triggers training immediately.
        """
        if len(args) != len(self.feature_classes) + 1:
            raise ValueError(f"Expected {len(self.feature_classes) + 1} arguments, got {len(args)}")

        features = args[:-1]
        target = args[-1]

        # Convert feature objects to vectors and concatenate
        feature_vector = []
        for feature in features:
            feature_vector.extend(feature.vector())

        self.datastore.add(feature_vector, target.vector()[0])
        self.buffer.append((feature_vector, target.vector()[0]))

        # Decide whether to train
        if len(self.buffer) >= self.batch_size or force_train:
            self._train()

    def _train(self):
        """Train the model on buffered data."""
        if not self.buffer:
            print("Buffer is empty. No training needed.")
            return

        X_batch, y_batch = zip(*self.buffer)
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)

        if not self.initialized:
            if self.model_type == 'regression':
                self.model.fit(X_batch, y_batch)
            elif self.model_type == 'classification':
                # For classification, need to provide the classes on the first call
                classes = np.unique(y_batch)
                self.model.fit(X_batch, y_batch)
            self.initialized = True
        else:
            if self.model_type == 'regression':
                self.model.partial_fit(X_batch, y_batch)
            elif self.model_type == 'classification':
                self.model.partial_fit(X_batch, y_batch)

        # Update residuals for confidence (only for regression)
        if self.model_type == 'regression':
            predictions = self.model.predict(X_batch)
            residuals = y_batch - predictions
            self.residuals.extend(residuals.tolist())

        # Clear the buffer after training
        self.buffer = []

    def predict(self, *args: Feature) -> Tuple[float, float]:
        """
        Predict the target variable based on the provided feature values.
        Returns a tuple of (prediction, confidence).
        """
        if len(args) != len(self.feature_classes):
            raise ValueError(f"Expected {len(self.feature_classes)} arguments, got {len(args)}")

        feature_vector = []
        for feature in args:
            feature_vector.extend(feature.vector())

        X_new = np.array(feature_vector).reshape(1, -1)
        prediction = self.model.predict(X_new)[0]

        # Calculate confidence
        if self.model_type == 'regression' and self.residuals:
            variance = np.var(self.residuals)
            confidence = max(0.0, 1.0 - variance)  # Simple heuristic
        elif self.model_type == 'classification':
            proba = self.model.predict_proba(X_new)
            confidence = np.max(proba)
        else:
            confidence = 0.0  # Default confidence

        return prediction, confidence

    def get_or_predict(self, *args: Feature) -> Tuple[float, float]:
        """
        Retrieve the target value if the exact feature combination exists.
        Otherwise, make a prediction.
        Returns a tuple of (target or prediction, confidence).
        """
        # Convert features to vector
        feature_vector = []
        for feature in args:
            feature_vector.extend(feature.vector())

        # Search for exact match in datastore
        target = self.datastore.find(feature_vector)
        if target is not None:
            confidence = 1.0
            return target, confidence

        # Otherwise, predict
        return self.predict(*args)

    def reconstruct_features(self, vector: List[float]) -> List[Feature]:
        """
        Reconstruct feature instances from a combined feature vector.
        Assumes the order of features as in self.feature_classes.
        """
        features = []
        idx = 0
        for feature in self.feature_classes:
            if isinstance(feature, OneHotFeature):
                num_cats = len(feature.categories)
                sub_vector = vector[idx:idx+num_cats]
                reconstructed = OneHotFeature.from_vector(
                    feature.feature_name,
                    feature.categories,
                    sub_vector
                )
                features.append(reconstructed)
                idx += num_cats
            elif isinstance(feature, IntFeature):
                sub_vector = vector[idx:idx+1]
                reconstructed = IntFeature.from_vector(sub_vector)
                features.append(reconstructed)
                idx += 1
            elif isinstance(feature, FloatFeature):
                sub_vector = vector[idx:idx+1]
                reconstructed = FloatFeature.from_vector(sub_vector)
                features.append(reconstructed)
                idx += 1
            elif isinstance(feature, BinaryFeature):
                sub_vector = vector[idx:idx+1]
                reconstructed = BinaryFeature.from_vector(sub_vector)
                features.append(reconstructed)
                idx += 1
            else:
                raise NotImplementedError(f"Reconstruction not implemented for {type(feature)}")
        return features

    def save(self, model_path: str, datastore_path: str):
        """Save the model and datastore to disk."""
        ModelSaver.save_model(self.model, model_path)
        self.datastore.save(datastore_path)

    def load(self, model_path: str, datastore_path: str):
        """Load the model and datastore from disk."""
        self.model = ModelLoader.load_model(filepath=model_path)
        self.datastore.load(datastore_path)
        self.initialized = True  # Assume model is trained

    def __str__(self):
        feature_str = ', '.join([str(f) for f in self.feature_classes])
        return f"Lake(features=[{feature_str}], target={self.target_class}, model_type='{self.model_type}')"

# ----------------------------
# Example Usage
# ----------------------------

def main():
    # Define feature classes for a polynomial regression (e.g., f(x) = x^3 - x^2 + 5x -1)
    feature_classes = [
        FloatFeature(0.0),  # x
        FloatFeature(0.0),  # x^2
        FloatFeature(0.0),  # x^3
        YFeature(0.0)       # y
    ]

    # Initialize the Lake model for regression with batch_size=10
    lake = Lake(
        model_type='regression',
        batch_size=10,
        feature_classes=feature_classes
    )

    print(lake)

    # Define the true function f(x) = x^3 - x^2 + 5x -1
    def true_function(x):
        return x**3 - x**2 + 5*x - 1

    # Generate data points
    np.random.seed(42)  # For reproducibility
    X_values = np.linspace(-10, 10, 100)
    Y_values = true_function(X_values) + np.random.normal(0, 100, size=X_values.shape)  # Adding noise

    # Lists to store predictions after each batch
    batch_predictions = []
    batch_numbers = []

    # Insert data points into the lake
    for i, (x, y) in enumerate(zip(X_values, Y_values)):
        lake.add(
            FloatFeature(x),
            FloatFeature(x**2),
            FloatFeature(x**3),
            YFeature(y)
        )
        # Optionally, print progress
        if (i+1) % 10 == 0:
            print(f"Inserted {i+1} points.")
            # Make predictions after this batch
            preds = []
            for x_pred in X_values:
                pred, _ = lake.predict(
                    FloatFeature(x_pred),
                    FloatFeature(x_pred**2),
                    FloatFeature(x_pred**3)
                )
                preds.append(pred)
            batch_predictions.append(preds)
            batch_numbers.append(i+1)
            lake.conf

    # After all inserts, force a final training to handle any remaining data in the buffer
    lake._train()
    # Make final prediction
    final_predictions = []
    for x_pred in X_values:
        pred, _ = lake.predict(
            FloatFeature(x_pred),
            FloatFeature(x_pred**2),
            FloatFeature(x_pred**3)
        )
        final_predictions.append(pred)

    # Plotting the results
    plt.figure(figsize=(14, 7))
    plt.scatter(X_values, Y_values, color='blue', label='Data Points', alpha=0.5)
    plt.plot(X_values, true_function(X_values), color='green', label='True Function', linewidth=2)

    # Define a color map
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(batch_predictions)))

    # Plot predictions after each batch
    for idx, preds in enumerate(batch_predictions):
        plt.plot(X_values, preds, color=colors[idx], linestyle='--', label=f'After {batch_numbers[idx]} inserts')

    # Plot the final prediction
    plt.plot(X_values, final_predictions, color='red', label='Final Prediction', linewidth=2)

    plt.title('Lake Predictions vs True Function After Each Batch of Inserts')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Save the model and datastore
    lake.save('lake_model.pkl', 'lake_datastore.pkl')
    print("Model and datastore saved.")

    # Load the model and datastore into a new Lake instance
    new_lake = Lake(
        model_type='regression',
        batch_size=10,
        feature_classes=feature_classes
    )
    new_lake.load('lake_model.pkl', 'lake_datastore.pkl')
    print("Loaded Lake:", new_lake)

    # Make a prediction using the loaded model
    sample_x = 5.0
    sample_x2 = sample_x**2
    sample_x3 = sample_x**3
    pred, conf = new_lake.predict(FloatFeature(sample_x),
                                  FloatFeature(sample_x2),
                                  FloatFeature(sample_x3))
    print(f"Sample Prediction for X={sample_x}: Y={pred:.2f}, Confidence={conf:.2f}")

    # Demonstrate get_or_predict
    # Existing data point
    existing_drop = new_lake.get_or_predict(
        FloatFeature(5.0),
        FloatFeature(25.0),
        FloatFeature(125.0)
    )
    print(f"Retrieved Y for existing X=5.0: {existing_drop[0]:.2f}, Confidence: {existing_drop[1]:.2f}")

    # Non-existing data point
    non_existing_drop = new_lake.get_or_predict(
        FloatFeature(6.0),
        FloatFeature(36.0),
        FloatFeature(216.0)
    )
    print(f"Predicted Y for X=6.0: {non_existing_drop[0]:.2f}, Confidence: {non_existing_drop[1]:.2f}")

    # Reconstruct features from a vector
    sample_vector = new_lake.datastore.features[0]  # Take the first drop's feature vector
    reconstructed_features = new_lake.reconstruct_features(sample_vector)
    print("Reconstructed Features:")
    for feature in reconstructed_features:
        print(f" - {feature}")

    # Display all drops
    print("\nAll Drops:")
    for idx, (vec, target) in enumerate(new_lake.datastore.get_all()[0:5], 1):  # Display first 5 for brevity
        print(f"Drop {idx}: Features={vec}, Target={target}")

if __name__ == "__main__":
    main()
