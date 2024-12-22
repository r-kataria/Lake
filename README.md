# Lake: Adaptive Learning Data Structure

**Lake** is an adaptive learning data structure designed to seamlessly integrate data storage, feature representation, and machine learning. It dynamically learns and updates regression or classification models as new data points (or "drops") are added. With a customizable feature structure, efficient in-memory datastore, and support for both batch and incremental training, Lake offers a robust solution for real-time predictive tasks.

Key features include feature reconstruction, on-demand prediction with confidence scoring, and seamless model persistence. Ideal for applications that require continuous learning with minimal overhead, **Lake** is a step towards developing a full-fledged AutoML library focused on **incremental ML**, with future plans to support database integrations, distributed backend nodes for training, and more. Furthermore, a Relational-like API or a Python Dictionary API is planned.

```python
lake.add(FloatFeature(5.0), FloatFeature(25.0), FloatFeature(125.0), YFeature(129.0))  # Add data
prediction, confidence = lake.predict(FloatFeature(6.0), FloatFeature(36.0), FloatFeature(216.0))  # Predict
```