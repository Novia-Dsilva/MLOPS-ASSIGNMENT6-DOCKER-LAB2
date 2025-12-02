
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import json

def load_pima_diabetes():
    """Load Pima Indians Diabetes Dataset"""
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    
    try:
        df = pd.read_csv(url, names=column_names)
    except:
        print("Creating sample dataset...")
        np.random.seed(42)
        df = pd.DataFrame({
            'Pregnancies': np.random.randint(0, 17, 768),
            'Glucose': np.random.randint(0, 200, 768),
            'BloodPressure': np.random.randint(0, 122, 768),
            'SkinThickness': np.random.randint(0, 100, 768),
            'Insulin': np.random.randint(0, 846, 768),
            'BMI': np.random.uniform(0, 67.1, 768),
            'DiabetesPedigreeFunction': np.random.uniform(0.078, 2.42, 768),
            'Age': np.random.randint(21, 81, 768),
            'Outcome': np.random.randint(0, 2, 768)
        })
    
    return df

if __name__ == '__main__':
    print("=" * 70)
    print("         DIABETES PREDICTION - MULTI-MODEL TRAINING")
    print("=" * 70)
    
    # Load dataset
    print("\n[Step 1/5] Loading Pima Indians Diabetes Dataset...")
    df = load_pima_diabetes()
    print(f"  ✓ Dataset loaded: {df.shape[0]} patients")
    print(f"  ✓ Diabetic cases: {df['Outcome'].sum()} ({df['Outcome'].sum()/len(df)*100:.1f}%)")
    
    # Separate features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split the dataset
    print("\n[Step 2/5] Splitting dataset (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  ✓ Training samples: {len(X_train)}")
    print(f"  ✓ Testing samples: {len(X_test)}")
    
    # Standardize features
    print("\n[Step 3/5] Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.pkl')
    print("  ✓ Feature scaling completed")
    
    # Store results
    model_results = {}
    trained_models = {}
    
    print("\n[Step 4/5] Training and evaluating multiple models...")
    print("-" * 70)
    
    # Model 1: Neural Network
    print("\n  [1/5] Training Neural Network...")
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, input_shape=(8,), activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    nn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, 
                 validation_split=0.2, verbose=0)
    
    nn_pred = (nn_model.predict(X_test_scaled, verbose=0) > 0.5).astype(int)
    nn_accuracy = accuracy_score(y_test, nn_pred)
    model_results['Neural Network'] = nn_accuracy
    trained_models['Neural Network'] = nn_model
    print(f"        Accuracy: {nn_accuracy:.4f} ({nn_accuracy*100:.2f}%)")
    
    # Model 2: Random Forest
    print("\n  [2/5] Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                     random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    rf_accuracy = rf_model.score(X_test_scaled, y_test)
    model_results['Random Forest'] = rf_accuracy
    trained_models['Random Forest'] = rf_model
    print(f"        Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
    
    # Model 3: Gradient Boosting
    print("\n  [3/5] Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                         max_depth=3, random_state=42)
    gb_model.fit(X_train_scaled, y_train)
    gb_accuracy = gb_model.score(X_test_scaled, y_test)
    model_results['Gradient Boosting'] = gb_accuracy
    trained_models['Gradient Boosting'] = gb_model
    print(f"        Accuracy: {gb_accuracy:.4f} ({gb_accuracy*100:.2f}%)")
    
    # Model 4: Support Vector Machine
    print("\n  [4/5] Training Support Vector Machine...")
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    svm_accuracy = svm_model.score(X_test_scaled, y_test)
    model_results['SVM'] = svm_accuracy
    trained_models['SVM'] = svm_model
    print(f"        Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
    
    # Model 5: Logistic Regression
    print("\n  [5/5] Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_accuracy = lr_model.score(X_test_scaled, y_test)
    model_results['Logistic Regression'] = lr_accuracy
    trained_models['Logistic Regression'] = lr_model
    print(f"        Accuracy: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")
    
    # Select best model
    print("\n[Step 5/5] Selecting best performing model...")
    print("-" * 70)
    
    sorted_models = sorted(model_results.items(), key=lambda x: x[1], reverse=True)
    
    print("\nModel Performance Ranking:")
    for i, (model_name, accuracy) in enumerate(sorted_models, 1):
        marker = "🏆" if i == 1 else f"  {i}."
        print(f"{marker} {model_name:.<30} {accuracy*100:.2f}%")
    
    # Save the best model
    best_model_name = sorted_models[0][0]
    best_model = trained_models[best_model_name]
    best_accuracy = sorted_models[0][1]
    
    print(f"\n{'='*70}")
    print(f"🏆 BEST MODEL SELECTED: {best_model_name}")
    print(f"   Accuracy: {best_accuracy*100:.2f}%")
    print(f"{'='*70}")
    
    # Save best model
    if best_model_name == 'Neural Network':
        best_model.save('best_model.keras')
        model_type = 'neural_network'
    else:
        joblib.dump(best_model, 'best_model.pkl')
        model_type = 'sklearn'
    
    # Save model metadata
    model_info = {
        'best_model_name': best_model_name,
        'best_model_accuracy': best_accuracy,
        'model_type': model_type,
        'all_models': model_results
    }
    
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\nBest model saved successfully!")
    print(f"   - Model file: {'best_model.keras' if model_type == 'neural_network' else 'best_model.pkl'}")
    print(f"   - Scaler: scaler.pkl")
    print(f"   - Metadata: model_info.json")
    print("\n" + "="*70)