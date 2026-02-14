# ============================================
# FAKE REVIEW DETECTOR - TRAINING SCRIPT
# ============================================
# Tumhe bas ye file run karni hai!

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting Fake Review Detector Training...")
print("=" * 50)

# STEP 1: Load Dataset
print("\n📊 Step 1: Loading dataset...")
try:
    df = pd.read_csv('reviews_dataset.csv')
    print(f"✅ Dataset loaded successfully!")
    print(f"   Total reviews: {len(df)}")
    print(f"   Fake reviews: {sum(df['label'] == 1)}")
    print(f"   Genuine reviews: {sum(df['label'] == 0)}")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("⚠️  Make sure 'reviews_dataset.csv' is in the same folder!")
    exit()

# STEP 2: Prepare Data
print("\n🔧 Step 2: Preparing data...")
X = df['review_text']  # Reviews
y = df['label']        # Labels (0 = Genuine, 1 = Fake)

# STEP 3: Split into Training and Testing
print("\n✂️  Step 3: Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# STEP 4: Convert Text to Numbers (TF-IDF)
print("\n🔢 Step 4: Converting text to numbers (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("✅ Text conversion complete!")

# STEP 5: Train the Model
print("\n🧠 Step 5: Training the AI model...")
print("   (This may take 10-20 seconds...)")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)
print("✅ Model trained successfully!")

# STEP 6: Test the Model
print("\n🎯 Step 6: Testing model accuracy...")
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'=' * 50}")
print(f"🎉 MODEL TRAINING COMPLETE!")
print(f"{'=' * 50}")
print(f"📊 Accuracy: {accuracy * 100:.2f}%")
print(f"\n📈 Detailed Report:")
print(classification_report(y_test, y_pred, target_names=['Genuine', 'Fake']))

# STEP 7: Save the Model
print("\n💾 Step 7: Saving model and vectorizer...")
joblib.dump(model, 'fake_review_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("✅ Model saved as: fake_review_model.pkl")
print("✅ Vectorizer saved as: tfidf_vectorizer.pkl")

print("\n" + "=" * 50)
print("🎊 ALL DONE! Your AI model is ready to use!")
print("=" * 50)
print("\n💡 Next step: Run 'test_model.py' to test your model!")