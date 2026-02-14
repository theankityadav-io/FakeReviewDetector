import joblib
import warnings
warnings.filterwarnings('ignore')

print("🔍 Fake Review Detector - Test Mode")
print("=" * 50)

# Load the trained model
print("\n📂 Loading trained model...")
try:
    model = joblib.load('fake_review_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("✅ Model loaded successfully!\n")
except Exception as e:
    print(f"❌ Error: {e}")
    print("⚠️  Please run 'train_model.py' first!")
    exit()

# Function to predict review
def predict_review(review_text):
    """Predict if a review is fake or genuine"""
    # Convert text to TF-IDF
    review_tfidf = vectorizer.transform([review_text])
    
    # Predict
    prediction = model.predict(review_tfidf)[0]
    confidence = model.predict_proba(review_tfidf)[0]
    
    # Results
    label = "🚨 FAKE" if prediction == 1 else "✅ GENUINE"
    conf_percentage = max(confidence) * 100
    
    return label, conf_percentage

# Test with sample reviews
print("🧪 Testing with sample reviews:\n")

test_reviews = [
    "Amazing product!!!! Best buy ever! 5 stars!!!!",
    "The product works well for my needs. Delivery took about a week.",
    "Perfect! Perfect! Perfect! Buy it now!!!",
    "Good quality overall. The size is slightly smaller than expected.",
    "Outstanding! Everyone should buy this! No complaints!",
    "It's okay for the price. Not the best quality but gets the job done."
]

for i, review in enumerate(test_reviews, 1):
    label, confidence = predict_review(review)
    print(f"Review {i}:")
    print(f"   Text: {review[:60]}...")
    print(f"   Prediction: {label}")
    print(f"   Confidence: {confidence:.2f}%")
    print()

# Interactive mode
print("=" * 50)
print("🎮 INTERACTIVE MODE - Test your own reviews!")
print("=" * 50)
print("Type 'quit' to exit\n")

while True:
    user_review = input("Enter a review to test: ")
    
    if user_review.lower() == 'quit':
        print("\n👋 Thanks for using Fake Review Detector!")
        break
    
    if user_review.strip() == "":
        print("⚠️  Please enter a valid review!\n")
        continue
    
    label, confidence = predict_review(user_review)
    print(f"\n   Prediction: {label}")
    print(f"   Confidence: {confidence:.2f}%\n")