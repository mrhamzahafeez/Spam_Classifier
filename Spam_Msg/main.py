import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Streamlit app
st.title("Message Classifier")
st.write("This app classifies messages as **Spam** or **Not Spam**.")

# Input message from user
user_message = st.text_area("Enter your message:", height=150)

# Button to classify the message
if st.button("Classify"):
    if user_message.strip() == "":
        st.warning("Please enter a message to classify!")
    else:
        # Transform the input text using the loaded vectorizer
        input_vector = vectorizer.transform([user_message])
        
        # Predict using the loaded model
        prediction = model.predict(input_vector)[0]
        probability = model.predict_proba(input_vector)[0]

        # Display the result
        if prediction == 1:
            st.error(f"**Spam** detected with confidence: {probability[1]:.2f}")
        else:
            st.success(f"**Not Spam** detected with confidence: {probability[0]:.2f}")
