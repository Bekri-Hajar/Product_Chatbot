import streamlit as st
import requests

# Frontend application
def main():
    st.title("Nature Leaves Chatbot")
    
    # Ask a question
    user_query = st.text_input("Ask a question about skin care:")
    
    if user_query:
        # Send the question to the Flask API
        response = requests.post('http://127.0.0.1:5000/ask', json={'question': user_query})
        
        if response.status_code == 200:
            answer = response.json().get('answer')
            st.write(f"Chatbot: {answer}")
        else:
            st.write("Sorry, there was an error processing your question.")

if __name__ == "__main__":
    main()
