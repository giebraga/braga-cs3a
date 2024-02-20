import streamlit as st


st.header("XOR Cipher")

st.text_area("Angelika:")

st.text_input("Key:")

plaintext = st.text_area("Plain Text:")

key =st.text_input("Key")

if st.button("Submit"):
    st.write(plaintext)