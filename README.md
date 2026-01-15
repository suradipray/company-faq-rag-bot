# Powershell commands:
# Go to app directory
cd company-faq-rag-bot
# activate python env
.venv\Scripts\activate
# run FAST API
uvicorn src.api:app --reload
# launch streamlit app url
python -m streamlit run src/app_streamlit.py
