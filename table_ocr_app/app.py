import streamlit as st
from table_extractor import TableOCRExtractor

st.set_page_config(layout="centered")
st.title("🧾 نظام استخراج الجدول من صورة واحدة")

uploaded_file = st.file_uploader("📤 ارفع صورة الجدول", type=["png", "jpg", "jpeg"])

if uploaded_file:
    app = TableOCRExtractor()
    app.load_image(uploaded_file)
    app.crop_image()

    if st.button("✂️ قص واستخراج"):
        cells = app.split_rows_and_columns()
        app.run_ocr(cells)
        app.display_results()
