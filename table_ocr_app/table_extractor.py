from PIL import Image
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
import streamlit as st
from streamlit_cropper import st_cropper


class TableOCRExtractor:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        self.image = None
        self.cropped_image = None
        self.result_df = None

        # إعداد الأعمدة وصفوف الجدول
        self.row_height = 42
        self.num_rows = 9
        self.cols = [
            [14, 344],
            [344, 473],
            [473, 630],
            [630, 848],
            [848, 1006],
            [1006, 1178],
            [1178, 1328]
        ]

    def load_image(self, uploaded_file):
        self.image = Image.open(uploaded_file).convert("RGB")

    def crop_image(self):
        st.subheader("🟩 حدد منطقة الجدول داخل الصورة")
        self.cropped_image = st_cropper(
            self.image,
            realtime_update=False,
            box_color='#00FF00',
            aspect_ratio=None,
            stroke_width=1
        )
        st.image(self.cropped_image, caption="📐 الجزء المحدد من الصورة")

    def split_rows_and_columns(self):
        img_np = np.array(self.cropped_image)
        rows = []
        y1, y2 = 0, self.row_height

        for _ in range(self.num_rows):
            row = img_np[y1:y2, :]
            rows.append(row)
            y1 = y2
            y2 += self.row_height

        all_cells = []
        for row in rows:
            current_row = []
            for col_start, col_end in self.cols:
                current_row.append(row[:, col_start:col_end])
            all_cells.append(current_row)

        return all_cells

    def run_ocr(self, cells):
        text_data = []
        for row in cells:
            row_result = []
            for cell in row:
                result = self.ocr.predict(cell)
                texts = result[0]['rec_texts']
                if len(texts) == 0:
                    row_result.append(np.nan)
                else:
                    row_result.append(' '.join(texts))
            text_data.append(row_result)
        self.result_df = pd.DataFrame(text_data)

    def display_results(self):
        st.subheader("📋 النصوص المستخرجة")
        st.dataframe(self.result_df)

        csv = self.result_df.to_csv(index=False, header=False)
        st.download_button(
            label="📥 تحميل CSV",
            data=csv,
            file_name="ocr_results_cleaned.csv",
            mime="text/csv"
        )
