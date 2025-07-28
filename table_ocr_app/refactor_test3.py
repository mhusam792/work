# import streamlit as st
# import cv2
# from PIL import Image
# import numpy as np
# from streamlit_extras.image_selector import image_selector
# from paddleocr import PaddleOCR

# ocr = PaddleOCR(use_angle_cls=True, lang='en')

# st.set_page_config(layout="wide")
# st.title("📐 نظام تحديد الأعمدة من صورة واحدة")

# uploaded_image = st.file_uploader("📷 اختر صورة المخطط الهندسي", type=["jpg", "png", "jpeg"])

# # 🎨 ألوان مختلفة لكل عمود
# colors = [
#     (255, 0, 0),      # أحمر
#     (0, 255, 0),      # أخضر
#     (0, 0, 255),      # أزرق
#     (255, 255, 0),    # أصفر
#     (255, 0, 255),    # بنفسجي
#     (0, 255, 255),    # سماوي
#     (128, 0, 128)     # بنفسجي غامق
# ]

# def draw_all_boxes(image_np, boxes):
#     for idx, box in enumerate(boxes):
#         color = colors[idx % len(colors)]
#         cv2.rectangle(
#             image_np,
#             (box['x_min'], box['y_min']),
#             (box['x_max'], box['y_max']),
#             color=color,
#             thickness=3
#         )
#     return image_np

# def get_box_coords(selection):
#     try:
#         if selection and selection.get("selection") and "box" in selection["selection"]:
#             boxes = selection["selection"]["box"]
#             if boxes:
#                 box = boxes[0]
#                 x_vals = box["x"]
#                 y_vals = box["y"]
#                 return {
#                     'x_min': int(min(x_vals)),
#                     'x_max': int(max(x_vals)),
#                     'y_min': int(min(y_vals)),
#                     'y_max': int(max(y_vals)),
#                 }
#     except Exception as e:
#         st.error(f"❌ خطأ أثناء استخراج الإحداثيات: {e}")
#     return None


# def extract_cells_with_gaps(ocr_result, max_rows, gap_threshold=25):
#     """
#     ocr_result: نتيجة الـ OCR (من ocr.predict)
#     max_rows: عدد الصفوف المتوقعة
#     gap_threshold: المسافة بين كل سطر والتاني لتقدير الخلايا الفاضية
#     """
#     texts = ocr_result[0]['rec_texts']
#     boxes = ocr_result[0]['rec_boxes']

#     lines_with_y = []
#     for text, box in zip(texts, boxes):
#         # تأكد إن box فيه 4 نقاط وكل نقطة فيها x,y
#         try:
#             if isinstance(box, (list, np.ndarray)) and len(box) == 4:
#                 y_center = (box[0][1] + box[2][1]) / 2
#                 lines_with_y.append((text, y_center))
#         except Exception as e:
#             continue  # لو حصل أي خطأ في box، تجاهله

#     # الترتيب الرأسي من أعلى لأسفل
#     lines_with_y.sort(key=lambda x: x[1])

#     rows = []
#     prev_y = None
#     for text, y in lines_with_y:
#         if prev_y is not None and abs(y - prev_y) > gap_threshold:
#             rows.append(None)  # صف فاضي
#         rows.append(text.strip())
#         prev_y = y

#     # استكمال عدد الصفوف الناقصة
#     while len(rows) < max_rows:
#         rows.append(None)

#     return rows[:max_rows]
# # ------------------------------
# # 🚀 التطبيق
# # ------------------------------
# if uploaded_image is not None:
#     image_pil = Image.open(uploaded_image).convert("RGB")
#     image_np_original = np.array(image_pil)
#     image_np_display = image_np_original.copy()

#     # حافظ على الأعمدة المحددة
#     if "all_boxes" not in st.session_state:
#         st.session_state.all_boxes = []
#         st.session_state.selected_regions = []
#         st.session_state.column_index = 0

#     if st.session_state.column_index < 7:
#         st.markdown(f"### ✂️ حدد العمود رقم {st.session_state.column_index + 1}")
#         image_np_with_boxes = draw_all_boxes(image_np_display.copy(), st.session_state.all_boxes)
#         image_with_boxes_pil = Image.fromarray(image_np_with_boxes)

#         # الصورة الوحيدة التي يحدد منها كل الأعمدة
#         selection = image_selector(image_with_boxes_pil, key=f"selector_{st.session_state.column_index}")
#         box_coords = get_box_coords(selection)

#         if box_coords:
#             st.session_state.all_boxes.append(box_coords)

#             # قص الجزء المحدد
#             x_min, x_max = box_coords['x_min'], box_coords['x_max']
#             y_min, y_max = box_coords['y_min'], box_coords['y_max']
#             cropped_np = image_np_original[y_min:y_max, x_min:x_max]
#             cropped_pil = Image.fromarray(cropped_np)
#             st.session_state.selected_regions.append(cropped_pil)

#             st.session_state.column_index += 1
#             st.rerun()  # 🔁 يعيد تحميل الصفحة بتحديث العمود التالي
#         else:
#             st.image(image_with_boxes_pil, caption="🖼️ الصورة الحالية", use_container_width=True)
#     else:
#         st.success("✅ تم تحديد الأعمدة السبعة بنجاح!")

#         # عرض الصورة النهائية بكل التحديدات
#         final_image_np = draw_all_boxes(image_np_original.copy(), st.session_state.all_boxes)
#         st.image(final_image_np, caption="📌 كل الأعمدة السبعة", use_container_width=True)

#         # تحليل النصوص
#         for idx, region in enumerate(st.session_state.selected_regions):
#             st.markdown(f"### 📍 العمود {idx + 1}")
#             st.image(region, caption=f"📸 الجزء {idx + 1}", use_container_width=True)

#             img_bgr = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2BGR)
#             result = ocr.predict(img_bgr)

#             # عدد الصفوف (يدخله المستخدم)
#             num_rows = st.number_input(f"🧮 كم عدد الصفوف في العمود {idx + 1}؟", min_value=1, max_value=100, step=1, key=f"rows_{idx}")

#             # استخراج النصوص وتقدير الصفوف
#             cells = extract_cells_with_gaps(result, num_rows)

#             # عرض النتيجة
#             st.markdown("### 🗂️ محتويات العمود مع صفوف فارغة:")
#             for i, cell in enumerate(cells, 1):
#                 display = cell if cell else "❌ null"
#                 st.write(f"صف {i}: {display}")


#         # زر إعادة التحديد
#         if st.button("🔄 إعادة التحديد من البداية"):
#             st.session_state.all_boxes = []
#             st.session_state.selected_regions = []
#             st.session_state.column_index = 0
#             st.rerun()


# 

import streamlit as st
import cv2
from PIL import Image
import numpy as np
from streamlit_extras.image_selector import image_selector

st.set_page_config(layout="wide")
st.title("📐 نظام تحديد الأعمدة والخلية من صورة واحدة")

uploaded_image = st.file_uploader("📷 اختر صورة المخطط الهندسي", type=["jpg", "png", "jpeg"])

colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 128)
]

def draw_all_boxes(image_np, boxes, extra_box=None):
    for idx, box in enumerate(boxes):
        color = colors[idx % len(colors)]
        cv2.rectangle(
            image_np,
            (box['x_min'], box['y_min']),
            (box['x_max'], box['y_max']),
            color=color,
            thickness=3
        )
    if extra_box:
        cv2.rectangle(
            image_np,
            (extra_box['x_min'], extra_box['y_min']),
            (extra_box['x_max'], extra_box['y_max']),
            (0, 0, 0),  # أسود
            thickness=3
        )
    return image_np

def get_box_coords(selection):
    try:
        if selection and selection.get("selection") and "box" in selection["selection"]:
            boxes = selection["selection"]["box"]
            if boxes:
                box = boxes[0]
                x_vals = box["x"]
                y_vals = box["y"]
                return {
                    'x_min': int(min(x_vals)),
                    'x_max': int(max(x_vals)),
                    'y_min': int(min(y_vals)),
                    'y_max': int(max(y_vals)),
                }
    except Exception as e:
        st.error(f"❌ خطأ أثناء استخراج الإحداثيات: {e}")
    return None

if uploaded_image is not None:
    image_pil = Image.open(uploaded_image).convert("RGB")
    image_np_original = np.array(image_pil)
    image_np_display = image_np_original.copy()

    # حالة البداية
    if "all_boxes" not in st.session_state:
        st.session_state.all_boxes = []
        st.session_state.column_index = 0
        st.session_state.cell_box = None
        st.session_state.step = "columns"

    if st.session_state.step == "columns":
        if st.session_state.column_index < 7:
            st.markdown(f"### ✂️ حدد العمود رقم {st.session_state.column_index + 1}")

            image_with_boxes = draw_all_boxes(image_np_display.copy(), st.session_state.all_boxes)
            image_with_boxes_pil = Image.fromarray(image_with_boxes)

            selection = image_selector(image_with_boxes_pil, key=f"selector_col_{st.session_state.column_index}")
            box = get_box_coords(selection)

            if box:
                st.session_state.all_boxes.append(box)
                st.session_state.column_index += 1
                st.rerun()
            else:
                st.image(image_with_boxes_pil, caption="🖼️ الصورة الحالية", use_container_width=True)
        else:
            st.session_state.step = "cell"
            st.rerun()

    elif st.session_state.step == "cell":
        st.markdown("### 🟩 حدد خلية واحدة داخل الصورة (المربع التامن)")
        image_with_boxes = draw_all_boxes(image_np_display.copy(), st.session_state.all_boxes)
        image_with_boxes_pil = Image.fromarray(image_with_boxes)
        selection = image_selector(image_with_boxes_pil, key="selector_cell")

        box = get_box_coords(selection)
        if box:
            st.session_state.cell_box = box
            st.session_state.step = "done"
            st.rerun()
        else:
            st.image(image_with_boxes_pil, caption="🖼️ الصورة بعد الأعمدة", use_container_width=True)

    elif st.session_state.step == "done":
        st.success("✅ تم تحديد الأعمدة والخلية بنجاح!")
        final_image_np = draw_all_boxes(image_np_original.copy(), st.session_state.all_boxes, st.session_state.cell_box)
        st.image(final_image_np, caption="📌 الشكل النهائي", use_container_width=True)

        with st.container():
            st.markdown("## 📋 الإحداثيات النهائية")
            for idx, box in enumerate(st.session_state.all_boxes, start=1):
                st.write(f"📦 العمود {idx}: {box}")
            st.write(f"🟩 الخلية (المربع التامن): {st.session_state.cell_box}")

        if st.button("🔄 إعادة التحديد من البداية"):
            st.session_state.all_boxes = []
            st.session_state.column_index = 0
            st.session_state.cell_box = None
            st.session_state.step = "columns"
            st.rerun()
