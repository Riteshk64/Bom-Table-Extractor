import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper
import fitz  # PyMuPDF
import os
from src.TableOCRExtractor import TableOCRExtractor

# --------------------------- Page Setup ---------------------------
st.set_page_config(page_title="Table OCR", layout="wide")
st.title("📄 Table OCR Tool")

# --------------------------- Helpers ---------------------------
def pdf_page_to_image(pdf_bytes: bytes, page_number: int = 0, zoom: int = 2) -> Image.Image:
    """Render a PDF page to a PIL image using PyMuPDF (no external binaries)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_number)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    return Image.open(io.BytesIO(pix.tobytes("png")))

def init_state():
    ss = st.session_state
    ss.setdefault("df", None)
    ss.setdefault("download_file_name", None)
    ss.setdefault("temp_file_name", ss["download_file_name"])

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure non-empty, unique headers for st.data_editor."""
    df.columns = [f"Column_{i}" if not (isinstance(c, str) and c.strip()) else str(c) for i, c in enumerate(df.columns)]
    seen = {}
    new_cols = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols
    return df

def default_bom_df(rows: int = 5) -> pd.DataFrame:
    cols = ["Part no", "Name", "Description", "Quantity", "Remarks"]
    df = pd.DataFrame(columns=cols)
    for _ in range(rows):
        df.loc[len(df)] = [""] * len(cols)
    return df

def sanitize_filename(name: str) -> str:
    name = (name or "").strip()
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-", " ") else "_" for ch in name) or "extracted_data"
    return safe.replace(" ", "_")

init_state()

# --------------------------- 1) Upload ---------------------------
st.header("1️⃣ Upload Drawing or PDF")
uploaded_file = st.file_uploader("Choose an image or PDF", type=["jpg", "jpeg", "png", "pdf"])

img = None
if uploaded_file:
    if uploaded_file.name.lower().endswith(".pdf"):
        pdf_bytes = uploaded_file.read()
        with st.container():
            left, right = st.columns([1, 4])
            with left:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                page_num = st.number_input("Page", min_value=1, max_value=len(doc), value=1, step=1)
        img = pdf_page_to_image(pdf_bytes, page_number=page_num - 1)
    else:
        img = Image.open(uploaded_file).convert("RGB")
    base_name = os.path.splitext(uploaded_file.name)[0]
    safe_name = sanitize_filename(base_name)
    st.session_state.download_file_name = st.session_state.download_file_name or safe_name

# --------------------------- 2) Crop ---------------------------
if img is not None:
    st.header("2️⃣ Crop Table")
    with st.container():
        crop_col, preview_col = st.columns([2, 1])
        with crop_col:
            cropped_img = st_cropper(img, realtime_update=True, box_color='red', aspect_ratio=None)
        with preview_col:
            st.caption("Cropped preview")
            st.image(cropped_img, use_container_width=True)

    # ----------------------- 3) OCR -----------------------
    st.header("3️⃣ Extract Table")
    if st.button("▶️ Run OCR", use_container_width=True):
        try:
            cropped_np = np.array(cropped_img)[:, :, ::-1]

            extractor = TableOCRExtractor(
                cropped_np,
                tesseract_path=r"C:\Users\121807\Documents\tesseract.exe"
            )
            df = extractor.execute()

            if df is None or df.empty or df.isna().all().all():
                raise ValueError("OCR returned empty table")

            df = sanitize_columns(df)
            st.success("Table ready! Continue to edit & download below.")

        except Exception as e:
            st.error(f"⚠️ OCR failed. Loaded a default 4x4 table you can edit.")
            df = pd.DataFrame([[""]*4 for _ in range(4)], columns=[f"Column_{i+1}" for i in range(4)])

        st.session_state.df = df

# --------------------------- 4) Edit ---------------------------
if st.session_state.df is not None:
    st.header("4️⃣ Review & Edit")

    st.markdown(
        """
        <style>
        /* Align st.button vertically with st.text_input / st.number_input */
        div.stButton > button:first-child {
            margin-top: 1.75em;   /* adjust between 1.5em–2em if needed */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.session_state.df = st.data_editor(
        st.session_state.df,
        num_rows="dynamic",
        use_container_width=True,
        key="table_editor"
    )

    with st.expander("⚙️ Table Editing Tools", expanded=False):
        st.markdown("#### ➕ Insert Row")
        r1, r2, r3 = st.columns([2, 2, 1])
        with r1:
            insert_idx = st.number_input(
                "Insert at row index",
                min_value=0,
                max_value=len(st.session_state.df),
                value=len(st.session_state.df),
                step=1,
                key="insert_idx"
            )
        with r2:
            st.caption("Row will be inserted above this index")
        with r3:
            if st.button("Add Row", use_container_width=True, key="add_row_btn"):
                empty_row = {col: "" for col in st.session_state.df.columns}
                st.session_state.df = pd.concat(
                    [st.session_state.df.iloc[:insert_idx], pd.DataFrame([empty_row]), st.session_state.df.iloc[insert_idx:]],
                    ignore_index=True
                )
                st.rerun()

        st.divider()

        st.markdown("#### 🛠️ Column Tools")

        rc1, rc2, rc3 = st.columns([2, 3, 1])
        with rc1:
            cols_list = list(st.session_state.df.columns)
            rename_idx = st.selectbox("Column", options=list(range(len(cols_list))),
                                      format_func=lambda i: cols_list[i], key="rename_idx")
        with rc2:
            new_name = st.text_input("New name", value=cols_list[rename_idx], key="rename_val")
        with rc3:
            if st.button("Rename", use_container_width=True, key="rename_btn"):
                if new_name.strip() and new_name not in cols_list:
                    cols_list[rename_idx] = new_name.strip()
                    st.session_state.df.columns = cols_list
                    st.rerun()

        st.divider()

        ac1, ac2, ac3 = st.columns([3, 2, 1])
        with ac1:
            add_name = st.text_input("New column name", key="add_col_name", placeholder="e.g., Notes")
        with ac2:
            insert_at = st.number_input("Insert at index", min_value=0, max_value=len(cols_list),
                                        value=len(cols_list), step=1, key="add_idx")
        with ac3:
            if st.button("Add", use_container_width=True, key="add_col_btn"):
                name = add_name.strip() or f"Column_{len(cols_list)+1}"
                if name not in cols_list:
                    st.session_state.df.insert(int(insert_at), name, "")
                    st.rerun()

        st.divider()

        dc1, dc2 = st.columns([4, 1])
        with dc1:
            drop_choice = st.selectbox("Remove column",
                                       options=["None"] + cols_list,
                                       key="drop_choice")
        with dc2:
            if st.button("Remove", use_container_width=True, key="drop_col_btn"):
                if drop_choice != "None":
                    st.session_state.df.drop(columns=[drop_choice], inplace=True)
                    st.rerun() 

# --------------------------- 5) Download ---------------------------
if st.session_state.df is not None:
    st.header("5️⃣ Download")

    fn1, fn2 = st.columns([3, 1])
    with fn1:
        new_name = st.text_input(
            "File name (without extension)",
            value=st.session_state.download_file_name,
            key="file_name_input"
        )
        st.session_state.download_file_name = sanitize_filename(new_name)
    with fn2:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.caption(" ")

    csv_data = st.session_state.df.to_csv(index=False)
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        st.session_state.df.to_excel(writer, index=False)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "📥 Download CSV",
            csv_data,
            file_name=f"{st.session_state.download_file_name}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with d2:
        st.download_button(
            "📥 Download Excel",
            excel_buf.getvalue(),
            file_name=f"{st.session_state.download_file_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
