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
    page = doc.load_page(page_number)  # 0-based
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    return Image.open(io.BytesIO(pix.tobytes("png")))

def init_state():
    ss = st.session_state
    ss.setdefault("df", None)
    ss.setdefault("download_file_name", None)
    ss.setdefault("temp_file_name", ss["download_file_name"])

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure non-empty, unique headers for st.data_editor."""
    # Replace blank/NaN headers with placeholders
    df.columns = [f"Column_{i}" if not (isinstance(c, str) and c.strip()) else str(c) for i, c in enumerate(df.columns)]
    # Ensure uniqueness
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
        # Quick page chooser laid out compactly
        with st.container():
            left, right = st.columns([1, 4])
            with left:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                page_num = st.number_input("Page", min_value=1, max_value=len(doc), value=1, step=1)
            # Render selected page
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
        # Convert PIL to OpenCV BGR for your OCR class
        cropped_np = np.array(cropped_img)[:, :, ::-1]

        extractor = TableOCRExtractor(
            cropped_np,
            tesseract_path=r"C:\Users\121807\Documents\tesseract.exe"  # your path
        )
        df = extractor.execute()

        # Fallback if OCR fails / only NaNs
        if df is None or df.empty or df.isna().all().all():
            st.info("No usable table detected by OCR. Loaded a default BOM template you can edit.")
            df = default_bom_df()
        else:
            df = sanitize_columns(df)

        st.session_state.df = df
        st.success("Table ready! Continue to edit & download below.")

# --------------------------- 4) Edit ---------------------------
if st.session_state.df is not None:
    st.header("4️⃣ Review & Edit")

    # Editable grid
    st.session_state.df = st.data_editor(
        st.session_state.df,
        num_rows="dynamic",
        use_container_width=True,
        key="table_editor"
    )

    # Insert row at index (simple & clear)
    st.subheader("➕ Insert Row")
    c1, c2 = st.columns([2, 1])
    with c1:
        insert_idx = st.number_input(
            "Insert at row index",
            min_value=0,
            max_value=len(st.session_state.df),
            value=len(st.session_state.df),
            step=1
        )
    with c2:
        if st.button("Add Row", use_container_width=True):
            cols = list(st.session_state.df.columns)
            empty_row = {col: "" for col in cols}
            st.session_state.df = pd.concat(
                [st.session_state.df.iloc[:insert_idx], pd.DataFrame([empty_row]), st.session_state.df.iloc[insert_idx:]],
                ignore_index=True
            )
            st.rerun()

    # Advanced column tools tucked away
    with st.expander("⚙️ Column Tools (Rename / Add / Remove)", expanded=False):
        cols_list = list(st.session_state.df.columns)

        # Rename
        st.markdown("**Rename a column**")
        rc1, rc2, rc3 = st.columns([2, 3, 1])
        with rc1:
            rename_idx = st.selectbox(
                "Choose column",
                options=list(range(len(cols_list))),
                format_func=lambda i: cols_list[i],
                key="rename_idx",
            )
        with rc2:
            new_name = st.text_input("New name", value=cols_list[rename_idx], key="rename_value")
        with rc3:
            if st.button("Rename", use_container_width=True, key="rename_btn"):
                new_name_s = (new_name or "").strip()
                if not new_name_s:
                    st.info("Column name cannot be empty.")
                elif new_name_s in cols_list and new_name_s != cols_list[rename_idx]:
                    st.info("A column with that name already exists.")
                else:
                    cols_list[rename_idx] = new_name_s
                    st.session_state.df.columns = cols_list
                    st.experimental_rerun()

        st.divider()

        # Add
        st.markdown("**Add a column**")
        ac1, ac2, ac3 = st.columns([3, 2, 1])
        with ac1:
            add_name = st.text_input("Column name", key="add_col_name", placeholder="e.g., Notes")
        with ac2:
            insert_at = st.number_input(
                "Insert at index",
                min_value=0, max_value=len(cols_list), value=len(cols_list), step=1, key="add_insert_idx"
            )
        with ac3:
            if st.button("Add", use_container_width=True, key="add_btn"):
                name = (add_name or "").strip() or f"Column_{len(cols_list)+1}"
                if name in cols_list:
                    st.info("A column with that name already exists.")
                else:
                    st.session_state.df.insert(int(insert_at), name, "")
                    st.experimental_rerun()

        st.divider()

        # Remove
        st.markdown("**Remove a column**")
        dc1, dc2 = st.columns([4, 1])
        with dc1:
            drop_choice = st.selectbox(
                "Column",
                options=(["None"] + [f"{i}: {n}" for i, n in enumerate(st.session_state.df.columns)]),
                key="drop_choice"
            )
        with dc2:
            if st.button("Remove", use_container_width=True, key="drop_btn"):
                if drop_choice != "None":
                    idx = int(drop_choice.split(":")[0])
                    col_to_drop = list(st.session_state.df.columns)[idx]
                    st.session_state.df.drop(columns=[col_to_drop], inplace=True)
                    st.experimental_rerun()

# --------------------------- 5) Download ---------------------------
if st.session_state.df is not None:
    st.header("5️⃣ Download")

    # Name input (simple; applies immediately)
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
        st.caption(" ")  # spacer

    # Prepare payloads
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
