import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import numpy as np
import pandas as pd
from TableOCRExtractor import TableOCRExtractor
import io
import fitz  # PyMuPDF

st.set_page_config(page_title="Table OCR")

st.title("📄 Table OCR from Drawing or PDF")

def pdf_page_to_image(pdf_bytes, page_number=0, zoom=2):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_number)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return img

def init_state():
    ss = st.session_state
    if "df" not in ss:
        ss.df = None
    if "pending_select_idx" not in ss:
        ss.pending_select_idx = None
    if "prev_col_select_idx" not in ss:
        ss.prev_col_select_idx = None

init_state()

# ---------- Upload ----------
uploaded_file = st.file_uploader("Upload a drawing image or PDF", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file:
    if uploaded_file.name.lower().endswith(".pdf"):
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_num = st.number_input("Select page", 1, len(doc), 1)
        img = pdf_page_to_image(pdf_bytes, page_number=page_num - 1)
    else:
        img = Image.open(uploaded_file).convert("RGB")

    st.subheader("Crop the table")
    cropped_img = st_cropper(img, realtime_update=True, box_color='red', aspect_ratio=None)
    st.image(cropped_img, caption="Cropped Table", use_container_width=True)

    if st.button("Run OCR"):
        cropped_np = np.array(cropped_img)[:, :, ::-1] 

        # Step 3: OCR
        extractor = TableOCRExtractor(cropped_np, tesseract_path=r"C:\Users\121807\Documents\tesseract.exe")
        df = extractor.execute()

        def sanitize_columns(df):
            df.columns = [
                f"Column_{i}" if (not isinstance(c, str) or not c.strip()) else str(c)
                for i, c in enumerate(df.columns)
            ]
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

        # --- Check if OCR failed ---
        if df.empty or df.isna().all().all():
            st.info("OCR failed to detect table. Showing default BOM template.")
            df = pd.DataFrame(columns=["Part no", "Name", "Description", "Quantity", "Remarks"])
            for _ in range(5):
                df.loc[len(df)] = ["", "", "", "", ""]
        else:
            df = sanitize_columns(df)

        st.session_state.df = df

# ---------- Editor & Column Tools ----------
if st.session_state.df is not None:
    st.subheader("Extracted Table Data")

    st.session_state.df = st.data_editor(
        st.session_state.df,
        num_rows="dynamic",
        use_container_width=True,
        key="table_editor"
    )

    st.subheader("Insert Row at Specific Location")

    row_cols = list(st.session_state.df.columns)

    r1, r2, r3 = st.columns([1.6, 1.0, 0.9])
    with r1:
        st.markdown("**Insert empty row**")
    with r2:
        st.markdown("**At index**")
    with r3:
        st.markdown("** **")

    c1, c2, c3 = st.columns([1.6, 1.0, 0.9])
    with c1:
        st.markdown("&nbsp;", unsafe_allow_html=True)
    with c2:
        insert_row_idx = st.number_input(
            "",
            min_value=0,
            max_value=len(st.session_state.df),
            value=len(st.session_state.df),
            step=1,
            key="insert_row_idx",
            label_visibility="collapsed"
        )
    with c3:
        if st.button("➕ Insert Row", use_container_width=True, key="insert_row_btn"):
            empty_row = {col: "" for col in row_cols}
            upper = st.session_state.df.iloc[:insert_row_idx]
            lower = st.session_state.df.iloc[insert_row_idx:]
            st.session_state.df = pd.concat([upper, pd.DataFrame([empty_row]), lower], ignore_index=True)
            st.rerun()

    # ---------------- Column Toolbar (aligned + instant) ----------------
    st.subheader("Edit Columns")

    cols_list = list(st.session_state.df.columns)

    default_index = 0
    if st.session_state.pending_select_idx is not None and 0 <= st.session_state.pending_select_idx < len(cols_list):
        if "col_select_idx" in st.session_state:
            del st.session_state["col_select_idx"]
        default_index = int(st.session_state.pending_select_idx)
        st.session_state.pending_select_idx = None
    elif "col_select_idx" in st.session_state:
        try:
            default_index = int(st.session_state.col_select_idx)
        except Exception:
            default_index = 0

    t1, t2, t3 = st.columns([1.3, 1.6, 0.9])
    with t1:
        st.markdown("**Column**")
    with t2:
        st.markdown("**New name**")
    with t3:
        st.markdown("** **")

    c1, c2, c3 = st.columns([1.3, 1.6, 0.9])

    with c1:
        selected_index = st.selectbox(
            "",
            options=list(range(len(cols_list))),
            index=min(default_index, max(len(cols_list)-1, 0)),
            format_func=lambda i: cols_list[i],
            key="col_select_idx",
            label_visibility="collapsed"
        )

    if st.session_state.prev_col_select_idx != selected_index:
        st.session_state.prev_col_select_idx = selected_index
        if "rename_input" in st.session_state:
            del st.session_state["rename_input"]

    with c2:
        new_col_name = st.text_input(
            "",
            value=cols_list[selected_index] if cols_list else "",
            key="rename_input",
            label_visibility="collapsed",
            placeholder="Type new column name…"
        )

    with c3:
        if st.button("Rename", use_container_width=True, key="rename_btn"):
            if cols_list:
                current = cols_list[selected_index]
                new_name = (new_col_name or "").strip()
                if not new_name:
                    st.warning("Column name cannot be empty.")
                elif new_name == current:
                    pass
                elif new_name in cols_list:
                    st.warning("A column with that name already exists.")
                else:
                    cols_list[selected_index] = new_name
                    st.session_state.df.columns = cols_list
                    st.session_state.pending_select_idx = selected_index
                    st.rerun()

    # ---- Add Column ----
    a1, a2, a3 = st.columns([1.6, 1.0, 0.9])
    with a1:
        st.markdown("**Add column (name)**")
    with a2:
        st.markdown("**Insert at index**")
    with a3:
        st.markdown("** **")

    c4, c5, c6 = st.columns([1.6, 1.0, 0.9])
    with c4:
        add_name = st.text_input("", key="add_col_name", label_visibility="collapsed", placeholder="e.g., Notes")
    with c5:
        insert_at = st.number_input(
            "",
            min_value=0,
            max_value=len(cols_list),
            value=len(cols_list),
            step=1,
            key="insert_idx",
            label_visibility="collapsed"
        )
    with c6:
        if st.button("➕ Add", use_container_width=True, key="add_btn"):
            name = (add_name or "").strip() or f"Column_{len(cols_list)+1}"
            if name in cols_list:
                st.warning("A column with that name already exists.")
            else:
                st.session_state.df.insert(int(insert_at), name, "")
                st.session_state.pending_select_idx = int(insert_at)
                st.rerun()

    # ---- Remove Column ----
    r1, r2 = st.columns([2.0, 0.9])
    with r1:
        st.markdown("**Remove column**")
    with r2:
        st.markdown("** **")

    c7, c8 = st.columns([2.0, 0.9])
    with c7:
        drop_index = st.selectbox(
            "",
            options=(["None"] + [f"{i}: {name}" for i, name in enumerate(cols_list)]),
            key="drop_col_select",
            label_visibility="collapsed"
        )
    with c8:
        if st.button("🗑 Remove", use_container_width=True, key="drop_btn"):
            if drop_index != "None" and cols_list:
                idx = int(drop_index.split(":")[0])
                st.session_state.df.drop(columns=[cols_list[idx]], inplace=True)
                new_len = len(st.session_state.df.columns)
                st.session_state.pending_select_idx = min(idx, max(new_len - 1, 0)) if new_len > 0 else 0
                st.rerun()

    # ---------- Download ----------
    st.subheader("Download Final Table")

    if "download_file_name" not in st.session_state:
        st.session_state.download_file_name = "extracted_data"
    if "temp_file_name" not in st.session_state:
        st.session_state.temp_file_name = st.session_state.download_file_name

    def _apply_filename():
        name = (st.session_state.temp_file_name or "").strip()
        safe = "".join(ch if ch.isalnum() or ch in ("_", "-", " ") else "_" for ch in name) or "extracted_data"
        st.session_state.download_file_name = safe.replace(" ", "_")

    f1, f2 = st.columns([3, 1])
    with f1:
        st.markdown("**Download file name (without extension)**")
    with f2:
        st.markdown("** **")

    cfn1, cfn2 = st.columns([3, 1])
    with cfn1:
        st.text_input(
            "",
            key="temp_file_name",
            label_visibility="collapsed",
            on_change=_apply_filename
        )
    with cfn2:
        if st.button("Update Name", use_container_width=True):
            _apply_filename()
            st.rerun()

    # Prepare download buffers
    csv_buffer = io.StringIO()
    st.session_state.df.to_csv(csv_buffer, index=False)

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        st.session_state.df.to_excel(writer, index=False)

    # Download buttons side-by-side
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "📥 Download CSV",
            csv_buffer.getvalue(),
            file_name=f"{st.session_state.download_file_name}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with d2:
        st.download_button(
            "📥 Download Excel",
            excel_buffer.getvalue(),
            file_name=f"{st.session_state.download_file_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )


