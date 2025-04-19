import streamlit as st
import aiohttp
import asyncio
import json
import numpy as np
import pydicom
from PIL import Image, ImageFilter
import re
import cv2

st.set_page_config(page_title="Indotelemed AI Radiologi", layout="centered")
st.title("Indotelemed AI Radiologi")

AI_MODEL = "ALIENTELLIGENCE/medicalimaginganalysis:latest"
TARGET_IMAGE_SIZE = (600, 600)

def format_response(text):
    table_pattern = r'(\|.*\|\n)(\|?-+?-+\|.*\n)((\|.*\|.*\n)+)'
    matches = re.finditer(table_pattern, text)
    
    for match in reversed(list(matches)):
        full_table = match.group(0)
        rows = [row.strip() for row in full_table.split('\n') if row.strip()]
        
        headers = [f"<th>{h.strip()}</th>" for h in rows[0].split('|') if h.strip()]
        header_row = f"<tr>{''.join(headers)}</tr>"
        
        body_rows = []
        for row in rows[2:]:
            cells = [f"<td>{c.strip()}</td>" for c in row.split('|') if c.strip()]
            body_rows.append(f"<tr>{''.join(cells)}</tr>")
        
        html_table = f"""
        <div style="margin:15px 0;border:1px solid #eee;border-radius:8px;overflow-x:auto">
            <table style="width:100%;border-collapse:collapse">
                <thead style="background-color:#f8f9fa">{header_row}</thead>
                <tbody>{''.join(body_rows)}</tbody>
            </table>
        </div>
        """
        text = text.replace(full_table, html_table)

    text = re.sub(r"(\d+\.)\s*", r"<br><b>\1</b> ", text)
    text = re.sub(r"•\s*", "<br>• ", text)
    text = re.sub(r"\n{2,}", "<br><br>", text)
    text = re.sub(r"(KESAN|DIAGNOSA|RECOMMENDATION):", r"<br><b>\1:</b>", text, flags=re.IGNORECASE)
    
    return f"<div style='line-height:1.6;text-align:justify'>{text}</div>"

@st.cache_data(max_entries=3)
def process_image(uploaded_file):
    try:
        if uploaded_file.name.endswith(".dcm"):
            dicom = pydicom.dcmread(uploaded_file)
            pixel_array = dicom.pixel_array.astype(np.float32)

            modality = str(dicom.get((0x0008, 0x0060), "UNKNOWN")).upper()
            if "CT" in modality:
                p1, p99 = np.percentile(pixel_array, (1, 99))
                wc = float((p99 + p1) / 2.0)
                ww = float(p99 - p1)
            else:
                wc = dicom.get('WindowCenter', None)
                ww = dicom.get('WindowWidth',  None)

            if wc is not None and ww is not None:
                mn = wc - ww/2; mx = wc + ww/2
                pixel_array = np.clip(pixel_array, mn, mx)
                pixel_array = (pixel_array - mn) / (mx - mn) * 255
            else:
                pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255

            if dicom.PhotometricInterpretation == "MONOCHROME1":
                pixel_array = 255 - pixel_array

            if "CT" in modality:

                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                pixel_array = clahe.apply(pixel_array.astype(np.uint8))

                gamma = 1.2
                inv_gamma = 1.0 / gamma
                lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
                pixel_array = cv2.LUT(pixel_array.astype(np.uint8), lut)

            image = Image.fromarray(pixel_array.astype(np.uint8))

            study_desc_elem = dicom.get((0x0008,0x1030), None)
            study_desc = str(study_desc_elem.value) if hasattr(study_desc_elem, "value") else "Tidak Ada Deskripsi"

            patient_pos_elem = dicom.get((0x0018,0x5100), None)
            patient_position = str(patient_pos_elem.value) if hasattr(patient_pos_elem, "value") else "Tidak Diketahui"

            image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

            return image.resize(TARGET_IMAGE_SIZE, Image.LANCZOS), {
                "body_part": str(dicom.get((0x0018,0x0015), "Tidak Diketahui")),
                "modality": modality,
                "study_desc": study_desc,
                "patient_position": patient_position,
                "window_center": wc,
                "window_width": ww
            }

        else:
            image = Image.open(uploaded_file)
            return image.resize(TARGET_IMAGE_SIZE, Image.LANCZOS), None

    except Exception as e:
        st.error(f"Gagal memproses citra medis: {e}")
        return None, None

async def generate_response(session, prompt):
    async with session.post(
        "http://localhost:11434/api/generate",
        json={"model": AI_MODEL, "prompt": prompt, "stream": True}
    ) as response:
        async for line in response.content:
            if line:
                decoded = json.loads(line)
                yield decoded.get("response", "")

with st.form("analysis_form"):
    st.subheader("Analisis citra Medis")
    
    uploaded_file = st.file_uploader(
        "Upload Citra Medis (DICOM)",
        type=["png", "jpg", "jpeg", "dcm"],
        accept_multiple_files=False
    )
    
    user_prompt = st.text_area(
        "Pertanyaan/Keluhan Pasien:",
        placeholder="Contoh: Tolong analisa kemungkinan anomali pada citra medis ini...",
        height=150
    )
    
    submit_btn = st.form_submit_button("Mulai Analisis", type="primary")

if submit_btn:
    if not uploaded_file and not user_prompt:
        st.warning("Harap upload citra medis atau masukkan pertanyaan!")
    else:
        with st.spinner("Sedang menganalisis..."):
            try:
                image_info = ""
                dicom_meta = {}
                analysis_guide = ""
                
                if uploaded_file:
                    image, dicom_meta = process_image(uploaded_file)
                    st.image(image, caption="Citra medis yang Dianalisis", width=320)
                    
                    if dicom_meta:
                        body_part = dicom_meta['body_part'].upper()
                        guide_mapping = {
                            "CHEST": "Fokus analisis pada: Paru-paru, jantung, tulang rusuk, diafragma, trakea, jaringan lunak thorax, efusi pleura, Skoliosis, hitung nilai cardio thoraxic ratio (CTR) dan tuberkulosis",
                            "ABDOMEN": "Fokus analisis pada: Organ abdomen, usus, ginjal, hati, limpa, pankreas, kandung empedu, dan adanya cairan bebas",
                            "HEAD": "Fokus analisis pada: Tengkorak, jaringan otak, sinus, ventrikel, struktur intrakranial, dan perdarahan",
                            "SPINE": "Fokus analisis pada: Vertebra, diskus intervertebralis, kanalis spinalis, alignment, dan lesi tulang",
                            "EXTREMITY": "Fokus analisis pada: Tulang panjang, sendi, jaringan lunak, fraktur, dan deformitas",
                            "HAND": "Fokus analisis pada: Tulang karpal/metakarpal/falang, sendi MCP/PIP/DIP, jaringan lunak, dan artritis",
                            "SKULL": "Fokus analisis pada: Tulang kranial, sutura, sinus paranasal, dan lesi destruktif",
                            "KNEE": "Fokus analisis pada: Kondilus femur/tibia, patella, ruang sendi, ligamen, meniskus, dan efusi",
                            "NECK": "Fokus analisis pada: Vertebra servikal, kelenjar tiroid, trakea, jaringan lunak leher, dan pembuluh darah",
                            "PELVIS": "Fokus analisis pada: Tulang pelvis, sendi sakroiliaka, kandung kemih, dan organ reproduksi",
                            "FOOT": "Fokus analisis pada: Tulang tarsal/metatarsal/falang, sendi Lisfranc, dan deformitas",
                            "SHOULDER": "Fokus analisis pada: Glenohumeral joint, akromion, rotator cuff, dan impingement",
                            "WRIST": "Fokus analisis pada: Tulang karpal, sendi radiokarpal, TFCC, dan fraktur Colles/Scaphoid",
                            "ELBOW": "Fokus analisis pada: Olecranon, fossa olecrani, sendi humeroulnar, dan efusi sendi",
                            "ANKLE": "Fokus analisis pada: Malleolus medial/lateral, talus, ligamen lateral, dan fraktur Weber",
                            "EXTREMITY": "Fokus analisis pada: Tulang panjang, sendi, jaringan lunak, fraktur, dan deformitas"
                        }

                        analysis_guide = guide_mapping.get(
                            body_part.split()[0] if ' ' in body_part else body_part,
                            "Lakukan analisis umum sesuai modalitas dan anatomi yang terlihat secara akurat"
                        )

                        image_info = f"""
                        ## Informasi DICOM:
                        **Bagian Tubuh:** {dicom_meta.get('body_part', 'Tidak Diketahui')}
                        **Modalitas:** {dicom_meta.get('modality', 'Tidak Diketahui')}
                        **Posisi Pasien:** {dicom_meta.get('patient_position', 'Tidak Diketahui')}
                        **Deskripsi Pemeriksaan:** {dicom_meta.get('study_desc', 'Tidak Ada Deskripsi')}

                        **Panduan Analisis:**
                        {analysis_guide}
                        """

                base_prompt = f"""
                **Instruksi Analisis:**
                1. Identifikasi body part: {dicom_meta.get('body_part', '')}
                2. Diagnosa Awal: {dicom_meta.get('study_desc', '')}
                3. Analisis gambar citra medis: {analysis_guide}
                4. Berikan 3-12 diagnosis diferensial termasuk prediksi Kanker dan Tumor dengan probabilitas (format: [Nama Diagnosis] - [X]%) jelaskan prediksinya tanpa definisi.

                **Data Input:**
                {image_info if uploaded_file else ''}
                **Keluhan Pasien:** {user_prompt or 'Analisis umum citra medis yang akurat, termasuk kelainan-kelainan yang terdapat pada body_part.'}
                """

                response_placeholder = st.empty()
                response_container = {"text": ""}

                async def main():
                    async with aiohttp.ClientSession() as session:
                        async for chunk in generate_response(session, base_prompt):
                            response_container["text"] += chunk
                            response_placeholder.markdown(
                                format_response(response_container["text"]), 
                                unsafe_allow_html=True
                            )
                
                asyncio.run(main())
                
                st.divider()
                st.caption("⚠️ Hasil analisis ini merupakan asisten virtual dan tidak menggantikan diagnosis dokter.")
            
            except aiohttp.ClientError as e:
                st.error(f"Gagal terkoneksi ke AI Engine: {str(e)}")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")