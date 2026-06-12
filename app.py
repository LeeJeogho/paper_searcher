import streamlit as st
import requests
import pandas as pd
import time
import io
import json
import fitz  # PyMuPDF (PDF 텍스트 추출용)
from google import genai
from google.genai import types

# ---------------------------------------------------------
# API 키 및 설정 영역
RAW_API_KEY = "s2k-rbPCLs8ltTBAC3ujE9MuS2dx0afIJC98xR89mhIl"
API_KEY = "".join(RAW_API_KEY.split()).strip() 
HEADERS = {"x-api-key": API_KEY} if API_KEY else {}

# 교수님 ID 고정
PROFESSOR_ID = "7651824" 

# 🎯 Gemini API 세팅 (3단계 분석용)
RAW_GEMINI_API_KEY = "AIzaSyCe17Tn8_E9cYr0UqUz-KRjlJb1jAgpzbI"
GEMINI_API_KEY = "".join(RAW_GEMINI_API_KEY.split()).strip()

client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------

@st.cache_data(ttl=3600)
def search_authors(name):
    time.sleep(1.1)
    url = f"https://api.semanticscholar.org/graph/v1/author/search?query={name}&fields=name,paperCount,affiliations"
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return response.json().get('data', [])
    except Exception as e:
        st.error(f"저자 검색 중 오류 발생: {e}")
        return []

@st.cache_data(ttl=3600)
def fetch_author_papers(author_id):
    time.sleep(1.1)
    url = f"https://api.semanticscholar.org/graph/v1/author/{author_id}/papers?fields=title,abstract,year,citationCount,authors,externalIds,openAccessPdf,url"
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return response.json().get('data', [])
    except Exception as e:
        st.error(f"저자 논문 수집 중 오류: {e}")
        return []

@st.cache_data(ttl=3600)
def search_global_papers(keywords, sort_by_citation=False):
    time.sleep(1.1)
    query_str = "+".join([kw.strip() for kw in keywords if kw.strip()])
    if not query_str:
        return []
    
    sort_param = "&sort=citationCount:desc" if sort_by_citation else ""
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query_str}&limit=100{sort_param}&fields=title,abstract,year,citationCount,authors,externalIds,openAccessPdf,url"
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return response.json().get('data', [])
    except Exception as e:
        st.error(f"전체 논문 검색 중 오류: {e}")
        return []

def filter_and_format_papers(papers, keywords, search_mode):
    filtered_results = []
    valid_keywords = [kw.strip().lower() for kw in keywords if kw.strip()]
    
    for p in papers:
        title = p.get('title', '') or ''
        abstract = p.get('abstract', '') or ''
        content = f"{title} {abstract}".lower()
        
        if not valid_keywords:
            is_match = True
        elif search_mode == "AND (모두 포함)":
            is_match = all(kw in content for kw in valid_keywords)
        else:
            is_match = any(kw in content for kw in valid_keywords)
        
        if is_match:
            doi = p.get('externalIds', {}).get('DOI')
            doi_url = f"https://doi.org/{doi}" if doi else "N/A"
            pdf_info = p.get('openAccessPdf')
            full_text_url = pdf_info.get('url') if pdf_info else p.get('url')
            
            authors_list = p.get('authors', [])
            author_names = ", ".join([a.get('name', '') for a in authors_list])
            
            is_professor_included = any(str(a.get('authorId')) == PROFESSOR_ID for a in authors_list)
            
            if is_professor_included:
                author_names = f"⭐ {author_names}"

            filtered_results.append({
                "선택": False,
                "우선순위": is_professor_included, 
                "연도": p.get('year'),
                "저자": author_names,
                "제목": title,
                "인용": p.get('citationCount'),
                "DOI 링크": doi_url,
                "원문 링크": full_text_url,
                "초록": abstract[:150] + "..." if abstract else "N/A"
            })
            
    if filtered_results:
        df = pd.DataFrame(filtered_results)
        df = df.sort_values(by=["우선순위", "연도"], ascending=[False, False])
        df = df.drop(columns=["우선순위"])
        return df
    else:
        return pd.DataFrame()

# ---------------------------------------------------------
# [2단계] PDF 다운로드 및 텍스트 추출 함수
# ---------------------------------------------------------
def extract_pdf_text(pdf_url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/pdf, text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Referer': 'https://scholar.google.com/'
        }
        response = requests.get(pdf_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        pdf_stream = io.BytesIO(response.content)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return f"PDF 추출 실패: {e}"

# ---------------------------------------------------------
# [3단계] LLM 파라미터 파싱 함수
# ---------------------------------------------------------
def analyze_paper_with_llm(text):
    if not client:
        return {"error": "Gemini API 키가 설정되지 않았습니다."}
    
    prompt = """
    다음은 반도체 공정/소자 관련 논문의 본문 텍스트입니다.
    이 논문에서는 다양한 실험 조건(예: 열처리 온도 변화, 두께 변화, 조성비 변화 등)에 따른 여러 개의 소자(Devices)가 비교되었을 수 있습니다.
    
    텍스트를 분석하여 '각 소자(또는 실험 조건)별'로 아래 파라미터들을 추출해 주세요.
    결과는 반드시 여러 소자의 정보를 담은 **JSON 배열(Array) 형태**로 반환해야 합니다.
    
    [🚨 데이터 추출 핵심 가이드라인 (정확도 및 누락 방지)]
    1. 문맥 기반 정밀 매칭: 이동도(mobility), SS(subthreshold swing), Vth 등의 값이 표(Table)에 없고 초록(Abstract)이나 결론(Conclusion)의 줄글에만 있다면, 해당 수치가 **'정확히 어떤 조건(예: 300°C 열처리, 10nm 두께 등)'에서 측정된 것인지 문맥을 파악**하여 반드시 그 조건과 일치하는 소자 배열에 채워 넣으세요.
    2. 출처가 불분명한 대표 수치 격리: 만약 초록에 "최고 이동도는 15를 달성했다"라고 적혀 있는데 이것이 어떤 소자의 결과인지 텍스트상으로 도저히 연결할 수 없다면, 기존 소자 데이터에 억지로 섞어 넣지 마세요. 대신 `"sample_condition": "Best Device (Abstract summary)"`라는 독립된 객체를 배열에 하나 더 추가하여 거기에만 기입하세요.
    3. 공통 값 일괄 적용: 게이트 절연막 두께(GI thickness), 절연막 물질, 채널 물질 등이 논문 전체 소자에 '공통'으로 사용되었다면, 모든 소자 배열 데이터에 동일하게 그 값을 복사해서 채워주세요. 알 수 없는 정보는 "N/A"로 표기하세요.
    3. 맞춤형 단위 강제 변환 (매우 중요):
        - 각종 '두께(thickness)' 데이터는 본문에 A(옹스트롬)나 um로 나와 있어도 반드시 **nm(나노미터)** 단위로 변환하여 숫자만 적으세요. (예: 1000A -> 100)
        - '채널 길이/너비(length, width)' 데이터는 반드시 **um(마이크로미터)** 단위로 변환하여 숫자만 적으세요.
        - '열처리 시간(time)' 데이터는 본문에 시간(hour)이나 분(min)으로 나와 있어도 반드시 계산을 통해 **초(sec)** 단위로 변환하여 숫자만 적으세요. (예: 1h -> 3600, 30min -> 1800)
        - '열처리 분위기(atmosphere)' 데이터는 뒤에 붙는 'atmosphere' 단어를 무조건 제외하고 **'Air', 'O2', 'N2', 'Vacuum'** 등 핵심 환경 이름만 깔끔하게 표기하세요. (예: "Air atmosphere" -> "Air")
    4. 조성비: composition_ratio는 논문에 언급된 비율(예: 1:1:1)이나 원자 퍼센트(at%)를 최대한 그대로 명시해 주세요.
    5. 값이 명확히 없거나 텍스트에서 알 수 없는 정보는 반드시 "N/A"로 표기하세요.
    
    [추출 대상 파라미터 (각 소자별)]
    - Name: 샘플명 또는 조건 (예: "300C Annealed", "Device A", "Best Device")
    - channel_material_name: 채널 물질명 (예: IGZO)
    - composition: 물질 조성비 (예: 1:1:1, 10 at%)
    - crystallinity: 결정성 (예: Amorphous, Crystalline)
    - device_structure_type: 소자 구조 (예: BGTC)
    - gate_electrode_material: 게이트 전극 물질
    - gate_insulator_material: 게이트 절연막 물질
    - gate_insulator_process: 게이트 절연막 공정/증착법
    - gate_insulator_thickness_nm: 게이트 절연막 두께 (nm 단위 숫자만)
    - substrate_material: 기판 물질
    - sd_electrode_material: Source/Drain 전극 물질
    - passivation_layer_material: 패시베이션(보호막) 물질
    - passivation_layer_thickness: 패시베이션 두께
    - passivation_process: 패시베이션 공정/증착법
    - passivation_RF Power: 패시베이션 RF 파워
    - passivation_Annealing temperature: 패시베이션 열처리 온도
    - passivation_Annealing time (sec): 패시베이션 열처리 시간 (초 단위 숫자만)
    - passivation_Annealing atmosphere: 패시베이션 열처리 분위기 ('atmosphere' 단어 제외, 예: Air, O2)
    - passivation_Partial O2 pressure: 패시베이션 산소 분압
    - semiconductor_thickness_nm: 반도체(채널) 두께 (nm 단위 숫자만)
    - channel_width_um: 채널 너비 (um 단위 숫자만)
    - channel_length_um: 채널 길이 (um 단위 숫자만)
    - field_effect_mobility_cm²/V⋅s: 이동도 (숫자만)
    - threshold_voltage_V: 문턱 전압 (숫자만)
    - subthreshold_swing_V/dec: SS (숫자만)
    - on_off_ratio: On/Off 전류비
    - CH_power_W: 채널 증착 Power (W)
    - target: 타겟 물질/종류
    - CH_gas_type: 채널 증착 가스 종류
    - CH_process_pressure_Torr: 채널 증착 공정 압력 (Torr)
    - CH_oxygen_partial_pressure_ratio (%): 채널 산소 분압 비율 (%)
    - CH_substrate_temperature_°C: 채널 기판 온도 (C)
    - CH_annealing_time_: 채널 열처리 시간 (초 단위 숫자만, 예: 1h -> 3600)
    - CH_annealing_temperature_°C: 채널 열처리 온도 (C)
    - CH_annealing_atmosphere: 채널 열처리 분위기 ('atmosphere' 단어 제외, 예: Air, O2)
    - Deposition_Method: 증착 방식 (예: ALD, Sputtering 등)

    반드시 마크다운 코드 블록 없이 순수한 JSON 배열만 출력하세요.
    
    [논문 본문]
    """ + text[:25000]

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        return json.loads(response.text)
    except Exception as e:
        return {"error": f"LLM 분석 실패: {e}"}

# --- UI 구성 ---
st.set_page_config(page_title="반도체 논문 검색 시스템", layout="wide")
st.title("논문 검색 및 상세 분석 시스템")

st.markdown("### 1. 검색 조건 설정")
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    search_name = st.text_input("찾으려는 저자 영문 성명 (비워두면 전 세계 검색)", placeholder="예: Jae Kyeong Jeong")
with col2:
    process_keywords = st.text_input("공정 키워드 (쉼표 구분)", placeholder="예: ALD, Oxide").split(",")
with col3:
    search_mode = st.radio("검색 방식 선택", ["AND (모두 포함)", "OR (하나라도 포함)"])

use_citation_sort = st.checkbox("🏆 전 세계 검색 시 '역대 인용 수'가 가장 많은 100개 논문 우선 가져오기 (체크 해제 시 연관성 순)")

st.divider()

if "search_results" not in st.session_state:
    st.session_state.search_results = pd.DataFrame()

# --- 분기 1: 전 세계 검색 ---
if not search_name.strip():
    if st.button("전 세계 논문 검색 시작"):
        with st.spinner("데이터 확보 중..."):
            prof_papers = fetch_author_papers(PROFESSOR_ID)
            global_papers = search_global_papers(process_keywords, sort_by_citation=use_citation_sort)
            
            combined_papers = []
            seen_titles = set()
            
            for p in prof_papers + global_papers:
                title = p.get('title', '')
                if title and title.lower() not in seen_titles:
                    combined_papers.append(p)
                    seen_titles.add(title.lower())

            st.session_state.search_results = filter_and_format_papers(combined_papers, process_keywords, search_mode)

# --- 분기 2: 특정 저자 검색 ---
else:
    authors = search_authors(search_name)
    if authors:
        st.markdown("### 2. 저자 선택 및 필터링")
        author_options = {
            f"{a['name']} (논문 {a['paperCount']}편, 소속: {a.get('affiliations', ['N/A'])[0] if a.get('affiliations') else 'N/A'})": a['authorId'] 
            for a in authors
        }
        selected_author_label = st.selectbox("정확한 저자를 선택하세요", options=list(author_options.keys()))
        selected_id = author_options[selected_author_label]
        
        if st.button("선택한 저자의 논문 필터링"):
            with st.spinner("해당 저자의 논문을 분석 중입니다..."):
                papers = fetch_author_papers(selected_id)
                st.session_state.search_results = filter_and_format_papers(papers, process_keywords, search_mode)
    else:
        if search_name:
            st.error("일치하는 저자를 찾을 수 없습니다.")

# ---------------------------------------------------------
# [상호작용] 체크박스 렌더링 및 상세 분석 트리거
# ---------------------------------------------------------
if not st.session_state.search_results.empty:
    st.success(f"총 {len(st.session_state.search_results)}건의 관련 논문을 찾았습니다. 분석할 논문을 선택하세요.")
    
    edited_df = st.data_editor(
        st.session_state.search_results,
        column_config={
            "선택": st.column_config.CheckboxColumn(
                "분석 선택",
                help="상세 파라미터를 추출할 논문을 선택하세요.",
                default=False,
            )
        },
        disabled=["연도", "저자", "제목", "인용", "DOI 링크", "원문 링크", "초록"],
        hide_index=True,
        width="stretch"  
    )
    
    selected_papers = edited_df[edited_df["선택"] == True]
    
    st.divider()
    st.markdown("### 3. 선택 논문 상세 분석 (Mobility-Reliability Trade-off)")
    st.info(f"현재 선택된 논문: {len(selected_papers)}건")
    
    if len(selected_papers) > 0:
        if st.button("✅ 선택한 논문 AI 상세 파라미터 추출 시작"):
            results_list = []
            all_device_data = [] # 🎯 엑셀 추출용 데이터 배열
            
            for index, row in selected_papers.iterrows():
                title = row['제목']
                pdf_url = row['원문 링크']
                
                st.write(f"**진행 중:** {title}")
                
                if pd.isna(pdf_url) or not str(pdf_url).startswith("http"):
                    st.warning("오픈 액세스 PDF 링크를 찾을 수 없습니다. 출판사 보안 정책(또는 유료화)으로 인해 자동 다운로드가 제한되었습니다.(수동 다운로드 필요)")
                    continue
                    
                with st.spinner("PDF 본문을 추출하는 중입니다..."):
                    extracted_text = extract_pdf_text(pdf_url)
                    
                if "PDF 추출 실패" in extracted_text:
                    st.warning(extracted_text)
                    continue
                    
                with st.spinner("AI가 공정 파라미터 및 소자 특성을 분석하고 있습니다..."):
                    llm_result = analyze_paper_with_llm(extracted_text)
                    
                results_list.append({
                    "제목": title,
                    "AI 추출 결과": llm_result
                })
                
                # 🎯 엑셀용 데이터 평면화: 추출된 소자별 데이터에 '논문 제목' 추가
                if isinstance(llm_result, list):
                    for dev in llm_result:
                        dev_copy = dev.copy()
                        dev_copy["Paper_Title"] = title
                        all_device_data.append(dev_copy)
                elif isinstance(llm_result, dict):
                    llm_result["Paper_Title"] = title
                    all_device_data.append(llm_result)

                st.success("추출 완료!")
                time.sleep(4.1) 
            
            if results_list:
                st.markdown("#### ✨ AI 분석 결과 요약")
                st.json(results_list)
                
            # 🎯 [기능 추가] 자동 다운로드 논문 엑셀 생성
            if all_device_data:
                df_excel = pd.DataFrame(all_device_data)
                # Paper_Title 열을 맨 앞으로 이동
                cols = ["Paper_Title"] + [c for c in df_excel.columns if c != "Paper_Title"]
                df_excel = df_excel[cols]
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_excel.to_excel(writer, index=False, sheet_name='Semiconductor_Data')
                
                st.download_button(
                    label="📥 자동 분석 결과 엑셀 파일로 다운로드",
                    data=output.getvalue(),
                    file_name="Semiconductor_Analysis_Result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

# ---------------------------------------------------------
# [추가 기능] 로컬 PDF 수동 업로드 및 AI 분석
# ---------------------------------------------------------
st.divider()
st.markdown("### 🛠️ 4. 수동 PDF 업로드 분석 (에러 난 논문용)")
st.info("오픈 액세스가 아니거나 다운로드가 막힌 논문은 직접 PDF를 다운로드하여 여기에 올려주세요.")

uploaded_file = st.file_uploader("PDF 파일 업로드", type=["pdf"])

if uploaded_file is not None:
    if st.button("업로드한 PDF 상세 파라미터 추출"):
        with st.spinner("업로드된 PDF 본문을 읽는 중입니다..."):
            try:
                # 업로드된 파일에서 텍스트 추출
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                manual_text = ""
                for page in doc:
                    manual_text += page.get_text()
                
                with st.spinner("Gemini AI가 공정 파라미터 및 소자 특성을 분석하고 있습니다..."):
                    llm_result = analyze_paper_with_llm(manual_text)
                
                st.success("수동 분석 완료!")
                st.json({"파일명": uploaded_file.name, "AI 추출 결과": llm_result})
                
                # 🎯 [기능 추가] 수동 업로드 논문 엑셀 생성
                manual_device_data = []
                if isinstance(llm_result, list):
                    for dev in llm_result:
                        dev_copy = dev.copy()
                        dev_copy["File_Name"] = uploaded_file.name
                        manual_device_data.append(dev_copy)
                elif isinstance(llm_result, dict):
                    llm_result["File_Name"] = uploaded_file.name
                    manual_device_data.append(llm_result)

                if manual_device_data:
                    df_manual = pd.DataFrame(manual_device_data)
                    cols_m = ["File_Name"] + [c for c in df_manual.columns if c != "File_Name"]
                    df_manual = df_manual[cols_m]
                    
                    output_manual = io.BytesIO()
                    with pd.ExcelWriter(output_manual, engine='xlsxwriter') as writer:
                        df_manual.to_excel(writer, index=False, sheet_name='Manual_Analysis_Data')
                    
                    st.download_button(
                        label="📥 수동 분석 결과 엑셀 파일로 다운로드",
                        data=output_manual.getvalue(),
                        file_name="Manual_Analysis_Result.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            except Exception as e:
                st.error(f"파일 처리 중 오류 발생: {e}")