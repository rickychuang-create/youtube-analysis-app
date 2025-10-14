import streamlit as st
import googleapiclient.discovery
import pandas as pd
from datetime import datetime, timedelta, timezone
from openai import OpenAI
import json
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ========= Streamlit 美化 & API 初始化 =========
st.set_page_config(
    page_title="YouTube頻道AI策略分析工具", 
    page_icon="▶️", layout="wide",
    menu_items={
        'Get Help': 'https://docs.google.com/document/d/1DqiaZmd7DEL5DM7U1qYymmkcnYtjD9EMhvFKO_1PnL8/edit?tab=t.nis2vgychuxj#heading=h.ff9yikkfazez', # 範例：連結到 Streamlit 官方文件
        'About': """

        ##### 專案簡介：
        這是一個整合 YouTube Data API 與 OpenAI 技術的 AI 策略分析工具，以YouTube數據(影片觀看&留言)分析作者頻道受眾輪廓 & 粉絲痛點，以協助 AM/MKT/BD 掌握目標客群Insights以及作者品牌價值主張，並透過AI工具產出符合用戶痛點的行銷文案。

        -----

        """
    })

st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    /* 讓App標題與新的紅色主題色呼應 */
    h1, h2, h3 {color: #ff4b4b;} 

    /* 所有分頁的容器 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px; /* 分頁之間的間距 */
        
        display: flex !important;
        width: 100% !important;
    }

    /* 未選中分頁的樣式 */
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #444;
        transition: background-color 0.3s, color 0.3s;
        flex-grow: 1; /* 允許分頁伸展以填滿空間 */
        justify-content: center; /* 水平置中文字 */
        text-align: center; /* 確保文字居中對齊 */
    }

    /* 滑鼠懸停在「未選中」分頁上的樣式 */
    .stTabs [data-baseweb="tab"]:not([aria-selected="true"]):hover {
        background-color: #e8e8e8;
        color: #ff4b4b;
    }

    /* 已選中分頁的樣式 */
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }

    </style>
""", unsafe_allow_html=True)


try:
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    client = OpenAI(api_key=OPENAI_API_KEY)
except (FileNotFoundError, KeyError):
    st.error("錯誤：請先在 .streamlit/secrets.toml 中設定您的 API 金鑰。")
    st.stop()

# ========= 功能模組 =========
@st.cache_data(ttl=3600)
def get_channel_info(channel_id):
    request = youtube.channels().list(part="contentDetails,snippet", id=channel_id)
    response = request.execute()
    if not response.get("items"): return None, None
    item = response["items"][0]
    return item['contentDetails']['relatedPlaylists']['uploads'], item['snippet']['title']

@st.cache_data(ttl=3600)
def get_channel_videos(uploads_playlist_id, max_videos=1000):
    video_ids, next_page_token = [], None
    progress_bar = st.progress(0, text="抓取影片ID中...")
    item_count = 0
    while True:
        pl_request = youtube.playlistItems().list(part="contentDetails", playlistId=uploads_playlist_id, maxResults=50, pageToken=next_page_token)
        pl_response = pl_request.execute()
        video_ids += [item['contentDetails']['videoId'] for item in pl_response['items']]
        next_page_token = pl_response.get("nextPageToken")
        item_count += len(pl_response['items'])
        if item_count % 100 == 0:
             progress_bar.progress(min(1.0, item_count / max_videos), text=f"已抓取 {item_count} 個影片ID...")
        if not next_page_token or len(video_ids) >= max_videos: break
    progress_bar.empty()
    videos = []
    for i in range(0, min(len(video_ids), max_videos), 50):
        batch = video_ids[i:i+50]
        v_request = youtube.videos().list(part="snippet,statistics", id=",".join(batch))
        v_response = v_request.execute()
        for item in v_response['items']:
            videos.append({"video_id": item['id'], "title": item['snippet']['title'],"publishedAt": pd.to_datetime(item['snippet']['publishedAt']),"viewCount": int(item['statistics'].get('viewCount', 0))})
    return pd.DataFrame(videos)

@st.cache_data(ttl=3600)
def get_recent_comments(videos_df, days=180, channel_name=None):
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    recent_videos = videos_df[videos_df['publishedAt'] >= cutoff_date]
    all_comments = []
    progress_bar = st.progress(0, text="抓取留言中...")
    for i, vid in enumerate(recent_videos['video_id']):
        try:
            next_page_token = None
            while True:
                c_request = youtube.commentThreads().list(part="snippet", videoId=vid, maxResults=100, pageToken=next_page_token)
                c_response = c_request.execute()
                for item in c_response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    if channel_name and comment['authorDisplayName'] == channel_name: continue
                    all_comments.append({"video_id": vid, "author": comment['authorDisplayName'], "published_at": comment['publishedAt'], "like_count": comment['likeCount'], "text": comment['textDisplay']})
                next_page_token = c_response.get("nextPageToken")
                if not next_page_token: break
        except Exception: continue
        finally: progress_bar.progress((i + 1) / len(recent_videos), text=f"抓取影片留言...({i+1}/{len(recent_videos)})")
    progress_bar.empty()
    return pd.DataFrame(all_comments)

def analyze_channel_with_openai(channel_id, videos_df):
    video_text = "\n".join([f"- {row['title']} (觀看數: {row['viewCount']})" for _, row in videos_df.iterrows()])
    prompt = f"""
    你是一位頂尖的 YouTube 頻道策略分析師。我正在研究一個頻道，其 ID 為 {channel_id}。
    請根據我提供的最新影片清單（標題與瀏覽數），用專業、有條理的方式分析這個頻道。
    影片清單:
    {video_text}
    請嚴格遵循以下 Markdown 表格格式進行分析，不要有任何多餘的文字描述：
    ### 1. YouTuber介紹
    | 創作者名稱 | 專長 | 風格 | 
    | :--- | :--- | :--- | 
    | (創作者名稱) | (根據影片內容推測創作者的專業領域) | (根據影片內容推測創作者的風格) |
    ### 2. 頻道介紹
    | 頻道核心內容與價值主張 |
    | :--- |
    | (總結頻道的核心內容與價值主張) |   
    ### 3.1 頻道內容剖析
    | 影片類型 | 範例影片標題 | 影片數量 | 平均瀏覽數 | 目標受眾需求 |
    | :--- | :--- | :--- | :--- | :--- |
    | (例如：個股分析) | (挑選1-2個代表性標題) | (影片數量) | (平均瀏覽數) | (說明滿足了觀眾什麼需求) |
    | (例如：市場趨勢) | (挑選1-2個代表性標題) | (影片數量) | (平均瀏覽數) | (說明滿足了觀眾什麼需求) |
    ### 3.2 Top 10 熱門影片分析
    | 排名 | 標題名稱 | 瀏覽數 | 洞察分析 (為何受歡迎) |
    | :--- | :--- | :--- | :--- |
    | 1 | (瀏覽數最高的影片標題) | (對應的瀏覽數) | (分析這支影片爆紅的原因) |
    | 2 | ... | ... | ... |
    ### 4. 受眾輪廓分析
    | 重要性排序 | 受眾類型 | 心理驅動 | 受眾特徵 | 觀看行為/內容偏好 | 代表影片（觀看數）
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | (根據該影片類型影片數以及平均瀏覽數兩個維度分析，將該影片類型的受眾依照重要性排序) | (該影片類型的受眾類型) | (該影片類型受眾的心理驅動) | (該影片類型的受眾特徵) | (該影片類型受眾觀看行為/內容偏好) | (該影片類型代表影片與觀看數) |
    """
    response = client.chat.completions.create(model="gpt-5-mini", messages=[{"role":"user","content": prompt}])
    return response.choices[0].message.content

def analyze_comments_with_openai(channel_id, comments_df):
    comment_text = "\n".join([f"- {text}" for text in comments_df['text'].tolist()])
    prompt = f"""
    你是一位敏銳的市場分析與產品開發專家。我正在研究 ID 為 {channel_id} 的 YouTube 頻道，並收集了觀眾最近的提問留言。
    請根據這些留言，分析粉絲的痛點，並提出具體的變現建議（例如：線上課程或 App）。
    用戶提問留言:
    {comment_text}
    請嚴格遵循以下 Markdown 表格格式進行分析，不要有任何多餘的文字描述：
    ### 5. 粉絲痛點分析
    | 痛點分類 | 核心問題 | 留言數 | 留言範例 |
    | :--- | :--- | :--- | :--- |
    | **(例如：知識系統化)** | 粉絲覺得資訊零散，希望能有系統地學習。 | (估算該痛點類型留言數) | (挑選1-2則代表性留言) |
    | **(例如：實作困難)** | 知道理論但不知如何實際操作或應用。 | (估算該痛點類型留言數) | (挑選1-2則代表性留言) |
    """
    response = client.chat.completions.create(model="gpt-5-mini", messages=[{"role":"user","content": prompt}])
    return response.choices[0].message.content

def analyze_target_audience_insight(product_category, channel_analysis, comment_analysis):
    prompt = f"""
    你是一位頂尖的市場策略家與消費者心理分析專家。請深度學習以下 KOL 的綜合分析資料，並針對「{product_category}」這個產品品類，挖掘出最核心的目標客群洞察 (TA Insights)。

    ---
    ### KOL 綜合分析資料
    #### 頻道受眾與內容分析:
    {channel_analysis}
    #### 粉絲痛點與需求分析:
    {comment_analysis}
    ---

    請嚴格依照以下 Markdown 架構，以第一人稱（"我"）的角度，深入地剖析目標客群對於「{product_category}」的心理狀態，產出洞察報告。
    ### 6. 目標客群洞察 (TA Insights)
    | Insights | 說明 |
    | :--- | :--- |
    | **Belief / Myth (信念/迷思)** | (我對於這類「{product_category}」的認知是什麼？我相信什麼？我所認定的事實是什麼？) |
    | **Need / Pain Point (需求/痛點)** | (我的核心需求或最大痛點是什麼？) |
    | **Current Solutions (現有解決方案)** | (為了解決這個痛點，我目前都是怎麼做的？) |
    | **Limitation / Unsatisfaction (限制/不滿)** | (為什麼我目前的需求或痛點，仍然不能被現有的解決方案完全滿足？) |

    ### 7. Benefits & Reason To Believe
    | Benefits & Reason-To-Believe | 說明 |
    | :--- | :--- |
    | **Functional Benefit (功能效益)** | (在功能上，我最想要這個產品帶給我什麼具體的好處？) |
    | **Emotional Benefit (情感效益)** | (在使用這個產品後，我最渴望獲得什麼樣的情感滿足或心理轉變？) |
    | **Parity Benefit (市場入場券)** | (我認為這類的產品一定要有哪些基本的功能或效益，才值得我考慮？) |
    | **Differentiation Benefit (差異化價值)** | (需要有什麼獨特的功能、體驗或價值，才能讓我眼睛一亮，並強烈地想要擁有這個產品？) |
    | **RTB (Reason-to-Believe)** | (為什麼我應該要相信這個產品真的能提供上述效益？) |
    """
    response = client.chat.completions.create(model="gpt-5-mini", messages=[{"role":"user", "content": prompt}])
    return response.choices[0].message.content

def analyze_commercialization_ideas(product_type, edited_insights):
    if product_type == "線上課程":
        prompt = f"""
        你是一位頂尖的線上課程設計專家。請根據下方提供的目標客群洞察 (TA Insights)，為這位 KOL 推薦 1 到 3 個最適合的線上課程，主題要跟投資相關。

        ---
        ### 目標客群洞察 (TA Insights)
        {edited_insights}
        ---

        請為每一個推薦的課程，嚴格依照以下 Markdown 格式進行規劃：
        ### 推薦課程一：(課程名稱)
        #### 課程簡介：(用 2-3 句話簡潔說明這門課的目標與價值)
        #### 課程大綱：
        | 課程章節 | 章節名稱 | 章節簡介 | 對應洞察/痛點 |
        | :--- | :--- | :--- | :--- |
        | (一) | (章節名稱) | (章節要闡述的內容，1-2句話) | (說明對應到解決目標客群洞察或是Benefits & Reason To Believe的哪一個點，可以是多選項) |
        | (二) | (章節名稱) | (章節要闡述的內容，1-2句話) | (說明對應到解決目標客群洞察或是Benefits & Reason To Believe的哪一個點，可以是多選項) |
        | ... | ... | ... | ... |
        """
    else: # App
        prompt = f"""
        你是一位經驗豐富的 App 產品經理。請根據下方提供的目標客群洞察 (TA Insights)，為這位 KOL 精心設計一款最能解決粉絲痛點的 App，App的設計要跟投資理財相關。
        ---
        ### 目標客群洞察 (TA Insights)
        {edited_insights}
        ---
        請嚴格依照以下 Markdown 格式進行 App 規劃：
        ### **App 名稱推薦**： (App 推薦名稱)
        #### **App 核心價值**：(用一句話說明這個 App 的核心目標與獨特賣點)
        | 核心功能 | 功能描述 | 對應洞察/痛點 |
        | :--- | :--- | :--- |
        | **(功能名稱)** | (功能描述) | (說明對應到解決目標客群洞察或是Benefits & Reason To Believe的哪一個點，可以是多選項) |
        | **(功能名稱)** | (功能描述) | (說明對應到解決目標客群洞察或是Benefits & Reason To Believe的哪一個點，可以是多選項) |
        """
    response = client.chat.completions.create(model="gpt-5-mini", messages=[{"role":"user","content": prompt}])
    return response.choices[0].message.content



def analyze_brand_value_proposition(product_description, audience_insights):
    """
    根據產品描述和客群洞察，生成一句話的品牌價值主張。
    """
    prompt = f"""
    你是一位世界級品牌策略家，擅長將複雜的市場洞察與受眾心理轉化為一句極具說服力的品牌價值主張 (Brand Value Proposition)。
    請深入理解以下關於某位 KOL 的產品定位與目標客群洞察，並產出該 KOL 在推廣此產品時，最能打動目標受眾的一句品牌價值主張。

    ---
    ### 規劃中的產品描述
    {product_description}

    ### 目標客群 Insights
    {audience_insights}
    ---

    請確保這句話：
    - 明確說出 KOL 的角色與他帶來的核心價值
    - 聚焦於受眾的痛點與渴望
    - 具備差異化與情感共鳴
    - 語氣自然、可口述、能用於行銷文案

    請只輸出一句話，並用以下Markdown格式回覆：

    ### 8. 品牌價值主張 (Brand Value Proposition)
    (Brand Value Proposition描述)
    """
    response = client.chat.completions.create(model="gpt-5-mini", messages=[{"role":"user", "content": prompt}])
    return response.choices[0].message.content.strip()


def analyze_marketing_funnel(kol_name, product_description, audience_insight, bvp_result, start_stage, end_stage):
    start_stage_desc = start_stage.split('：')[1]
    end_stage_desc = end_stage.split('：')[1]
    prompt = f"""
    你是一位世界級的行銷漏斗策略專家 (Marketing Funnel Strategist)。
    ### 分析背景
    我們把行銷 funnel 分成以下幾個階段：
    - 階段0：陌生、未知這項產品或服務。
    - 階段1：知悉、接觸過這項產品或服務。
    - 階段2：感興趣、比較這項產品或服務與現有解決方案的差異。
    - 階段3：體驗、試用這項產品或服務。
    - 階段4：首購、使用這項產品或服務。
    - 階段5：再購、續用這項產品或服務。
    - 階段6：分享、推薦這項產品或服務。
    ### 任務目標
    我們現在的目標客群是 **{kol_name}** 的粉絲。
    他們對於這項產品，他們目前正處於 **「{start_stage_desc} (階段{start_stage[2]})」**。
    我們的目標是引導他們從 **階段{start_stage[2]}** 移動到 **「{end_stage_desc} (階段{end_stage[2]})」**。
    ### 核心指令
    請根據下方提供的【目標客群深度 Insight】、【產品描述】、【品牌價值主張】，一步一步地分析：為了讓目標客群完成上述的階段移動，我們在每一個過渡階段會遇到哪些**阻力(Barriers)**或**驅力(Drivers)**？
    **請特別注意：**
    1.  這裡的阻力與驅力，請專注於**與產品效益(Benefits)無直接相關**的因素，例如：使用者習慣、心理門檻、社群影響、轉換流程的便利性、價格感知等。
    2.  請明確列出在每個階段可以與目標客群互動的**接觸點 (Touchpoints)**。
    3.  針對每一項阻力，提出對應的**關鍵任務 (Key Task)** 或 **突破點**，說明該如何設計行動來幫助用戶跨越障礙，順利往下一階段移動。
    ---
    ### 【目標客群深度 Insight】
    {audience_insight}

    ### 【產品描述】
    {product_description}
    
    ### 【品牌價值主張】
    {bvp_result}
    ---
    請嚴格依照以下 Markdown 架構格式呈現你的分析報告，不用其他多餘的文字，如果選取範圍內還有其他階段就繼續往下產出。

    ### 9. 行銷 Funnel 分析
    #### **階段x → 階段y**：
    * **接觸點(Touchpoints)**：
        * **接觸點1**
        * **接觸點2**
        * **接觸點...**
    * **阻力(Barrier)**：
        * **阻力1**
        * **阻力2**
        * **阻力...**
    * **驅力(Driver)**：
        * **驅力1**
        * **驅力2**
        * **驅力...**
    * **突破點(Key Tasks)**：
        * **突破點1**
        * **突破點2**
        * **突破點...**
    """
    response = client.chat.completions.create(model="gpt-5-mini", messages=[{"role":"user", "content": prompt}])
    return response.choices[0].message.content

def create_blank_doc_in_folder(title, folder_id, user_email):
    """在指定的共享資料夾中，建立一份空白的 Google Docs 文件並分享。"""
    try:
        creds_info = dict(st.secrets["google_credentials"])
        creds_info['private_key'] = creds_info['private_key'].replace('\\n', '\n')
        SCOPES = ['https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
        drive_service = build('drive', 'v3', credentials=creds)
        file_metadata = {'name': title, 'mimeType': 'application/vnd.google-apps.document', 'parents': [folder_id]}
        file = drive_service.files().create(body=file_metadata, supportsAllDrives=True, fields='id, webViewLink').execute()
        doc_id = file.get('id')
        doc_url = file.get('webViewLink')
        if not doc_id: return None, "未能取得 Document ID。"
        if user_email:
            permission = {'type': 'user', 'role': 'writer', 'emailAddress': user_email}
            drive_service.permissions().create(fileId=doc_id, body=permission, sendNotificationEmail=False, supportsAllDrives=True).execute()
        return doc_url, None
    except HttpError as e:
        error_details = json.loads(e.content.decode('utf-8'))
        error_message = error_details.get('error', {}).get('message', str(e))
        return None, f"建立 Google Docs 失敗: {error_message}"
    except Exception as e:
        return None, f"建立 Google Docs 失敗: {e}"

# ========= Streamlit UI (全新互動式流程) =========

st.title("▶️ YouTube 頻道 AI 策略分析工具")

SHARED_FOLDER_ID = "1-lJlBB5n3lJzu_LlM15HDeKghjBZ3dbY"

if 'current_step' not in st.session_state:
    st.session_state.current_step = 1


tab_list = [
    "Step 1: 鎖定頻道", "Step 2: 頻道受眾分析", "Step 3: 粉絲痛點洞察",
    "Step 4: 目標客群 Insight", "Step 5: 產品內容變現建議", "Step 6: 品牌價值主張", 
    "Step 7: 行銷 Funnel 分析", "Step 8: 總結與下載"
]

tabs = st.tabs(tab_list)


def show_gdoc_link():
    if 'gdoc_url' in st.session_state and st.session_state.gdoc_url:
        st.success(f"**報告文件已建立！** 隨時 [點此在新分頁開啟]({st.session_state.gdoc_url})，並將複製的內容貼上。")
        st.markdown("---")

def display_and_copy_block(section_title, content_key, help_text=""):
    if content_key in st.session_state and st.session_state[content_key]:
        st.markdown("---")
        st.subheader(section_title, help=help_text)
        view_tab, copy_tab = st.tabs(["閱讀分析結果", "複製 Markdown 原始碼"])
        with view_tab:
            st.markdown(st.session_state[content_key])
        with copy_tab:
            st.markdown("👇 點擊右上方圖示複製，然後使用點擊右鍵選擇【從Markdown貼上】貼到您的 Google Docs 文件中。")
            st.code(st.session_state[content_key], language="markdown")

with tabs[0]: # Step 1
    st.header("🎯 請輸入您想分析的 YouTube 頻道 ID")
    st.markdown("這是整個分析流程的起點，請貼上目標頻道的 ID。")
    channel_id_input = st.text_input("YouTube Channel ID", value=st.session_state.get('channel_id', ''), placeholder="例如：UC-qgS_2Q2nF_3a9hAlh_aYg")
    if st.button("驗證並鎖定頻道 ▶", key="lock_channel"):
        if channel_id_input:
            with st.spinner("驗證頻道資訊中..."):
                uploads_id, channel_title = get_channel_info(channel_id_input)
                if not uploads_id:
                    st.error("找不到該頻道，請檢查 Channel ID 是否正確。")
                else:
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.session_state.channel_id = channel_id_input
                    st.session_state.uploads_id = uploads_id
                    st.session_state.channel_title = channel_title
                    st.session_state.current_step = 2
                    st.success(f"成功鎖定頻道：**{channel_title}**！請前往 Step 2 繼續分析。")
 
        else:
            st.warning("請先輸入 Channel ID")

with tabs[1]: # Step 2
    if st.session_state.current_step < 2:
        st.info("請先在 Step 1 完成頻道的鎖定。")
    else:
        st.header(f"📊 **{st.session_state.channel_title}** - 頻道整體內容與受眾分析")
        show_gdoc_link()
        if 'gdoc_url' not in st.session_state:
            with st.expander("📂 想要開始建立 Google Docs 報告嗎？"):
                st.markdown("您可以現在就建立一份空白報告，後續步驟的產出就能隨時複製貼上。")
                user_email_input = st.text_input("您的 Google Email (用於共享文件)", key="user_email_input", placeholder="your.name@company.com")
                if st.button("在新分頁建立空白報告文件", key="create_gdoc"):
                    if not user_email_input:
                        st.warning("請輸入您的 Email 以便共享文件。")
                    else:
                        doc_title = f"{st.session_state.channel_title}_AI策略分析報告_{datetime.now().strftime('%Y-%m-%d')}"
                        with st.spinner("正在建立 Google Docs 文件..."):
                            doc_url, error = create_blank_doc_in_folder(doc_title, SHARED_FOLDER_ID, user_email_input)
                        if error:
                            st.error(error)
                        else:
                            st.session_state.gdoc_url = doc_url
                            st.rerun()
        if 'videos_df' not in st.session_state:
            if st.button("抓取頻道所有影片", key="fetch_videos"):
                with st.spinner("抓取影片資料中..."): 
                    st.session_state.videos_df = get_channel_videos(st.session_state.uploads_id)
                st.success(f"成功抓取 {len(st.session_state.videos_df)} 支影片！")

        if 'videos_df' in st.session_state:
            st.subheader("影片清單預覽")
            st.dataframe(st.session_state.videos_df.head(10))
            st.download_button(
                label="⬇️ 下載完整影片清單 (CSV)",
                data=st.session_state.videos_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{st.session_state.get('channel_title', 'export')}_videos.csv",
                mime="text/csv"
            )

            if st.button("🤖 使用 AI 進行受眾與內容深度分析", key="openai_channel_analysis"):
                with st.spinner("AI 正在進行深度分析..."): 
                    st.session_state.channel_analysis_result = analyze_channel_with_openai(st.session_state.channel_id, st.session_state.videos_df)

            
            display_and_copy_block("AI 全頻道分析結果", "channel_analysis_result", "分析此頻道的影片主題、內容類型與熱門影片特徵，並描繪出可能的目標受眾輪廓。")
            if 'channel_analysis_result' in st.session_state and st.button("前往下一步：粉絲痛點洞察 →", key="goto_step3"):
                st.session_state.current_step = 3
                st.info("已解鎖 Step 3，請點擊上方分頁標籤繼續。")


with tabs[2]: # Step 3
    if st.session_state.current_step < 3: 
        st.info("請先在 Step 2 完成分析並點擊「前往下一步」。")
    else:
        st.header(f"💬 **{st.session_state.channel_title}** - 粉絲留言與痛點分析")
        show_gdoc_link()
        days = st.number_input("設定要分析最近幾天內的影片留言", 7, 3650, 180, 1)
        if st.button("抓取近期留言", key="fetch_comments"):
            if 'videos_df' not in st.session_state: st.warning("請先返回 Step 2 抓取影片清單。")
            else:
                with st.spinner("抓取留言資料中..."): st.session_state.comments_df = get_recent_comments(st.session_state.videos_df, days=days, channel_name=st.session_state.channel_title)
                st.success(f"成功抓取 {len(st.session_state.comments_df)} 則留言！")

        if 'comments_df' in st.session_state:
            st.subheader(f"最近 {days} 天內上傳影片的留言預覽")
            st.dataframe(st.session_state.comments_df.head(10))
            st.download_button(
                label="⬇️ 下載完整留言清單 (CSV)",
                data=st.session_state.comments_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{st.session_state.get('channel_title', 'export')}_comments.csv",
                mime="text/csv"
            )
            if st.button("🤖 使用 AI 分析粉絲痛點", key="openai_comment_analysis"):
                with st.spinner("AI 正在分析粉絲留言..."):
                    question_patterns = r"\?|？|怎麼|如何|為什麼|嗎|能不能|可不可以|怎么|为什么|吗"
                    questions_df = st.session_state.comments_df[st.session_state.comments_df['text'].str.contains(question_patterns, na=False, regex=True)]
                    if questions_df.empty: st.session_state.comment_analysis_result = "找不到包含問題的留言，無法進行痛點分析。"
                    else: st.session_state.comment_analysis_result = analyze_comments_with_openai(st.session_state.channel_id, questions_df)
                st.rerun()
            
            display_and_copy_block("AI 粉絲痛點分析結果", "comment_analysis_result", "歸納粉絲在留言中提出的問題與困擾。")
            if 'comment_analysis_result' in st.session_state and st.button("前往下一步：目標客群 Insight →", key="goto_step4"):
                st.session_state.current_step = 4
                st.info("已解鎖 Step 4，請點擊上方分頁標籤繼續。")

with tabs[3]: # Step 4
    if st.session_state.current_step < 4: 
        st.info("請先在 Step 3 完成分析並點擊「前往下一步」。")
    else:
        st.header(f"🧠 **{st.session_state.channel_title}** - Step 4: 目標客群 Insight")
        show_gdoc_link()
        if 'channel_analysis_result' not in st.session_state or 'comment_analysis_result' not in st.session_state: 
            st.warning("⚠️ 警告：缺少 Step 2 或 Step 3 的 AI 分析結果。")
        else:
            product_category = st.radio("請選擇要針對哪個產品「品類」進行客群洞察分析：", ("線上課程", "App"), horizontal=True, key="product_category_s4")

            if st.button(f"🤖 針對「{product_category}」產生目標客群 Insight", key="openai_insight_analysis"):
                with st.spinner("AI 正在深度挖掘目標客群 Insight..."):
                    st.session_state.insight_analysis_result = analyze_target_audience_insight(product_category, st.session_state.channel_analysis_result, st.session_state.comment_analysis_result)
            
            display_and_copy_block("AI 目標客群 Insight 報告", "insight_analysis_result", "深入剖析潛在顧客對於特定產品品類的深層心理動機、需求、痛點與價值觀。")

            if 'insight_analysis_result' in st.session_state:
                st.markdown("---")
                st.subheader("下一步？")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### 需要 AI 建議產品內容？")
                    if st.button("執行 Step 5：獲取 產品內容變現建議 →", use_container_width=True, key="goto_step5"):
                        st.session_state.current_step = 5
                        st.info("已解鎖 Step 5，請點擊上方分頁標籤繼續。")
                with col2:
                    st.markdown("##### 已有既定產品？")
                    if st.button("跳至 Step 6：輸入既定產品 →", type="secondary", use_container_width=True, key="skip_step5"):
                        st.session_state.current_step = 6
                        # 清除可能存在的舊產品建議，確保 Step 6 知道我們是「跳過」的
                        if 'commercialization_result' in st.session_state:
                            del st.session_state['commercialization_result']
                        st.info("已略過 Step 5 並解鎖 Step 6，請點擊上方分頁標籤繼續。")

# <<< 全新 Step 5 >>>
with tabs[4]: # Step 5
    if st.session_state.current_step < 5: 
        st.info("請先在 Step 4 完成分析並選擇對應的下一步。")
    else:
        st.header(f"💡 **{st.session_state.channel_title}** - Step 5: 產品內容變現建議")
        show_gdoc_link()
        if 'insight_analysis_result' not in st.session_state: 
            st.warning("⚠️ 警告：缺少 Step 4 的 AI Insight 分析結果。")
        else:
            product_type = st.session_state.get("product_category_s4", "線上課程")
            st.markdown(f"此階段將根據您在下方提供的**客群洞察**，產出更具體的 **{product_type}** 建議。")

            edited_insights = st.text_area(
                "目標客群 Insights (您可以根據 Step 4 的結果進行編輯)",
                value=st.session_state.get("insight_analysis_result", ""),
                height=300,
                key="edited_insights_s5"
            )
            
            if st.button(f"🤖 根據以上 Insight 產生「{product_type}」推薦內容", key="openai_commercialization_analysis"):
                if not edited_insights.strip():
                    st.warning("目標客群 Insights 內容不可為空。")
                else:
                    with st.spinner(f"AI 正在為您規劃 {product_type} ..."):
                        st.session_state.commercialization_result = analyze_commercialization_ideas(product_type, edited_insights)
            
            display_and_copy_block("產品內容變現建議", "commercialization_result", "根據您提供的客群洞察，生成具體的線上課程或 App 產品規劃。")

            if 'commercialization_result' in st.session_state and st.button("前往下一步：品牌價值主張 →", key="goto_step6_from5"):
                st.session_state.current_step = 6
                st.info("已解鎖 Step 6，請點擊上方分頁標籤繼續。")


with tabs[5]: # Step 6
    if st.session_state.current_step < 6: 
        st.info("請先在 Step 4 或 Step 5 完成分析並點擊「前往下一步」。")
    else:
        st.header(f"⭐️ **{st.session_state.channel_title}** - Step 6: 品牌價值主張")
        show_gdoc_link()
        if 'insight_analysis_result' not in st.session_state:
            st.warning("⚠️ 警告：缺少 Step 4 的 AI Insight 分析結果。")
        else:
            st.markdown("此階段將根據您最終確認的產品描述與客群洞察，為 KOL 提煉出一句核心的品牌價值主張 (Brand Value Proposition)。")
            st.markdown("---")

            # --- 1. 產品描述確認區 (與前版相同) ---
            st.subheader("1. 請確認或輸入產品描述")
            
            product_description = ""
            if 'commercialization_result' in st.session_state and st.session_state.commercialization_result:
                product_description = st.text_area("請在此手動輸入或修改您的產品描述：", value=st.session_state.commercialization_result, height=250)
            else:
                st.info("您已選擇略過 AI 產品建議，請於下方手動輸入您的既定產品內容。")
                product_description = st.text_area("請在此手動輸入您的產品描述：", height=250, placeholder="例如：課程名稱是「美股新手從 0 到 1」，課程大綱包含...")

            st.session_state.final_product_description = product_description
            
            st.markdown("---")

            # --- 2. 客群 Insights 編輯區 ---
            st.subheader("2. 請確認或編輯目標客群 Insights")

            # <<< 關鍵修正：預設值會優先讀取 Step 5 編輯過的版本 >>>
            default_insights = st.session_state.get("edited_insights_s5", st.session_state.get("insight_analysis_result", ""))
            
            edited_insights = st.text_area(
                "目標客群 Insights (您可以根據先前步驟的結果進行最終編輯)",
                value=default_insights,
                height=400,
                key="edited_insights_s6"
            )

            st.markdown("---")
            
            # --- 3. AI 分析觸發 ---
            if st.button("🤖 提煉品牌價值主張 (一句話)", key="openai_bvp_analysis"):
                if not edited_insights.strip() or not st.session_state.get("final_product_description", "").strip():
                    st.warning("請確保「產品描述」和「目標客群 Insights」皆有內容。")
                else:
                    # 更新最終版的 Insights 到 session_state，供後續步驟使用
                    st.session_state.final_edited_insights = edited_insights
                    with st.spinner("AI 正在提煉品牌價值主張..."):
                        st.session_state.bvp_result = analyze_brand_value_proposition(
                            st.session_state.final_product_description,
                            edited_insights
                        )
            
            display_and_copy_block(
                section_title="AI 品牌價值主張結果",
                content_key="bvp_result",
                help_text="這是根據產品描述與客群洞察，為KOL提煉出的一句話核心價值主張。"
            )

            if 'bvp_result' in st.session_state and st.button("前往下一步：行銷 Funnel 分析 →", key="goto_step7"):
                st.session_state.current_step = 7
                st.info("已解鎖 Step 7，請點擊上方分頁標籤繼續。")


with tabs[6]: # Step 7
    if st.session_state.current_step < 7: 
        st.info("請先在 Step 6 完成分析並點擊「前往下一步」。")
    else:
        st.header(f"📈 **{st.session_state.channel_title}** - Step 7: 行銷 Funnel 分析")
        show_gdoc_link()
        if 'insight_analysis_result' not in st.session_state or 'bvp_result' not in st.session_state:
            st.warning("⚠️ 警告：缺少 Step 5 或 Step 6 的分析結果。")
        else:
            st.subheader("行銷 Funnel 簡介")
            st.markdown("""
            行銷漏斗是一個描述顧客從初次接觸商品(品牌)到最終完成購買或成為忠實粉絲所經歷的旅程模型。我們的目標是分析在每個階段中，有哪些因素會 **驅使(Drivers)** 他們前進，又有哪些會 **阻礙(Barriers)** 他們，並找出關鍵的突破點。
            - **階段0：** 陌生、未知這項產品或服務。
            - **階段1：** 知悉、接觸過這項產品或服務。
            - **階段2：** 感興趣、比較這項產品或服務與現有解決方案的差異。
            - **階段3：** 體驗、試用這項產品或服務。
            - **階段4：** 首購、使用這項產品或服務。
            - **階段5：** 再購、續用這項產品或服務。
            - **階段6：** 分享、推薦這項產品或服務。
            """)
            st.markdown("---")

            st.subheader("分析前提預覽")
            with st.expander("點此查看本次 Funnel 分析的基礎資料", expanded=False):
                st.markdown("##### 規劃中的產品")
                st.info(st.session_state.get("final_product_description", "N/A"))
                st.markdown("##### 目標客群 Insights")
                st.info(st.session_state.get("insight_analysis_result", "N/A"))
                st.markdown("##### 品牌價值主張")
                st.info(st.session_state.get('bvp_result', 'N/A'))
            
            st.subheader("設定 Funnel 分析範圍")
            funnel_stages = ["階段0：陌生、未知", "階段1：知悉、接觸", "階段2：感興趣、比較", "階段3：體驗、試用", "階段4：首購、使用", "階段5：再購、續用", "階段6：分享、推薦"]
            start_stage = st.selectbox("目標客群【起始階段】：", options=funnel_stages, index=1, key="start_stage_s7")
            end_stage = st.selectbox("目標客群【結束階段】：", options=funnel_stages, index=4, key="end_stage_s7")
            
            st.markdown("---")
            if st.button(f"🤖 分析從 {start_stage.split('：')[0]} 到 {end_stage.split('：')[0]} 的 Barriers & Drivers", key="openai_funnel_analysis"):
                with st.spinner("AI 正在分析行銷 Funnel 策略..."): 
                    st.session_state.funnel_analysis_result = analyze_marketing_funnel(
                        st.session_state.channel_title,
                        st.session_state.final_product_description,
                        st.session_state.insight_analysis_result,
                        st.session_state.bvp_result,
                        start_stage, 
                        end_stage
                    )
                st.rerun()
            
            display_and_copy_block("AI 行銷 Funnel 策略報告", "funnel_analysis_result", "分析引導用戶在行銷漏斗中前進的關鍵驅動力與阻礙因素，並提出對應的策略建議。")
            if 'funnel_analysis_result' in st.session_state and st.button("前往最終步驟 →", key="goto_step8"):
                st.session_state.current_step = 8
                st.info("已解鎖 Step 8，請點擊上方分頁標籤繼續。")


with tabs[7]: # Step 8
    if st.session_state.current_step < 6:
        st.info("請先在 Step 7 完成分析並點擊「前往下一步」。")
    else:
        st.header("✅ 總結與下載")
        show_gdoc_link()
        st.markdown("您已完成所有分析步驟。如果需要，您可以在此下載分析過程中的原始數據。")
        if 'videos_df' in st.session_state:
            st.download_button(label="⬇️ 下載影片清單 (CSV)", data=st.session_state.videos_df.to_csv(index=False).encode("utf-8-sig"), file_name=f"{st.session_state.get('channel_title', 'export')}_videos.csv")
        if 'comments_df' in st.session_state:
            st.download_button(label="⬇️ 下載留言清單 (CSV)", data=st.session_state.comments_df.to_csv(index=False).encode("utf-8-sig"), file_name=f"{st.session_state.get('channel_title', 'export')}_comments.csv")
        st.markdown("---")
        st.info("若要重新分析一個新的頻道，請回到 Step 1 輸入新的 Channel ID。若需要分析同個KOL不同品類的目標客群Insight，請回到 Step 4 選擇品類並繼續進行分析。")


