import streamlit as st
import googleapiclient.discovery
import pandas as pd
from datetime import datetime, timedelta, timezone
from openai import OpenAI
import re
from collections import Counter
import plotly.express as px

# ========= Streamlit 美化 =========
st.set_page_config(page_title="YouTube頻道分析工具", page_icon="📊", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #f0f2f6; padding: 20px;}
    h1 {color: #1a73e8;}
    .stButton>button {background-color: #1a73e8; color: white;}
    </style>
""", unsafe_allow_html=True)

# ========= API 初始化 (安全地從 st.secrets 讀取) =========
try:
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    client = OpenAI(api_key=OPENAI_API_KEY)
except FileNotFoundError:
    st.error("錯誤：找不到 secrets.toml 檔案。請先建立 .streamlit/secrets.toml 並設定您的 API 金鑰。")
    st.stop()
except KeyError:
    st.error("錯誤：請在 secrets.toml 中設定 'YOUTUBE_API_KEY' 和 'OPENAI_API_KEY'。")
    st.stop()


# ========= 功能模組 (與您原本的程式碼相同，稍作優化) =========
@st.cache_data(ttl=3600) # 快取資料 1 小時，避免重複抓取
def get_channel_info(channel_id):
    """抓取頻道基本資訊，包含 uploads_playlist_id 和頻道名稱"""
    request = youtube.channels().list(part="contentDetails,snippet", id=channel_id)
    response = request.execute()
    if not response.get("items"):
        return None, None
    item = response["items"][0]
    uploads_playlist_id = item['contentDetails']['relatedPlaylists']['uploads']
    channel_title = item['snippet']['title']
    return uploads_playlist_id, channel_title

@st.cache_data(ttl=3600)
def get_channel_videos(uploads_playlist_id, max_videos=1000):
    # ... (此函式內容與您提供的版本幾乎相同，為簡潔省略) ...
    # ... (建議保留您原本高效的爬取邏輯) ...
    video_ids, next_page_token = [], None
    while True:
        pl_request = youtube.playlistItems().list(
            part="contentDetails", playlistId=uploads_playlist_id, maxResults=50, pageToken=next_page_token
        )
        pl_response = pl_request.execute()
        video_ids += [item['contentDetails']['videoId'] for item in pl_response['items']]
        next_page_token = pl_response.get("nextPageToken")
        if not next_page_token or len(video_ids) >= max_videos: break
    
    videos = []
    for i in range(0, min(len(video_ids), max_videos), 50):
        batch = video_ids[i:i+50]
        v_request = youtube.videos().list(part="snippet,statistics", id=",".join(batch))
        v_response = v_request.execute()
        for item in v_response['items']:
            snippet = item['snippet']
            stats = item['statistics']
            videos.append({
                "video_id": item['id'],
                "title": snippet['title'],
                "publishedAt": pd.to_datetime(snippet['publishedAt']),
                "viewCount": int(stats.get('viewCount', 0))
            })
    return pd.DataFrame(videos)


@st.cache_data(ttl=3600)
def get_recent_comments(videos_df, days=180, channel_name=None):
    # ... (此函式內容與您提供的版本相同，為簡潔省略) ...
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    videos_df['publishedAt'] = pd.to_datetime(videos_df['publishedAt'])
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
                    author = comment['authorDisplayName']
                    if channel_name and author == channel_name:
                        continue
                    all_comments.append({
                        "video_id": vid, "author": author, "published_at": comment['publishedAt'],
                        "like_count": comment['likeCount'], "text": comment['textDisplay']
                    })
                next_page_token = c_response.get("nextPageToken")
                if not next_page_token: break
        except Exception as e:
            st.warning(f"抓取影片 {vid} 留言失敗: {e}", icon="⚠️")
            continue
        finally:
            progress_bar.progress((i + 1) / len(recent_videos), text=f"抓取留言中...({i+1}/{len(recent_videos)})")
    progress_bar.empty()
    return pd.DataFrame(all_comments)

def analyze_channel_with_openai(channel_id, videos_df):
    # ... (此函式內容與您提供的版本相同，為簡潔省略) ...
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
    | ... | ... | ... | ... | ... |

    ### 3.2 Top 10 熱門影片分析
    | 排名 | 標題名稱 | 瀏覽數 | 洞察分析 (為何受歡迎) |
    | :--- | :--- | :--- | :--- |
    | 1 | (瀏覽數最高的影片標題) | (對應的瀏覽數) | (分析這支影片爆紅的原因) |
    | 2 | ... | ... | ... |
    | 3 | ... | ... | ... |
    | 4 | ... | ... | ... |
    | 5 | ... | ... | ... |
    | 6 | ... | ... | ... |
    | 7 | ... | ... | ... |
    | 8 | ... | ... | ... |
    | 9 | ... | ... | ... |
    | 10 | ... | ... | ... |

    ### 4. 受眾輪廓分析
    | 重要性排序 | 受眾類型 | 心理驅動 | 受眾特徵 | 觀看行為/內容偏好 | 代表影片（觀看數）
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | (根據該影片類型影片數以及平均瀏覽數兩個維度分析，將該影片類型的受眾依照重要性排序) | (該影片類型的受眾類型) | (該影片類型受眾的心理驅動) | (該影片類型的受眾特徵) | (該影片類型受眾觀看行為/內容偏好) | (該影片類型代表影片與觀看數) |
    | (可自行增加) | ... | ... | ... | ... | ... |

    """
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role":"user","content": prompt}]
    )
    return response.choices[0].message.content

def analyze_comments_with_openai(channel_id, comments_df):
    # ... (此函式內容與您提供的版本相同，為簡潔省略) ...
    comment_text = "\n".join([f"- {text}" for text in comments_df['text'].tolist()])
    prompt = f"""
    你是一位敏銳的市場分析與產品開發專家。我正在研究 ID 為 {channel_id} 的 YouTube 頻道，並收集了觀眾最近的提問留言。
    請根據這些留言，分析粉絲的痛點，並提出具體的變現建議（例如：線上課程或 App）。

    用戶提問留言:
    {comment_text}

    請嚴格遵循以下 Markdown 表格格式進行分析，不要有任何多餘的文字描述：

    ### 1. 粉絲痛點分析
    | 痛點分類 | 核心問題 | 留言數 | 留言範例 |
    | :--- | :--- | :--- | :--- |
    | **(例如：知識系統化)** | 粉絲覺得資訊零散，希望能有系統地學習。 | (估算該痛點類型留言數) | (挑選1-2則代表性留言) |
    | **(例如：實作困難)** | 知道理論但不知如何實際操作或應用。 | (估算該痛點類型留言數) | (挑選1-2則代表性留言) |
    | **(可自行增加)** | ... | ... |

    ### 2. 商業變現建議
    | 欲解決的痛點 | 解決方案 | 理由 | 推薦內容/功能 |
    | :--- | :--- | :--- | :--- |
    | (想要解決的粉絲痛點分類，根據上方1. 粉絲痛點分析的痛點分類) | (解決方案建議) | (說明為何這個方案適合解決粉絲痛點) | (具體提出課程單元或 App 核心功能) |
    """
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role":"user","content": prompt}]
    )
    return response.choices[0].message.content

def plot_top_questions_plotly(comments_df, top_n=10):
    # ... (此函式內容與您提供的版本相同) ...
    pass # 為簡潔省略

# ========= Streamlit UI (全新改寫) =========
# <<< 修改點 2: 全新的 UI 流程 >>>
st.title("📊 YouTube 頻道分析工具")

# --- 階段 1: 輸入與模式選擇 ---
st.header("Step 1: 輸入目標頻道與選擇模式")
channel_id_input = st.text_input("輸入 YouTube Channel ID", placeholder="例如：UC-qgS_2Q2nF_3a9hAlh_aYg")
analysis_option = st.radio(
    "選擇分析模式",
    ("全頻道分析 (分析影片類型與受眾)", "留言痛點分析 (分析粉絲問題與需求)"),
    horizontal=True, key="analysis_option"
)

# --- 階段 2: 根據模式顯示對應功能 ---
if analysis_option == "全頻道分析 (分析影片類型與受眾)":
    st.header("Step 2: 全頻道分析")
    if st.button("抓取頻道所有影片", key="fetch_videos"):
        if channel_id_input:
            with st.spinner("抓取頻道資訊中..."):
                uploads_id, _ = get_channel_info(channel_id_input)
                if not uploads_id:
                    st.error("找不到該頻道，請檢查 Channel ID 是否正確。")
                else:
                    st.session_state.videos_df = get_channel_videos(uploads_id)
                    st.success(f"成功抓取 {len(st.session_state.videos_df)} 支影片！")
        else:
            st.warning("請先輸入 Channel ID")

    if 'videos_df' in st.session_state:
        st.subheader("影片清單預覽")
        st.dataframe(st.session_state.videos_df.head(10))
        st.download_button("下載完整影片清單 (CSV)", st.session_state.videos_df.to_csv(index=False).encode("utf-8-sig"), "videos.csv")
        
        # <<< 修改點 3: 獨立的 AI 分析按鈕 >>>
        if st.button("🚀 使用 OpenAI 進行深度分析", key="openai_channel_analysis"):
            with st.spinner("🤖 OpenAI 正在分析頻道中，請稍候..."):
                analysis_result = analyze_channel_with_openai(channel_id_input, st.session_state.videos_df)
                st.session_state.channel_analysis_result = analysis_result
    
    if 'channel_analysis_result' in st.session_state:
        st.subheader("🤖 OpenAI 全頻道分析結果")
        # <<< 修改點 4: 使用 st.markdown 顯示漂亮的表格 >>>
        st.markdown(st.session_state.channel_analysis_result)

elif analysis_option == "留言痛點分析 (分析粉絲問題與需求)":
    st.header("Step 2: 留言痛點分析")
    days = st.number_input("設定要分析最近幾天內的影片留言", min_value=7, max_value=3650, value=180, step=1)
    
    if st.button("抓取近期留言", key="fetch_comments"):
        if channel_id_input:
            with st.spinner("抓取頻道與影片資訊中..."):
                uploads_id, channel_title = get_channel_info(channel_id_input)
                if not uploads_id:
                    st.error("找不到該頻道，請檢查 Channel ID 是否正確。")
                else:
                    videos_df = get_channel_videos(uploads_id, max_videos=200) # 留言分析不需要全抓，抓近期影片即可
                    st.session_state.comments_df = get_recent_comments(videos_df, days=days, channel_name=channel_title)
                    st.success(f"成功抓取 {len(st.session_state.comments_df)} 則留言！")
        else:
            st.warning("請先輸入 Channel ID")

    if 'comments_df' in st.session_state:
        st.subheader(f"最近 {days} 天留言預覽")
        st.dataframe(st.session_state.comments_df.head(10))
        st.download_button("下載完整留言清單 (CSV)", st.session_state.comments_df.to_csv(index=False).encode("utf-8-sig"), "comments.csv")

        # <<< 修改點 3: 獨立的 AI 分析按鈕 >>>
        if st.button("🚀 使用 OpenAI 分析粉絲痛點", key="openai_comment_analysis"):
            with st.spinner("🤖 OpenAI 正在分析留言中，請稍候..."):
                # 簡單篩選包含問句的留言給 AI，提高分析精準度
                question_patterns = r"\?|？|怎麼|如何|為什麼|嗎|能不能|可不可以|怎么|为什么|吗"
        
        
                questions_df = st.session_state.comments_df[st.session_state.comments_df['text'].str.contains(question_patterns, na=False, regex=True)]
                if questions_df.empty:
                    st.warning("找不到包含問題的留言，無法進行痛點分析。")
                    st.session_state.comment_analysis_result = "找不到可分析的問題留言。"
                else:
                    analysis_result = analyze_comments_with_openai(channel_id_input, questions_df)
                    st.session_state.comment_analysis_result = analysis_result

    if 'comment_analysis_result' in st.session_state:
        st.subheader("🤖 OpenAI 粉絲痛點分析結果")
        # <<< 修改點 4: 使用 st.markdown 顯示漂亮的表格 >>>
        st.markdown(st.session_state.comment_analysis_result)