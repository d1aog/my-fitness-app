import streamlit as st
import pandas as pd
import sqlite3
import datetime
import os
import hashlib
import altair as alt  # 引入图表库，用于画漂亮的折线图

# --- 兼容性导入 ---
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError as e:
    st.error(f"环境检测报错: {e}")
    st.stop()

# --- 0. 配置与初始化 ---
st.set_page_config(page_title="AI 健身助手 Pro", page_icon="🏋️", layout="wide")

# === 密码加密工具 ===
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return True
    return False

# === 数据库初始化 ===
def init_db():
    conn = sqlite3.connect('fitness_data.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS usersTable
                 (username TEXT PRIMARY KEY, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS workouts
                 (username TEXT, date TEXT, body_part TEXT, exercise TEXT, weight REAL, reps INTEGER, sets INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS diet
                 (username TEXT, date TEXT, food_item TEXT, calories REAL, protein REAL, carbs REAL, fat REAL)''')
    conn.commit()
    return conn

conn = init_db()
c = conn.cursor()

# === 预设动作库 ===
GYM_MENU = {
    "胸": ["平板卧推", "上斜卧推", "哑铃卧推", "器械夹胸", "双杠臂屈伸", "绳索夹胸"],
    "背": ["引体向上", "高位下拉", "杠铃划船", "坐姿划船", "直臂下压", "单臂哑铃划船"],
    "肩": ["坐姿推举", "哑铃侧平举", "俯身飞鸟", "面拉", "杠铃推举", "前平举"],
    "腿": ["深蹲", "硬拉", "腿举(倒蹬)", "哈克深蹲", "坐姿腿屈伸", "俯身腿弯举"],
    "手臂": ["杠铃弯举", "哑铃弯举", "绳索下压", "窄距卧推", "锤式弯举"],
    "核心": ["卷腹", "平板支撑", "悬垂举腿", "俄罗斯转体", "健腹轮"]
}

# --- 1. 登录/注册逻辑 ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''

def login_page():
    st.title("🔐 欢迎来到 AI 健身助手")
    menu = ["登录", "注册新账号"]
    choice = st.sidebar.selectbox("菜单", menu)

    if choice == "登录":
        st.subheader("请登录")
        username = st.text_input("用户名")
        password = st.text_input("密码", type='password')
        if st.button("登录"):
            c.execute('SELECT * FROM usersTable WHERE username = ?', (username,))
            data = c.fetchall()
            if data:
                if check_hashes(password, data[0][1]):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.success(f"欢迎回来, {username}!")
                    st.rerun()
                else:
                    st.warning("密码错误")
            else:
                st.warning("用户名不存在")

    elif choice == "注册新账号":
        st.subheader("创建新账号")
        new_user = st.text_input("设置用户名")
        new_password = st.text_input("设置密码", type='password')
        if st.button("注册"):
            c.execute('SELECT * FROM usersTable WHERE username = ?', (new_user,))
            if c.fetchall():
                st.error("该用户名已被占用")
            else:
                c.execute('INSERT INTO usersTable(username,password) VALUES (?,?)', 
                          (new_user, make_hashes(new_password)))
                conn.commit()
                st.success("注册成功！请登录。")

# --- 2. 主程序 ---
def main_app():
    current_user = st.session_state['username']
    
    with st.sidebar:
        st.markdown(f"### 👤 用户: **{current_user}**")
        if st.button("注销"):
            st.session_state['logged_in'] = False
            st.rerun()
        
        st.markdown("---")
        st.title("⚙️ AI 设置")
        mode = st.radio("模式", ["☁️ 在线 AI", "💻 本地 AI", "📝 仅记录 (无 AI)"])
        
        llm = None
        embeddings = None
        
        if mode == "☁️ 在线 AI":
            provider = st.selectbox("服务商", ["OpenAI / 中转", "Google Gemini", "DeepSeek"])
            api_key = st.text_input("API Key", type="password")
            base_url = "https://api.openai.com/v1"
            model_name = "gpt-4o-mini"
            
            if provider == "OpenAI / 中转":
                st.info("💡 第三方 Key 配置")
                base_url = st.text_input("Base URL", value="https://once.novai.su/v1")
                model_name = st.text_input("模型名", value="gemini-1.5-pro")
                use_local_embed = st.checkbox("✅ 强制本地 PDF 引擎", value=True)
            
            if api_key:
                try:
                    if provider == "DeepSeek":
                        llm = ChatOpenAI(model="deepseek-chat", openai_api_key=api_key, openai_api_base="https://api.deepseek.com")
                    elif provider == "Google Gemini":
                        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
                        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
                    else:
                        llm = ChatOpenAI(model=model_name, openai_api_key=api_key, openai_api_base=base_url)
                        if use_local_embed:
                            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                        else:
                            try: embeddings = OpenAIEmbeddings(openai_api_key=api_key, openai_api_base=base_url)
                            except: embeddings = None
                except Exception as e: st.error(f"配置错: {e}")

        elif mode == "💻 本地 AI":
            model_name = st.text_input("模型", "deepseek-r1:1.5b")
            base_url = st.text_input("地址", "http://localhost:11434")
            if st.button("连接"):
                try:
                    llm = ChatOllama(model=model_name, base_url=base_url)
                    embeddings = OllamaEmbeddings(model=model_name, base_url=base_url)
                    st.success("已连接")
                except: st.error("失败")

        # 加载知识库
        if "vector_db" not in st.session_state: st.session_state.vector_db = None
        if mode != "📝 仅记录 (无 AI)" and embeddings and st.session_state.vector_db is None:
            knowledge_folder = "knowledge"
            if os.path.exists(knowledge_folder) and os.listdir(knowledge_folder):
                try:
                    all_docs = []
                    for f in os.listdir(knowledge_folder):
                        if f.endswith('.pdf'):
                            loader = PyPDFLoader(os.path.join(knowledge_folder, f))
                            all_docs.extend(loader.load())
                    splits = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(all_docs)
                    st.session_state.vector_db = FAISS.from_documents(splits, embeddings)
                    st.success(f"📚 知识库就绪")
                except: st.warning("知识库加载略过")

    tab1, tab2, tab3, tab4 = st.tabs(["🏋️ 训练", "🍽️ 饮食", "📈 进步", "🤖 分析"])

    # === Tab 1: 训练记录 ===
    with tab1:
        st.subheader(f"🔥 {current_user} 的快速打卡")
        part_selected = st.pills("部位", list(GYM_MENU.keys()), default="胸", selection_mode="single")
        exercise_list = GYM_MENU.get(part_selected, ["自定义"])
        exercise_selected = st.pills("动作", exercise_list, default=exercise_list[0], selection_mode="single")
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1: w_weight = st.number_input("重量 (kg)", value=0.0, step=2.5)
        with c2: w_reps = st.number_input("次数", value=8, step=1)
        w_sets = st.pills("组数", [1, 2, 3, 4, 5], default=1, selection_mode="single")
        st.markdown("<br>", unsafe_allow_html=True) 
        if st.button("✅ 确认保存", use_container_width=True, type="primary"):
            c.execute("INSERT INTO workouts VALUES (?, ?, ?, ?, ?, ?, ?)", 
                      (current_user, str(datetime.date.today()), part_selected, exercise_selected, w_weight, w_reps, w_sets))
            conn.commit()
            st.success(f"已保存: {exercise_selected}")

    # === Tab 2: 饮食记录 ===
    with tab2:
        st.subheader("饮食")
        d_input = st.text_input("吃了什么？")
        c1, c2 = st.columns(2)
        if c1.button("记录"):
            c.execute("INSERT INTO diet VALUES (?, ?, ?, ?, ?, ?, ?)", 
                      (current_user, str(datetime.date.today()), d_input, 0, 0, 0, 0)) 
            conn.commit()
            st.success("已记录")
        if c2.button("AI 估算") and llm:
            with st.spinner("计算中..."):
                try:
                    res = llm.invoke(f"分析食物:'{d_input}'。返回格式:食物名,热量,蛋白,碳水,脂肪。例:面,300,10,60,5").content
                    item, cal, prot, carb, fat = res.replace("`","").strip().split(',')
                    c.execute("INSERT INTO diet VALUES (?, ?, ?, ?, ?, ?, ?)", 
                              (current_user, str(datetime.date.today()), item, float(cal), float(prot), float(carb), float(fat)))
                    conn.commit()
                    st.success(f"已记录: {item}")
                except: st.error("AI 解析失败")

    # === Tab 3: 进步可视化 (核心修改) ===
    with tab3:
        st.subheader("📈 见证你的变强之路")
        
        # 1. 获取该用户练过的所有动作
        df_all = pd.read_sql_query("SELECT DISTINCT exercise FROM workouts WHERE username = ?", conn, params=(current_user,))
        
        if df_all.empty:
            st.info("👋 你还没有训练记录，快去 Tab 1 打卡第一次训练吧！")
        else:
            # 2. 动作选择器
            exercise_list = df_all['exercise'].tolist()
            target_exercise = st.selectbox("请选择要查看的动作", exercise_list)
            
            # 3. 获取该动作的历史数据 (只取每天的最大重量作为代表)
            query = """
                SELECT date, MAX(weight) as max_weight 
                FROM workouts 
                WHERE username = ? AND exercise = ? 
                GROUP BY date 
                ORDER BY date ASC
            """
            df_hist = pd.read_sql_query(query, conn, params=(current_user, target_exercise))
            
            if not df_hist.empty:
                # 4. 数据计算与文案生成
                latest_weight = df_hist.iloc[-1]['max_weight']  # 当前重量
                start_weight = df_hist.iloc[0]['max_weight']    # 初始重量
                
                # 计算长期变化
                total_growth = latest_weight - start_weight
                if start_weight > 0:
                    growth_pct = int((total_growth / start_weight) * 100)
                else:
                    growth_pct = 0
                
                # 计算短期变化 (和上一次比)
                if len(df_hist) >= 2:
                    prev_weight = df_hist.iloc[-2]['max_weight']
                    short_change = latest_weight - prev_weight
                else:
                    prev_weight = latest_weight
                    short_change = 0
                
                # 生成激励文案
                long_term_msg = f"比初始提升了 {growth_pct}%" if growth_pct > 0 else "保持初心"
                
                short_term_msg = ""
                if short_change > 0:
                    short_term_msg = f"，比上一次增加了 {short_change}kg 🔥"
                elif short_change == 0:
                    short_term_msg = "，与上次持平 🛡️"
                else:
                    short_term_msg = f"，调整状态 ({short_change}kg) 💤"

                final_msg = f"**{target_exercise}** {long_term_msg}{short_term_msg}"

                # 5. 绘制 Altair 漂亮图表 (带交互)
                chart = alt.Chart(df_hist).mark_line(point=True).encode(
                    x=alt.X('date', title='训练日期'),
                    y=alt.Y('max_weight', title='重量 (kg)', scale=alt.Scale(zero=False)),
                    tooltip=['date', 'max_weight']
                ).properties(
                    height=300
                ).interactive()

                st.altair_chart(chart, use_container_width=True)
                
                # 6. 显示简洁有力的结果
                st.info(final_msg)
            else:
                st.warning("暂无数据")

    # === Tab 4: AI 分析 ===
    with tab4:
        st.subheader("教练点评")
        if st.button("生成报告") and llm:
            user_data = pd.read_sql_query("SELECT * FROM workouts WHERE username = ?", conn, params=(current_user,)).to_string()
            with st.spinner("分析中..."):
                try:
                    prompt = f"基于数据:\n{user_data}\n\n给出专业简短的训练建议。"
                    if st.session_state.vector_db:
                        retriever = st.session_state.vector_db.as_retriever()
                        chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, ChatPromptTemplate.from_messages([("system", "基于书籍:{context}\n分析:{input}")])))
                        st.markdown(chain.invoke({"input": user_data})["answer"])
                    else:
                        st.markdown(llm.invoke(prompt).content)
                except Exception as e: st.error(f"失败: {e}")

if st.session_state['logged_in']: main_app()
else: login_page()
